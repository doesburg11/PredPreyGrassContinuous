"""Train Aquarium predator/prey agents with RLlib's new API stack (RLModule + Learner).

Usage:
    python train.py --mode il --iterations 5
    python train.py --mode ps --predator-count 2 --prey-count 8 --iterations 50

--mode il  (independent learning): every agent gets its own policy -- extends the
           paper's IL condition to predators too (the paper only ever trained prey;
           Aquarium's action/observation/reward interfaces are symmetric between
           predator and prey, so nothing stops training both).
--mode ps  (parameter sharing): one shared policy per species (predator_policy,
           prey_policy) -- extends the paper's PS condition the same way.
"""

import argparse
import math
import os
import time
from pathlib import Path

# Set native thread limits before loading NumPy/PyTorch in CLI runs.
if __name__ == "__main__":
    for _variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[_variable] = "1"

import numpy as np
import torch
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.core.rl_module.default_model_config import DefaultModelConfig
from ray.rllib.core.rl_module.multi_rl_module import MultiRLModuleSpec
from ray.rllib.core.rl_module.rl_module import RLModuleSpec
from tensorboardX import SummaryWriter

from env_wrapper import ENV_NAME, register


def _species_of(agent_id: str) -> str:
    if agent_id.startswith("predator_"):
        return "predator"
    if agent_id.startswith("prey_"):
        return "prey"
    raise ValueError(
        f"Unrecognized agent id {agent_id!r}: expected 'predator_<i>' or "
        "'prey_<j>'. Aquarium's own agent-naming scheme may have changed."
    )


def _flatten_scalars(prefix: str, value, out: dict) -> None:
    """Collect every finite number in a nested RLlib result dict as 'a/b/c'."""
    if isinstance(value, dict):
        for key, sub in value.items():
            _flatten_scalars(f"{prefix}/{key}" if prefix else str(key), sub, out)
    elif (
        isinstance(value, (int, float, np.integer, np.floating))
        and not isinstance(value, bool)
        and math.isfinite(value)
    ):
        out[prefix] = float(value)


def _per_agent_mapping(agent_id, *args, **kwargs):
    return agent_id


def _per_species_mapping(agent_id, *args, **kwargs):
    return f"{_species_of(agent_id)}_policy"


def build_policies(mode: str, predator_count: int, prey_count: int):
    if mode == "il":
        policies = {f"predator_{i}" for i in range(predator_count)} | {
            f"prey_{j}" for j in range(prey_count)
        }
        return policies, _per_agent_mapping
    if mode == "ps":
        return {"predator_policy", "prey_policy"}, _per_species_mapping
    raise ValueError(f"Unknown mode: {mode!r} (expected 'il' or 'ps')")


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError(f"must be a positive integer, got {parsed}")
    return parsed


def _non_negative_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed < 0:
        raise argparse.ArgumentTypeError(
            f"must be a finite number >= 0, got {parsed}"
        )
    return parsed


def _positive_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed <= 0:
        raise argparse.ArgumentTypeError(
            f"must be a finite positive number, got {parsed}"
        )
    return parsed


def _unit_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or not 0 <= parsed <= 1:
        raise argparse.ArgumentTypeError("must be finite and in [0, 1]")
    return parsed


def _fov_degrees(value: str) -> int:
    parsed = int(value)
    if not 0 < parsed <= 360:
        raise argparse.ArgumentTypeError(
            f"must be a field of view in (0, 360] degrees, got {parsed}"
        )
    return parsed


def build_parser():
    """CLI defaults favor stable PPO updates; every setting is overridable."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["il", "ps"], default="ps")
    parser.add_argument("--predator-count", type=_positive_int, default=1)
    parser.add_argument("--prey-count", type=_positive_int, default=4)
    parser.add_argument("--max-time-steps", type=_positive_int, default=200)
    parser.add_argument(
        "--prey-fov",
        type=_fov_degrees,
        default=None,
        help="Prey field of view in degrees (Aquarium's default is 120). "
        "Omit to keep Aquarium's default.",
    )
    parser.add_argument(
        "--predator-fov",
        type=_fov_degrees,
        default=None,
        help="Predator field of view in degrees (Aquarium's default is 150).",
    )
    parser.add_argument(
        "--no-respawn",
        action="store_true",
        help="Caught prey die for good (terminated, removed from the episode) "
        "instead of respawning at a random position. Still punished with "
        "-prey_punishment on death.",
    )
    parser.add_argument(
        "--obs",
        choices=["aquarium", "egocentric"],
        default="egocentric",
        help="Observation encoding: Aquarium's own (absolute positions, "
        "bearings as angles) or egocentric (wrapped offsets and velocities "
        "relative to the observer, a seen flag per slot; see "
        "env_wrapper._patch_egocentric_obs). Checkpoints remember it.",
    )
    parser.add_argument(
        "--predator-shaping",
        type=_non_negative_float,
        default=0.0,
        help="Potential-based reward shaping for predators: Phi = -C * "
        "(distance to the nearest prey) / (half the arena diagonal), so each "
        "decision adds --gamma * Phi(next) - Phi(now) (before --reward-scale). The "
        "catch reward is 10; try C = 1. 0 (default) = off. eval.py reports "
        "unshaped rewards.",
    )
    parser.add_argument(
        "--action-repeat",
        type=_positive_int,
        default=1,
        help="Apply each chosen action for N env steps (rewards summed), so "
        "an episode of --max-time-steps env steps has about 1/N as many "
        "decisions. Aquarium's predators turn slowly, so single-step "
        "actions barely change anything.",
    )
    parser.add_argument("--iterations", type=_positive_int, default=5)
    parser.add_argument("--num-env-runners", type=int, default=16)
    parser.add_argument(
        "--reward-scale",
        type=_positive_float,
        default=0.01,
        help="Multiply every reward by this during training (e.g. 0.01 so the "
        "-1000 prey punishment fits PPO's value-loss clipping). eval.py always "
        "reports unscaled rewards.",
    )
    parser.add_argument(
        "--entropy-coeff",
        type=_non_negative_float,
        default=0.01,
        help="Weight of the entropy bonus in PPO's loss (RLlib's default is "
        "0.0, i.e. no pressure against a policy collapsing to a near-constant "
        "action -- see runs/ps_1000's prey_policy). Try e.g. 0.01.",
    )
    parser.add_argument(
        "--lr",
        type=_positive_float,
        default=3e-4,
        help="PPO learning rate (RLlib's default 5e-5 is small for these "
        "networks; try 3e-4).",
    )
    parser.add_argument(
        "--gae-lambda",
        type=_unit_float,
        default=0.95,
        help="GAE lambda (RLlib's default 1.0 means Monte Carlo advantages; "
        "0.95 is the common choice).",
    )
    parser.add_argument("--train-batch-size", type=_positive_int, default=4800)
    parser.add_argument("--minibatch-size", type=_positive_int, default=1024)
    parser.add_argument("--num-epochs", type=_positive_int, default=10)
    parser.add_argument(
        "--vf-share-layers",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Share the encoder trunk between the policy and value function "
        "(RLlib's default). If the critic isn't learning (vf_explained_var "
        "stuck near 0), --no-vf-share-layers stops it from also corrupting "
        "the actor's features.",
    )
    parser.add_argument(
        "--grad-clip",
        type=_positive_float,
        default=0.5,
        help="Clip the global gradient norm to this value each update "
        "(RLlib's default is no clipping).",
    )
    parser.add_argument(
        "--no-grad-clip", dest="grad_clip", action="store_const", const=None,
        help="Disable gradient clipping.",
    )
    parser.add_argument(
        "--num-learners",
        type=int,
        default=0,
        help="Remote learner actors (0 = learn inside the driver process).",
    )
    parser.add_argument(
        "--num-gpus-per-learner",
        type=_non_negative_float,
        default=1,
        help="GPUs for each learner (or for the driver's own learner when "
        "--num-learners is 0), e.g. 1 to run the PPO update on the GPU and "
        "leave the CPU to the env. Needs a CUDA build of torch.",
    )
    parser.add_argument(
        "--tensorboard-dir",
        default=None,
        help="Write every numeric metric RLlib reports each iteration here as "
        "TensorBoard scalars (view with `tensorboard --logdir <dir>`). If "
        "omitted, nothing is written.",
    )
    parser.add_argument(
        "--checkpoint-dir",
        default=None,
        help="Save an RLlib checkpoint here after training (load it with "
        "`eval.py --checkpoint <dir>` to visualize the trained agents). If "
        "omitted, no checkpoint is saved.",
    )
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=0,
        help="Also save a checkpoint every N iterations, to "
        "<checkpoint-dir>/iter_<NNNNNN> (0 = only the final one). Needs "
        "--checkpoint-dir.",
    )
    parser.add_argument(
        "--gamma", type=_unit_float, default=0.99,
        help="Discount per policy decision (also used for predator shaping). "
        "With action repeat, one decision spans multiple physics steps.",
    )
    parser.add_argument(
        "--vf-clip-param", type=_positive_float, default=1000.0,
        help="PPO squared value-error cap. With reward-scale 0.01 a death "
        "costs 10; the default RLlib cap of 10 can suppress critic gradients "
        "for such targets. Increase this for larger reward scales.",
    )
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    if args.num_env_runners < 0:
        parser.error("--num-env-runners must be >= 0")
    if args.checkpoint_every < 0:
        parser.error("--checkpoint-every must be >= 0")
    if args.checkpoint_every and not args.checkpoint_dir:
        parser.error("--checkpoint-every needs --checkpoint-dir")
    if args.num_gpus_per_learner < 0:
        parser.error("--num-gpus-per-learner must be >= 0")
    if args.checkpoint_dir:
        # algo.save() hands the path to pyarrow, which rejects relative paths
        # ("URI has empty scheme") -- only after training has finished.
        args.checkpoint_dir = os.path.abspath(args.checkpoint_dir)

    # Keep each sampler and the local learner from starting competing BLAS
    # thread pools. Ray workers inherit these variables when Ray starts.
    for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[variable] = "1"
    torch.set_num_threads(1)
    # Remote workers must be able to import env_wrapper and mapping functions
    # even when the script is launched from outside dying_agents/.
    module_dir = str(Path(__file__).resolve().parent)
    python_path = os.environ.get("PYTHONPATH", "")
    os.environ["PYTHONPATH"] = os.pathsep.join(
        [module_dir] + ([python_path] if python_path else [])
    )
    register()

    policies, mapping_fn = build_policies(
        args.mode, args.predator_count, args.prey_count
    )

    env_config = {
        "predator_count": args.predator_count,
        "prey_count": args.prey_count,
        "max_time_steps": args.max_time_steps,
        "render_mode": None,
        "reward_scale": args.reward_scale,
        "obs_mode": args.obs,
        "predator_shaping": args.predator_shaping,
        "shaping_gamma": args.gamma,
        "action_repeat": args.action_repeat,
    }
    if args.prey_fov is not None:
        env_config["prey_fov"] = args.prey_fov
    if args.predator_fov is not None:
        env_config["predator_fov"] = args.predator_fov
    if args.no_respawn:
        env_config["keep_prey_count_constant"] = False

    config = (
        PPOConfig()
        .environment(
            ENV_NAME,
            env_config=env_config,
        )
        .resources(num_cpus_for_main_process=1)
        .env_runners(
            num_env_runners=args.num_env_runners,
            num_cpus_per_env_runner=1,
        )
        .training(
            train_batch_size_per_learner=args.train_batch_size,
            minibatch_size=args.minibatch_size,
            num_epochs=args.num_epochs,
            lr=args.lr,
            lambda_=args.gae_lambda,
            gamma=args.gamma,
            vf_clip_param=args.vf_clip_param,
            entropy_coeff=args.entropy_coeff,
            grad_clip=args.grad_clip,
            grad_clip_by="global_norm",
        )
        .learners(
            num_learners=args.num_learners,
            num_gpus_per_learner=args.num_gpus_per_learner,
        )
        .multi_agent(policies=policies, policy_mapping_fn=mapping_fn)
        .rl_module(
            rl_module_spec=MultiRLModuleSpec(
                rl_module_specs={p: RLModuleSpec() for p in policies},
            ),
            model_config=DefaultModelConfig(vf_share_layers=args.vf_share_layers),
        )
    )

    algo = config.build_algo()
    writer = SummaryWriter(args.tensorboard_dir) if args.tensorboard_dir else None
    try:
        for i in range(args.iterations):
            t0 = time.time()
            result = algo.train()
            env_runner_stats = result.get("env_runners", {})
            reward_mean = env_runner_stats.get("episode_return_mean")
            num_episodes = env_runner_stats.get("num_episodes")
            module_means = env_runner_stats.get("module_episode_returns_mean", {})
            module_returns = {k: round(v, 1) for k, v in module_means.items()}
            if writer is not None:
                scalars = {}
                _flatten_scalars("", result, scalars)
                for tag, scalar in scalars.items():
                    writer.add_scalar(tag, scalar, i + 1)
                writer.flush()
            print(
                f"iter {i + 1}/{args.iterations}  "
                f"episode_return_mean={reward_mean}  "
                f"num_episodes={num_episodes}  "
                f"module_returns={module_returns}  "
                f"{time.time() - t0:.1f}s",
                flush=True,
            )
            done = i + 1
            if (
                args.checkpoint_every
                and done % args.checkpoint_every == 0
                and done < args.iterations
            ):
                path = os.path.join(args.checkpoint_dir, f"iter_{done:06d}")
                algo.save(path)
                print(f"checkpoint saved to {path}", flush=True)
        if args.checkpoint_dir:
            algo.save(args.checkpoint_dir)
            print(f"checkpoint saved to {args.checkpoint_dir}", flush=True)
    finally:
        if writer is not None:
            writer.close()
        algo.stop()


if __name__ == "__main__":
    main()
