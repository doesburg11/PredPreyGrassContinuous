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
import time

import numpy as np
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


def build_policies(mode: str, predator_count: int, prey_count: int):
    if mode == "il":
        policies = {f"predator_{i}" for i in range(predator_count)} | {
            f"prey_{j}" for j in range(prey_count)
        }

        def mapping_fn(agent_id, *args, **kwargs):
            return agent_id

    elif mode == "ps":
        policies = {"predator_policy", "prey_policy"}

        def mapping_fn(agent_id, *args, **kwargs):
            return f"{_species_of(agent_id)}_policy"

    else:
        raise ValueError(f"Unknown mode: {mode!r} (expected 'il' or 'ps')")
    return policies, mapping_fn


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


def _fov_degrees(value: str) -> int:
    parsed = int(value)
    if not 0 < parsed <= 360:
        raise argparse.ArgumentTypeError(
            f"must be a field of view in (0, 360] degrees, got {parsed}"
        )
    return parsed


def main():
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
    parser.add_argument("--iterations", type=_positive_int, default=5)
    parser.add_argument("--num-env-runners", type=int, default=0)
    parser.add_argument(
        "--reward-scale",
        type=float,
        default=1.0,
        help="Multiply every reward by this during training (e.g. 0.01 so the "
        "-1000 prey punishment fits PPO's value-loss clipping). eval.py always "
        "reports unscaled rewards.",
    )
    parser.add_argument(
        "--entropy-coeff",
        type=_non_negative_float,
        default=0.0,
        help="Weight of the entropy bonus in PPO's loss (RLlib's default is "
        "0.0, i.e. no pressure against a policy collapsing to a near-constant "
        "action -- see runs/ps_1000's prey_policy). Try e.g. 0.01.",
    )
    parser.add_argument("--train-batch-size", type=_positive_int, default=4000)
    parser.add_argument("--minibatch-size", type=_positive_int, default=128)
    parser.add_argument("--num-epochs", type=_positive_int, default=30)
    parser.add_argument(
        "--vf-share-layers",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Share the encoder trunk between the policy and value function "
        "(RLlib's default). If the critic isn't learning (vf_explained_var "
        "stuck near 0), --no-vf-share-layers stops it from also corrupting "
        "the actor's features.",
    )
    parser.add_argument(
        "--grad-clip",
        type=_positive_float,
        default=None,
        help="Clip the global gradient norm to this value each update "
        "(RLlib's default is no clipping).",
    )
    parser.add_argument(
        "--num-learners",
        type=int,
        default=0,
        help="Remote learner actors (0 = learn inside the driver process).",
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
    args = parser.parse_args()
    if args.num_env_runners < 0:
        parser.error("--num-env-runners must be >= 0")

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
        .env_runners(num_env_runners=args.num_env_runners)
        .training(
            train_batch_size_per_learner=args.train_batch_size,
            minibatch_size=args.minibatch_size,
            num_epochs=args.num_epochs,
            entropy_coeff=args.entropy_coeff,
            grad_clip=args.grad_clip,
            grad_clip_by="global_norm",
        )
        .learners(num_learners=args.num_learners)
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
                f"{time.time() - t0:.1f}s"
            )
        if args.checkpoint_dir:
            result = algo.save(args.checkpoint_dir)
            print(f"checkpoint saved to {result.checkpoint.path}")
    finally:
        if writer is not None:
            writer.close()
        algo.stop()


if __name__ == "__main__":
    main()
