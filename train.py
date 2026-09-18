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

from env_wrapper import ENV_NAME, register
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.core.rl_module.default_model_config import DefaultModelConfig
from ray.rllib.core.rl_module.multi_rl_module import MultiRLModuleSpec
from ray.rllib.core.rl_module.rl_module import RLModuleSpec


def _species_of(agent_id: str) -> str:
    if agent_id.startswith("predator_"):
        return "predator"
    if agent_id.startswith("prey_"):
        return "prey"
    raise ValueError(
        f"Unrecognized agent id {agent_id!r}: expected 'predator_<i>' or "
        "'prey_<j>'. Aquarium's own agent-naming scheme may have changed."
    )


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["il", "ps"], default="ps")
    parser.add_argument("--predator-count", type=_positive_int, default=1)
    parser.add_argument("--prey-count", type=_positive_int, default=4)
    parser.add_argument("--max-time-steps", type=_positive_int, default=200)
    parser.add_argument("--iterations", type=_positive_int, default=5)
    parser.add_argument("--num-env-runners", type=int, default=0)
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

    policies, mapping_fn = build_policies(args.mode, args.predator_count, args.prey_count)

    config = (
        PPOConfig()
        .environment(
            ENV_NAME,
            env_config={
                "predator_count": args.predator_count,
                "prey_count": args.prey_count,
                "max_time_steps": args.max_time_steps,
                "render_mode": None,
            },
        )
        .env_runners(num_env_runners=args.num_env_runners)
        .multi_agent(policies=policies, policy_mapping_fn=mapping_fn)
        .rl_module(
            rl_module_spec=MultiRLModuleSpec(
                rl_module_specs={p: RLModuleSpec() for p in policies},
            ),
            model_config=DefaultModelConfig(vf_share_layers=True),
        )
    )

    algo = config.build_algo()
    try:
        for i in range(args.iterations):
            result = algo.train()
            env_runner_stats = result.get("env_runners", {})
            reward_mean = env_runner_stats.get("episode_return_mean")
            num_episodes = env_runner_stats.get("num_episodes")
            print(
                f"iter {i + 1}/{args.iterations}  "
                f"episode_return_mean={reward_mean}  "
                f"num_episodes={num_episodes}"
            )
        if args.checkpoint_dir:
            result = algo.save(args.checkpoint_dir)
            print(f"checkpoint saved to {result.checkpoint.path}")
    finally:
        algo.stop()


if __name__ == "__main__":
    main()
