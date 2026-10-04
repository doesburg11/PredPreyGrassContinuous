"""Train Aquarium agents using the environment and PPO configuration files.

Run from the repository root: .conda/bin/python add_grass/train.py
"""

import math
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from zoneinfo import ZoneInfo

from config.config_env import config_env
from config.config_ppo import config_ppo

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


def load_settings():
    """Copy and validate editable settings without modifying the source dictionaries."""
    env = dict(config_env)
    ppo = dict(config_ppo)
    for settings, names in (
        (
            env,
            (
                "predator_count",
                "prey_count",
                "max_time_steps",
                "action_repeat",
                "obs_stack",
            ),
        ),
        (ppo, ("iterations", "train_batch_size", "minibatch_size", "num_epochs")),
    ):
        for name in names:
            value = settings[name]
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
    for name in ("num_env_runners", "num_learners", "checkpoint_every"):
        if type(ppo[name]) is not int or ppo[name] < 0:
            raise ValueError(f"{name} must be a nonnegative integer")
    for settings, name, positive in (
        (env, "reward_scale", True),
        (env, "predator_shaping", False),
        (ppo, "lr", True),
        (ppo, "vf_clip_param", True),
        (ppo, "entropy_coeff", False),
        (ppo, "num_gpus_per_learner", False),
        (ppo, "grad_clip", True),
    ):
        value = settings[name]
        if name == "grad_clip" and value is None:
            continue
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value < 0
            or (positive and value == 0)
        ):
            raise ValueError(
                f"{name} must be finite and {'positive' if positive else 'nonnegative'}"
            )
    for name in ("gamma", "gae_lambda"):
        value = ppo[name]
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or not 0 <= value <= 1
        ):
            raise ValueError(f"{name} must be finite and in [0, 1]")
    for name in ("prey_fov", "predator_fov"):
        if name in env and (type(env[name]) is not int or not 0 < env[name] <= 360):
            raise ValueError(f"{name} must be an integer in (0, 360]")
    if env["obs_mode"] not in ("aquarium", "egocentric"):
        raise ValueError("obs_mode must be aquarium or egocentric")
    if (
        type(env["keep_prey_count_constant"]) is not bool
        or type(ppo["vf_share_layers"]) is not bool
    ):
        raise ValueError(
            "keep_prey_count_constant and vf_share_layers must be booleans"
        )
    if ppo["mode"] not in ("il", "ps"):
        raise ValueError("mode must be il or ps")
    if ppo["checkpoint_every"] and not ppo["checkpoint_dir"]:
        raise ValueError("checkpoint_every requires checkpoint_dir")
    module_name = Path(__file__).resolve().parent.name
    timestamp = datetime.now(ZoneInfo("Europe/Amsterdam")).strftime(
        "%Y-%m-%d_%H-%M-%S_%f"
    )
    run_name = f"{module_name}_{timestamp}"
    for name in ("checkpoint_dir", "tensorboard_dir"):
        if ppo[name]:
            ppo[name] = os.path.abspath(
                os.path.expanduser(os.fspath(ppo[name]).replace("{run_name}", run_name))
            )
    env["shaping_gamma"] = ppo["gamma"]
    return env, SimpleNamespace(**ppo)


def main():
    env_config, ppo = load_settings()
    # Keep each sampler and the local learner from starting competing BLAS
    # thread pools. Ray workers inherit these variables when Ray starts.
    for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[variable] = "1"
    torch.set_num_threads(1)
    # Remote workers must be able to import env_wrapper and mapping functions
    # even when the script is launched from outside add_grass/.
    module_dir = str(Path(__file__).resolve().parent)
    python_path = os.environ.get("PYTHONPATH", "")
    os.environ["PYTHONPATH"] = os.pathsep.join(
        [module_dir] + ([python_path] if python_path else [])
    )
    register()

    policies, mapping_fn = build_policies(
        ppo.mode, env_config["predator_count"], env_config["prey_count"]
    )

    config = (
        PPOConfig()
        .environment(
            ENV_NAME,
            env_config=env_config,
        )
        .resources(num_cpus_for_main_process=1)
        .env_runners(
            num_env_runners=ppo.num_env_runners,
            num_cpus_per_env_runner=1,
        )
        .training(
            train_batch_size_per_learner=ppo.train_batch_size,
            minibatch_size=ppo.minibatch_size,
            num_epochs=ppo.num_epochs,
            lr=ppo.lr,
            lambda_=ppo.gae_lambda,
            gamma=ppo.gamma,
            vf_clip_param=ppo.vf_clip_param,
            entropy_coeff=ppo.entropy_coeff,
            grad_clip=ppo.grad_clip,
            grad_clip_by="global_norm",
        )
        .learners(
            num_learners=ppo.num_learners,
            num_gpus_per_learner=ppo.num_gpus_per_learner,
        )
        .multi_agent(policies=policies, policy_mapping_fn=mapping_fn)
        .rl_module(
            rl_module_spec=MultiRLModuleSpec(
                rl_module_specs={p: RLModuleSpec() for p in policies},
            ),
            model_config=DefaultModelConfig(vf_share_layers=ppo.vf_share_layers),
        )
    )

    algo = config.build_algo()
    writer = SummaryWriter(ppo.tensorboard_dir) if ppo.tensorboard_dir else None
    try:
        for i in range(ppo.iterations):
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
                f"iter {i + 1}/{ppo.iterations}  "
                f"episode_return_mean={reward_mean}  "
                f"num_episodes={num_episodes}  "
                f"module_returns={module_returns}  "
                f"{time.time() - t0:.1f}s",
                flush=True,
            )
            done = i + 1
            if (
                ppo.checkpoint_every
                and done % ppo.checkpoint_every == 0
                and done < ppo.iterations
            ):
                path = os.path.join(ppo.checkpoint_dir, f"iter_{done:06d}")
                algo.save(path)
                print(f"checkpoint saved to {path}", flush=True)
        if ppo.checkpoint_dir:
            algo.save(ppo.checkpoint_dir)
            print(f"checkpoint saved to {ppo.checkpoint_dir}", flush=True)
    finally:
        if writer is not None:
            writer.close()
        algo.stop()


if __name__ == "__main__":
    if len(sys.argv) > 1:
        raise SystemExit(
            "Training takes no command-line arguments. "
            "Edit config/config_env.py and config/config_ppo.py."
        )
    main()
