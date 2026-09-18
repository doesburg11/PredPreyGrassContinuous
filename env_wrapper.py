"""Wraps the Aquarium PettingZoo ParallelEnv as an RLlib MultiAgentEnv.

Aquarium (Koelle et al. 2024, https://github.com/michaelkoelle/marl-aquarium) is
already a PettingZoo ParallelEnv with Box observations and Discrete actions, so no
custom MultiAgentEnv subclass is needed -- RLlib's built-in ParallelPettingZooEnv
wrapper is sufficient. This module just registers it with Ray Tune's env registry
under a fixed name so RLlib configs can reference it by string.
"""

import random

from marl_aquarium import aquarium_v0
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
from ray.tune.registry import register_env

ENV_NAME = "aquarium"


class SafeParallelPettingZooEnv(ParallelPettingZooEnv):
    """ParallelPettingZooEnv with two Aquarium-specific fixes (see Codex review,
    2026-09-18):

    1. Aquarium's own close() unconditionally calls sys.exit()
       (marl_aquarium/env/aquarium.py:320, apparently meant for a pygame render
       window, but fires unconditionally). RLlib calls close() during algo.stop()
       and interpreter shutdown; letting SystemExit propagate from there can kill
       the calling thread/process before cleanup finishes. Swallow it here.
    2. Aquarium accepts a `seed` in reset() but never uses it -- initialization
       draws from Python's process-global `random` module. Seeding that module
       here at least makes a *single* env's reset reproducible; it does NOT
       isolate multiple Aquarium envs sharing one process (e.g. num_env_runners
       with more than one env per runner), since they'd still share one global
       stream. Fixing that properly requires an upstream Aquarium change.
    """

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            random.seed(seed)
        return super().reset(seed=seed, options=options)

    def close(self):
        try:
            super().close()
        except SystemExit:
            pass


def make_env(env_config: dict) -> ParallelPettingZooEnv:
    """env_config keys are passed straight through to aquarium_v0.parallel_env(),
    except `procreate` which is rejected: with it enabled, Aquarium creates prey
    IDs absent from the initial possible_agents/observation-space snapshot that
    ParallelPettingZooEnv takes at construction time, and train.py's IL mode has
    no policy for an agent ID it can't enumerate ahead of time. Not supported here.
    """
    env_config = dict(env_config or {})
    env_config.setdefault("render_mode", None)
    if env_config.get("procreate", False):
        raise ValueError(
            "procreate=True is not supported by this wrapper: Aquarium would "
            "create prey agent IDs not present in the environment's initial "
            "possible_agents/space snapshot, and train.py's IL mode has no "
            "policy for an agent ID it can't enumerate ahead of time."
        )
    return SafeParallelPettingZooEnv(aquarium_v0.parallel_env(**env_config))


def register() -> None:
    register_env(ENV_NAME, make_env)
