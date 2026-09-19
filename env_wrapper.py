"""Wraps the Aquarium PettingZoo ParallelEnv as an RLlib MultiAgentEnv.

Aquarium (Koelle et al. 2024,
https://github.com/michaelkoelle/marl-aquarium) is already a PettingZoo
ParallelEnv with Box observations and Discrete actions, so no custom
MultiAgentEnv subclass is needed -- RLlib's built-in ParallelPettingZooEnv
wrapper is sufficient. This module just registers it with Ray Tune's env
registry under a fixed name so RLlib configs can reference it by string.
"""

import random

from marl_aquarium import aquarium_v0
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
from ray.tune.registry import register_env

ENV_NAME = "aquarium"


def _patch_predator_catch_rewards(raw_env) -> None:
    """Give predators the reward Aquarium meant to pay for a catch.

    With keep_prey_count_constant=True (the default), raw_env.update_prey
    respawns a caught prey (recently_died=True, alive stays True) instead of
    killing it, and raw_env.step only records a catch event when
    `not prey.alive`. So `catches` is always empty and get_predator_rewards
    always returns 0 -- predators get no learning signal (Aquarium even has
    a "TODO: Sharks now get no reward for eating fish" at that spot).

    Patches this one env instance: update_prey records the prey's position
    at the moment of collision (before the respawn teleports it), and
    get_rewards feeds those events to Aquarium's own predator-reward logic
    (predator_reward split among predators within catch_radius). Prey
    rewards and punishment are untouched.
    """
    needed = ("update_prey", "get_rewards", "keep_prey_count_constant")
    missing = [name for name in needed if not hasattr(raw_env, name)]
    if missing:
        raise RuntimeError(
            f"marl_aquarium internals changed (missing {missing}); "
            "_patch_predator_catch_rewards needs updating"
        )
    if not raw_env.keep_prey_count_constant:
        return  # prey really die, so Aquarium's own catch events work
    if hasattr(raw_env, "_pending_catches"):
        return  # already patched; wrapping twice would double the reward
    pending = raw_env._pending_catches = []
    original_update_prey = raw_env.update_prey
    original_get_rewards = raw_env.get_rewards

    def update_prey(prey, predators, desired_velocity):
        position = prey.position.copy()
        result = original_update_prey(prey, predators, desired_velocity)
        if prey.recently_died:
            pending.append({"killed": prey.id(), "position": position})
        return result

    def get_rewards(catches):
        all_catches = list(catches) + pending
        pending.clear()
        return original_get_rewards(all_catches)

    raw_env.update_prey = update_prey
    raw_env.get_rewards = get_rewards


class SafeParallelPettingZooEnv(ParallelPettingZooEnv):
    """ParallelPettingZooEnv with Aquarium-specific fixes (see Codex
    review, 2026-09-18). Besides the two below, __init__ applies
    _patch_predator_catch_rewards.

    1. Aquarium's own close() unconditionally calls sys.exit()
       (marl_aquarium/env/aquarium.py:320, apparently meant for a pygame
       render window, but fires unconditionally). RLlib calls close() during
       algo.stop() and interpreter shutdown; letting SystemExit propagate
       from there can kill the calling thread/process before cleanup
       finishes. Swallow it here.
    2. Aquarium accepts a `seed` in reset() but never uses it -- initialization
       draws from Python's process-global `random` module. Seeding that module
       here at least makes a *single* env's reset reproducible; it does NOT
       isolate multiple Aquarium envs sharing one process (e.g. num_env_runners
       with more than one env per runner), since they'd still share one global
       stream. Fixing that properly requires an upstream Aquarium change.
    """

    def __init__(self, env):
        super().__init__(env)
        _patch_predator_catch_rewards(env.aec_env.unwrapped)

    def reset(self, *, seed=None, options=None):
        # Drop catch events left over from a step that raised mid-way.
        getattr(self.par_env.aec_env.unwrapped, "_pending_catches", []).clear()
        if seed is not None:
            random.seed(seed)
        return super().reset(seed=seed, options=options)

    def close(self):
        try:
            super().close()
        except SystemExit:
            pass


def make_env(env_config: dict) -> ParallelPettingZooEnv:
    """env_config keys are passed straight through to
    aquarium_v0.parallel_env(), except `procreate` which is rejected: with
    it enabled, Aquarium creates prey IDs absent from the initial
    possible_agents/observation-space snapshot that ParallelPettingZooEnv
    takes at construction time, and train.py's IL mode has no policy for an
    agent ID it can't enumerate ahead of time. Not supported here.
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
