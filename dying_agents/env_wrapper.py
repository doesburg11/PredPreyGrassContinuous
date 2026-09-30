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


def _patch_prey_death(raw_env) -> None:
    """Make keep_prey_count_constant=False (prey die for good) usable.

    1. Aquarium's prey_observe pads its output to prey_count entries, using
       made-up "fish_<i>" IDs for every dead prey's slot. Those IDs are not
       agents, so PettingZoo's parallel/AEC conversion crashes with a
       KeyError on the first step after a death. Only observe live prey;
       Aquarium's step already fills a zero observation for the prey that
       died this step.
    2. Aquarium removes a caught prey from raw_env.prey before computing
       rewards, and get_prey_rewards only applies prey_punishment to a
       *respawned* prey (recently_died). So a prey that dies for good gets a
       final reward of 0 -- no death penalty at all. Give each prey killed
       this step -prey_punishment instead.
    3. Aquarium's step() loops over raw_env.all_entities while update_prey
       removes the dead prey from that same list, so the entity right after
       it is silently skipped that step (no movement, action ignored).
       Rebind all_entities to a copy before each prey update, so the removal
       hits the copy and step()'s loop keeps iterating the original.
    """
    needed = (
        "prey_observe",
        "get_prey_observations",
        "get_rewards",
        "update_prey",
        "all_entities",
        "prey_punishment",
        "keep_prey_count_constant",
    )
    missing = [name for name in needed if not hasattr(raw_env, name)]
    if missing:
        raise RuntimeError(
            f"marl_aquarium internals changed (missing {missing}); "
            "_patch_prey_death needs updating"
        )
    if raw_env.keep_prey_count_constant:
        return  # respawn mode: no prey ever dies
    if getattr(raw_env, "_prey_death_patched", False):
        return
    raw_env._prey_death_patched = True
    original_get_rewards = raw_env.get_rewards
    original_update_prey = raw_env.update_prey

    def update_prey(prey, predators, desired_velocity):
        raw_env.all_entities = list(raw_env.all_entities)
        return original_update_prey(prey, predators, desired_velocity)

    def prey_observe():
        return {
            prey.id(): raw_env.get_prey_observations(prey) for prey in raw_env.prey
        }

    def get_rewards(catches):
        rewards = original_get_rewards(catches)
        for catch in catches:
            rewards[catch["killed"]] = -raw_env.prey_punishment
        return rewards

    raw_env.update_prey = update_prey
    raw_env.prey_observe = prey_observe
    raw_env.get_rewards = get_rewards


class SafeParallelPettingZooEnv(ParallelPettingZooEnv):
    """ParallelPettingZooEnv with Aquarium-specific fixes (see Codex
    review, 2026-09-18). Besides the two below, __init__ applies
    _patch_predator_catch_rewards (respawn mode) or
    _patch_prey_death (no-respawn mode).

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
    3. In no-respawn mode, Aquarium ends the episode when the last prey dies
       by marking every agent *truncated*, which tells RLlib to bootstrap the
       predator's value as if the episode could have gone on. Nothing is left
       to catch, so report it as a termination instead.
    """

    def __init__(self, env, reward_scale: float = 1.0):
        super().__init__(env)
        self.reward_scale = reward_scale
        _patch_predator_catch_rewards(env.aec_env.unwrapped)
        _patch_prey_death(env.aec_env.unwrapped)

    def reset(self, *, seed=None, options=None):
        # Drop catch events left over from a step that raised mid-way.
        getattr(self.par_env.aec_env.unwrapped, "_pending_catches", []).clear()
        if seed is not None:
            random.seed(seed)
        return super().reset(seed=seed, options=options)

    def step(self, action_dict):
        obs, rewards, terminateds, truncateds, infos = super().step(action_dict)
        raw_env = self.par_env.aec_env.unwrapped
        if not raw_env.keep_prey_count_constant and not raw_env.prey:
            terminateds = {agent: True for agent in terminateds}
            truncateds = {agent: False for agent in truncateds}
        if self.reward_scale != 1.0:
            rewards = {agent: r * self.reward_scale for agent, r in rewards.items()}
        return obs, rewards, terminateds, truncateds, infos

    def close(self):
        try:
            super().close()
        except SystemExit:
            pass


def make_env(env_config: dict) -> ParallelPettingZooEnv:
    """env_config keys are passed straight through to
    aquarium_v0.parallel_env(), except `reward_scale` (multiplies every reward,
    default 1.0; PPO's value-loss clipping copes badly with Aquarium's -1000
    prey punishment) and `procreate` which is rejected: with
    it enabled, Aquarium creates prey IDs absent from the initial
    possible_agents/observation-space snapshot that ParallelPettingZooEnv
    takes at construction time, and train.py's IL mode has no policy for an
    agent ID it can't enumerate ahead of time. Not supported here.
    """
    env_config = dict(env_config or {})
    reward_scale = env_config.pop("reward_scale", 1.0)
    env_config.setdefault("render_mode", None)
    if env_config.get("procreate", False):
        raise ValueError(
            "procreate=True is not supported by this wrapper: Aquarium would "
            "create prey agent IDs not present in the environment's initial "
            "possible_agents/space snapshot, and train.py's IL mode has no "
            "policy for an agent ID it can't enumerate ahead of time."
        )
    return SafeParallelPettingZooEnv(
        aquarium_v0.parallel_env(**env_config), reward_scale=reward_scale
    )


def register() -> None:
    register_env(ENV_NAME, make_env)
