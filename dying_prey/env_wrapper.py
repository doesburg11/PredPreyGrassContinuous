"""Wraps the Aquarium PettingZoo ParallelEnv as an RLlib MultiAgentEnv.

Aquarium (Koelle et al. 2024,
https://github.com/michaelkoelle/marl-aquarium) is already a PettingZoo
ParallelEnv with Box observations and Discrete actions, so no custom
MultiAgentEnv subclass is needed -- RLlib's built-in ParallelPettingZooEnv
wrapper is sufficient. This module just registers it with Ray Tune's env
registry under a fixed name so RLlib configs can reference it by string.
"""

import math
import random
from collections import deque

import numpy as np
from gymnasium.spaces import Box, Dict
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


def _patch_torus_view(raw_env) -> None:
    """Fix Aquarium's view-cone check across torus edges.

    Torus.check_if_entity_is_in_view_in_torus (marl_aquarium/env/utils.py)
    is what both species use to decide whom they see. Two bugs:

    1. It tests the observer's cone shifted by each of the 8 torus offsets,
       but assigns is_in_view on every loop pass instead of OR-ing, so only
       the last offset (-width, -height) counts: an animal just across the
       left, right, top or bottom edge is never seen.
    2. The angle test is abs(angle_to_animal - heading) with both in
       [0, 360), without wrapping, so a cone straddling 0/360 degrees is
       cut off on one side.

    Replace it on this env's torus with the nearest-image offset (the
    wrapped dx, dy) and a wrapped angular difference. Uses Aquarium's own
    angle convention (-atan2(dy, dx) in degrees, as for orientation_angle).
    Only the nearest image is tested, which equals testing all 9 images as
    long as view_distance <= min(width, height) / 2 (100/200 vs 800 by
    default).
    """
    torus = getattr(raw_env, "torus", None)
    if torus is None or not all(
        hasattr(torus, name)
        for name in ("width", "height", "check_if_entity_is_in_view_in_torus")
    ):
        raise RuntimeError(
            "marl_aquarium internals changed (Torus); "
            "_patch_torus_view needs updating"
        )
    if getattr(torus, "_view_patched", False):
        return
    torus._view_patched = True

    def check_if_entity_is_in_view_in_torus(observer, animal, view_distance, fov):
        dx = animal.position.x - observer.position.x
        dy = animal.position.y - observer.position.y
        dx = (dx + torus.width / 2) % torus.width - torus.width / 2
        dy = (dy + torus.height / 2) % torus.height - torus.height / 2
        if math.hypot(dx, dy) > view_distance:
            return False
        angle = -math.degrees(math.atan2(dy, dx))
        diff = abs((angle - observer.orientation_angle + 180) % 360 - 180)
        return diff <= fov / 2

    torus.check_if_entity_is_in_view_in_torus = check_if_entity_is_in_view_in_torus


def _patch_predator_vision(raw_env) -> None:
    """Let predators see prey with their own view cone.

    1. Aquarium's predator_nearby_fish_observations checks visibility with
       prey_view_distance/prey_fov (100, 120 deg) instead of the predator's
       own predator_view_distance/predator_fov (200, 150 deg), so the
       predator_fov option had no effect on what predators observe (only on
       the rendered view cone), and predators almost never saw a prey.
    2. It also appends every prey in view without a cap, so more than
       prey_observe_count prey in view overflows the observation and trips
       the length assert in get_predator_observations. Keep the
       prey_observe_count nearest (by torus distance) instead.

    Only the fov_enabled path (the default, and all train.py uses) is
    replaced. The fov_enabled=False path is left as-is and is broken
    upstream: it takes the n nearest prey regardless of distance, and
    nearby_animal_observation's scale() asserts on any prey farther than
    the predator's view distance.

    Visibility goes through Torus.check_if_entity_is_in_view_in_torus,
    which _patch_torus_view fixes.
    """
    needed = (
        "predator_nearby_fish_observations",
        "nearby_animal_observation",
        "torus",
        "prey",
        "fov_enabled",
        "predator_view_distance",
        "predator_fov",
        "prey_observe_count",
        "obs_size",
    )
    missing = [name for name in needed if not hasattr(raw_env, name)]
    if missing:
        raise RuntimeError(
            f"marl_aquarium internals changed (missing {missing}); "
            "_patch_predator_vision needs updating"
        )
    if not raw_env.fov_enabled:
        return
    if getattr(raw_env, "_predator_vision_patched", False):
        return
    raw_env._predator_vision_patched = True
    torus = raw_env.torus

    def predator_nearby_fish_observations(observer):
        in_view = [
            fish
            for fish in raw_env.prey
            if torus.check_if_entity_is_in_view_in_torus(
                observer, fish, raw_env.predator_view_distance, raw_env.predator_fov
            )
        ]
        in_view.sort(
            key=lambda fish: torus.get_distance_in_torus(
                observer.position, fish.position
            )
        )
        observations = []
        for fish in in_view[: raw_env.prey_observe_count]:
            observations += raw_env.nearby_animal_observation(observer, fish)
        slots = raw_env.prey_observe_count * raw_env.obs_size
        observations += [0] * (slots - len(observations))
        return observations

    raw_env.predator_nearby_fish_observations = predator_nearby_fish_observations


OBS_MODES = ("aquarium", "egocentric")
_EGO_SELF_SIZE = 4  # own vx, vy (/ max speed), sin and cos of own heading
_EGO_SLOT_SIZE = 6  # seen flag, dx, dy, distance (/ view distance), vx, vy


def _patch_egocentric_obs(raw_env) -> int:
    """Replace Aquarium's observations with an egocentric encoding.

    Aquarium's own observations make the policy work hard for basic
    geometry: positions are absolute (meaningless on a torus), the bearing
    to another animal is an angle squashed into [0, 1] with a jump at
    -180/180 degrees (and computed wrongly across torus edges by
    Torus.get_direction_in_torus), an empty slot is all zeros, which also
    reads as "predator in the corner", and a prey's own entry can fill
    one of its "other prey" slots. Aquarium's actions are absolute
    directions, so this encoding stays in the absolute frame but makes
    everything relative to the observer:

    - self: own velocity / max speed (2) and sin/cos of own heading (2),
      the heading being what the view cone points along;
    - then predator_observe_count predator slots and prey_observe_count
      prey slots, each [seen, dx, dy, distance, vx, vy]: the wrapped
      (nearest-image) offset to the animal and the distance, both divided
      by the observer's view distance, and the animal's velocity / its
      max speed. Slots hold the nearest animals in the observer's view cone
      (own species' view distance and fov), never the observer itself;
      unused slots are all zeros, with seen = 0.

    All values are in [-1, 1]. Every agent gets the same layout. Requires
    fov_enabled=True (the default). Returns the observation size.
    """
    needed = (
        "get_obs",
        "predators",
        "prey",
        "torus",
        "predator_observe_count",
        "prey_observe_count",
        "predator_view_distance",
        "predator_fov",
        "prey_view_distance",
        "prey_fov",
    )
    missing = [name for name in needed if not hasattr(raw_env, name)]
    if missing:
        raise RuntimeError(
            f"marl_aquarium internals changed (missing {missing}); "
            "_patch_egocentric_obs needs updating"
        )
    size = _EGO_SELF_SIZE + _EGO_SLOT_SIZE * (
        raw_env.predator_observe_count + raw_env.prey_observe_count
    )
    if not getattr(raw_env, "fov_enabled", True):
        raise ValueError(
            "obs_mode='egocentric' always uses each species' view cone; "
            "fov_enabled=False is not supported with it"
        )
    if getattr(raw_env, "_egocentric_obs_patched", False):
        return size
    raw_env._egocentric_obs_patched = True
    torus = raw_env.torus
    space = Box(low=-1.0, high=1.0, shape=(size,), dtype=np.float32)

    def offset(observer, animal):
        dx = animal.position.x - observer.position.x
        dy = animal.position.y - observer.position.y
        dx = (dx + torus.width / 2) % torus.width - torus.width / 2
        dy = (dy + torus.height / 2) % torus.height - torus.height / 2
        return dx, dy

    def slots(observer, animals, count, view_distance, fov):
        seen = []
        for animal in animals:
            if animal is observer:
                continue
            if not torus.check_if_entity_is_in_view_in_torus(
                observer, animal, view_distance, fov
            ):
                continue
            dx, dy = offset(observer, animal)
            seen.append((math.hypot(dx, dy), dx, dy, animal))
        seen.sort(key=lambda entry: entry[0])
        values = []
        for distance, dx, dy, animal in seen[:count]:
            values += [
                1.0,
                dx / view_distance,
                dy / view_distance,
                distance / view_distance,
                animal.velocity.x / animal.max_speed,
                animal.velocity.y / animal.max_speed,
            ]
        return values + [0.0] * (_EGO_SLOT_SIZE * count - len(values))

    def observe(observer, view_distance, fov):
        heading = math.radians(observer.orientation_angle)
        values = [
            observer.velocity.x / observer.max_speed,
            observer.velocity.y / observer.max_speed,
            math.sin(heading),
            math.cos(heading),
        ]
        values += slots(
            observer,
            raw_env.predators,
            raw_env.predator_observe_count,
            view_distance,
            fov,
        )
        values += slots(
            observer, raw_env.prey, raw_env.prey_observe_count, view_distance, fov
        )
        obs = np.asarray(values, dtype=np.float32)
        # Aquarium caps speeds at max_speed, so only rounding can overshoot.
        if not np.all(np.abs(obs) <= 1.0 + 1e-4):
            raise RuntimeError(f"egocentric observation out of [-1, 1]: {obs}")
        return np.clip(obs, -1.0, 1.0)

    def get_obs():
        obs = {
            predator.id(): observe(
                predator, raw_env.predator_view_distance, raw_env.predator_fov
            )
            for predator in raw_env.predators
        }
        for prey in raw_env.prey:
            obs[prey.id()] = observe(
                prey, raw_env.prey_view_distance, raw_env.prey_fov
            )
        return obs

    raw_env.get_obs = get_obs
    raw_env.observation_space = lambda agent: space
    # step() pads agents missing from get_obs() (prey that died this step)
    # with a zero list of these lengths.
    raw_env.number_of_predator_observations = size
    raw_env.number_of_fish_observations = size
    return size


class SafeParallelPettingZooEnv(ParallelPettingZooEnv):
    """ParallelPettingZooEnv with Aquarium-specific fixes (see Codex
    review, 2026-09-18). Besides the ones below, __init__ applies
    _patch_torus_view and _patch_predator_vision, plus
    _patch_predator_catch_rewards (respawn mode) or _patch_prey_death
    (no-respawn mode), and _patch_egocentric_obs with obs_mode="egocentric".

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

    observation_space: Dict

    def __init__(
        self,
        env,
        reward_scale: float = 1.0,
        obs_mode: str = "aquarium",
        predator_shaping: float = 0.0,
        shaping_gamma: float = 0.99,
        action_repeat: int = 1,
        obs_stack: int = 1,
    ):
        if obs_stack < 1:
            raise ValueError(f"obs_stack must be >= 1, got {obs_stack}")
        if action_repeat < 1:
            raise ValueError(f"action_repeat must be >= 1, got {action_repeat}")
        if obs_mode not in OBS_MODES:
            raise ValueError(f"obs_mode must be one of {OBS_MODES}, got {obs_mode!r}")
        raw_env = env.aec_env.unwrapped
        # Patch before super().__init__, which resets the env and snapshots
        # the observation spaces.
        _patch_torus_view(raw_env)
        _patch_predator_vision(raw_env)
        _patch_predator_catch_rewards(raw_env)
        _patch_prey_death(raw_env)
        if obs_mode == "egocentric":
            _patch_egocentric_obs(raw_env)
        super().__init__(env)
        self.reward_scale = reward_scale
        self.predator_shaping = predator_shaping
        self.shaping_gamma = shaping_gamma
        self.action_repeat = action_repeat
        self._potentials = {}
        self.obs_stack = obs_stack
        self._frames = {}
        if obs_stack > 1:
            spaces: dict = {}
            for agent, space in self.observation_space.spaces.items():
                assert isinstance(space, Box)
                spaces[agent] = Box(
                    low=np.tile(space.low, obs_stack),
                    high=np.tile(space.high, obs_stack),
                    dtype=np.float32,
                )
            self.observation_space = Dict(spaces)
            self.observation_spaces = spaces

    def _stack(self, obs, reset=False):
        """Each agent's last obs_stack observations (one per decision),
        oldest first, concatenated. A new episode starts with obs_stack
        copies of its first observation (as gymnasium's FrameStack does)."""
        if self.obs_stack == 1:
            return obs
        if reset:
            self._frames = {}
        stacked = {}
        for agent, agent_obs in obs.items():
            agent_obs = np.asarray(agent_obs, dtype=np.float32)
            frames = self._frames.get(agent)
            if frames is None:
                frames = deque([agent_obs] * self.obs_stack, maxlen=self.obs_stack)
                self._frames[agent] = frames
            else:
                frames.append(agent_obs)
            stacked[agent] = np.concatenate(frames)
        return stacked

    def _predator_potentials(self):
        """Phi(s) = -predator_shaping * (torus distance to the nearest prey)
        / (half the arena diagonal), per predator; 0 once no prey is left
        (the terminal state of a no-respawn episode)."""
        raw_env = self.par_env.aec_env.unwrapped
        scale = math.hypot(raw_env.width, raw_env.height) / 2
        potentials = {}
        for predator in raw_env.predators:
            if raw_env.prey:
                nearest = min(
                    raw_env.torus.get_distance_in_torus(
                        predator.position, prey.position
                    )
                    for prey in raw_env.prey
                )
                potentials[predator.id()] = -self.predator_shaping * nearest / scale
            else:
                potentials[predator.id()] = 0.0
        return potentials

    def reset(self, *, seed=None, options=None):
        # Drop catch events left over from a step that raised mid-way.
        getattr(self.par_env.aec_env.unwrapped, "_pending_catches", []).clear()
        if seed is not None:
            random.seed(seed)
        obs, infos = super().reset(seed=seed, options=options)
        if self.predator_shaping:
            self._potentials = self._predator_potentials()
        return self._stack(obs, reset=True), infos

    def _env_step(self, action_dict):
        obs, rewards, terminateds, truncateds, infos = super().step(action_dict)
        raw_env = self.par_env.aec_env.unwrapped
        if not raw_env.keep_prey_count_constant and not raw_env.prey:
            terminateds = {agent: True for agent in terminateds}
            truncateds = {agent: False for agent in truncateds}
        return obs, rewards, terminateds, truncateds, infos

    def step(self, action_dict):
        # Apply each action for action_repeat env steps, summing rewards. An
        # agent that is done mid-way keeps its last obs/flags and drops out
        # of the remaining sub-steps; Aquarium's max_time_steps still counts
        # env steps.
        obs, rewards, terminateds, truncateds, infos = {}, {}, {}, {}, {}
        actions = dict(action_dict)
        for _ in range(self.action_repeat):
            step_obs, step_rewards, step_terms, step_truncs, step_infos = (
                self._env_step(actions)
            )
            obs.update(step_obs)
            for agent, reward in step_rewards.items():
                rewards[agent] = rewards.get(agent, 0.0) + reward
            terminateds.update(step_terms)
            truncateds.update(step_truncs)
            infos.update(step_infos)
            if step_terms.get("__all__") or step_truncs.get("__all__"):
                break
            actions = {
                agent: action
                for agent, action in actions.items()
                if not (step_terms.get(agent) or step_truncs.get(agent))
            }
            if not actions:
                break
        if self.predator_shaping:
            # Potential-based shaping (Ng, Harada & Russell 1999): adding
            # gamma * Phi(s') - Phi(s) leaves the optimal policy unchanged,
            # so the predator can't profit from hovering near prey.
            potentials = self._predator_potentials()
            for agent, potential in potentials.items():
                if agent in rewards:
                    rewards[agent] += (
                        self.shaping_gamma * potential
                        - self._potentials.get(agent, potential)
                    )
            self._potentials = potentials
        if self.reward_scale != 1.0:
            rewards = {agent: r * self.reward_scale for agent, r in rewards.items()}
        return self._stack(obs), rewards, terminateds, truncateds, infos

    def close(self):
        try:
            super().close()
        except SystemExit:
            pass


def make_env(env_config: dict) -> SafeParallelPettingZooEnv:
    """env_config keys are passed straight through to
    aquarium_v0.parallel_env(), except `reward_scale` (multiplies every reward,
    default 1.0; PPO's value-loss clipping copes badly with Aquarium's -1000
    prey punishment), `obs_mode` ("aquarium", the default, or "egocentric";
    see _patch_egocentric_obs), `predator_shaping` / `shaping_gamma`
    (potential-based reward for predators closing in on the nearest prey,
    off by default; shaping_gamma should match PPO's gamma, 0.99),
    `action_repeat` (apply each action for N env steps, summing rewards;
    default 1), `obs_stack` (give each agent its last N observations,
    one per decision, as a crude memory; default 1) and
    `procreate` which is rejected: with
    it enabled, Aquarium creates prey IDs absent from the initial
    possible_agents/observation-space snapshot that ParallelPettingZooEnv
    takes at construction time, and train.py's IL mode has no policy for an
    agent ID it can't enumerate ahead of time. Not supported here.
    """
    env_config = dict(env_config or {})
    reward_scale = env_config.pop("reward_scale", 1.0)
    obs_mode = env_config.pop("obs_mode", "aquarium")
    predator_shaping = env_config.pop("predator_shaping", 0.0)
    shaping_gamma = env_config.pop("shaping_gamma", 0.99)
    action_repeat = env_config.pop("action_repeat", 1)
    obs_stack = env_config.pop("obs_stack", 1)
    env_config.setdefault("render_mode", None)
    if env_config.get("procreate", False):
        raise ValueError(
            "procreate=True is not supported by this wrapper: Aquarium would "
            "create prey agent IDs not present in the environment's initial "
            "possible_agents/space snapshot, and train.py's IL mode has no "
            "policy for an agent ID it can't enumerate ahead of time."
        )
    return SafeParallelPettingZooEnv(
        aquarium_v0.parallel_env(**env_config),
        reward_scale=reward_scale,
        obs_mode=obs_mode,
        predator_shaping=predator_shaping,
        shaping_gamma=shaping_gamma,
        action_repeat=action_repeat,
        obs_stack=obs_stack,
    )


def register() -> None:
    register_env(ENV_NAME, make_env)
