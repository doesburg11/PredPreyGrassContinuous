"""Tests for the Aquarium vision patches in env_wrapper.py.

Run from the repo root with `.conda/bin/python -m pytest dying_agents`.
"""

import math
import random

import numpy as np
import pytest
from marl_aquarium.env.utils import Torus
from marl_aquarium.env.vector import Vector

from env_wrapper import _patch_torus_view, make_env

WIDTH = HEIGHT = 800
OBS_SIZE = 6
# Predator observation: 5 self values, 1 predator slot, then 3 prey slots.
PREY_SLOTS = slice(5 + OBS_SIZE, None)
DISTANCE = 3  # index of the scaled distance within a slot


class _Entity:
    def __init__(self, x, y, heading=0.0):
        self.position = Vector(x, y)
        self.orientation_angle = heading


class _RawEnv:
    def __init__(self):
        self.torus = Torus(WIDTH, HEIGHT)


@pytest.fixture
def old_torus():
    return Torus(WIDTH, HEIGHT)


@pytest.fixture
def new_torus():
    raw_env = _RawEnv()
    _patch_torus_view(raw_env)
    return raw_env.torus


def _reference_in_view(observer, animal, view_distance, fov):
    """Brute force: visible in any of the 9 torus images of the animal."""
    for ox in (-WIDTH, 0, WIDTH):
        for oy in (-HEIGHT, 0, HEIGHT):
            dx = animal.position.x + ox - observer.position.x
            dy = animal.position.y + oy - observer.position.y
            if math.hypot(dx, dy) > view_distance:
                continue
            angle = -math.degrees(math.atan2(dy, dx))
            diff = abs((angle - observer.orientation_angle + 180) % 360 - 180)
            if diff <= fov / 2:
                return True
    return False


def test_torus_view_matches_brute_force(new_torus):
    rng = random.Random(0)
    for _ in range(20000):
        observer = _Entity(
            rng.uniform(0, WIDTH),
            rng.uniform(0, HEIGHT),
            round(rng.uniform(-180, 180), 1),
        )
        animal = _Entity(rng.uniform(0, WIDTH), rng.uniform(0, HEIGHT))
        view_distance = rng.choice([100, 200])
        fov = rng.choice([120, 150, 280, 360])
        assert new_torus.check_if_entity_is_in_view_in_torus(
            observer, animal, view_distance, fov
        ) == _reference_in_view(observer, animal, view_distance, fov)


# Aquarium headings: 0 = +x, 90 = -y, 180/-180 = -x, -90 = +y.
@pytest.mark.parametrize(
    "observer, animal",
    [
        ((790, 400, 0), (10, 400)),  # across the right edge
        ((10, 400, 180), (790, 400)),  # across the left edge
        ((400, 10, 90), (400, 790)),  # across the top edge
        ((400, 790, -90), (400, 10)),  # across the bottom edge
        ((790, 790, -45), (10, 10)),  # across the bottom-right corner
    ],
)
def test_torus_view_sees_across_edges(old_torus, new_torus, observer, animal):
    observer, animal = _Entity(*observer), _Entity(*animal)
    assert new_torus.check_if_entity_is_in_view_in_torus(observer, animal, 200, 150)
    # Guard that the case really exercises the upstream bug.
    if observer.position.x != 790 or observer.position.y != 790:
        assert not old_torus.check_if_entity_is_in_view_in_torus(
            observer, animal, 200, 150
        )


def test_torus_view_cone_straddling_180(old_torus, new_torus):
    # Heading 179 (almost -x); the animal is at -177 degrees, 4 degrees away.
    observer, animal = _Entity(400, 400, 179), _Entity(300, 405)
    assert new_torus.check_if_entity_is_in_view_in_torus(observer, animal, 200, 150)
    assert not old_torus.check_if_entity_is_in_view_in_torus(
        observer, animal, 200, 150
    )


def test_torus_view_respects_distance_and_fov(new_torus):
    observer = _Entity(400, 400, 0)
    check = new_torus.check_if_entity_is_in_view_in_torus
    assert check(observer, _Entity(550, 400), 200, 150)
    assert not check(observer, _Entity(550, 400), 100, 150)  # too far
    assert not check(observer, _Entity(250, 400), 200, 150)  # behind
    assert check(observer, _Entity(250, 400), 200, 360)  # fov 360 sees behind


def _predator_env(**env_config):
    config = {"predator_count": 1, "prey_count": 5, "render_mode": None}
    config.update(env_config)
    env = make_env(config)
    raw_env = env.par_env.aec_env.unwrapped
    predator = raw_env.predators[0]
    predator.position = Vector(400, 400)
    predator.orientation_angle = 0  # facing +x
    for prey in raw_env.prey:
        prey.position = Vector(0, 0)  # ~566 away, out of every view
    return raw_env, predator


def _prey_slots(raw_env, predator):
    obs = np.asarray(raw_env.get_predator_observations(predator))
    assert obs.shape == (raw_env.number_of_predator_observations,)
    return obs[PREY_SLOTS].reshape(-1, OBS_SIZE)


def _at(distance, degrees):
    """Position `distance` from (400, 400) at an Aquarium heading."""
    rad = math.radians(degrees)
    return Vector(400 + distance * math.cos(rad), 400 - distance * math.sin(rad))


def test_predator_uses_its_own_view_distance():
    raw_env, predator = _predator_env()
    # 150 away: inside predator_view_distance (200), outside prey's (100).
    raw_env.prey[0].position = _at(150, 0)
    slots = _prey_slots(raw_env, predator)
    assert slots[0].any()
    assert slots[0, DISTANCE] == pytest.approx(150 / 200)


def test_predator_uses_its_own_fov():
    # 70 degrees off-axis: inside predator_fov/2 (75), outside prey_fov/2 (60).
    raw_env, predator = _predator_env()
    raw_env.prey[0].position = _at(80, 70)
    assert _prey_slots(raw_env, predator)[0].any()

    raw_env, predator = _predator_env(predator_fov=100)
    raw_env.prey[0].position = _at(80, 70)
    assert not _prey_slots(raw_env, predator).any()


def test_predator_keeps_nearest_prey_first_and_caps_slots():
    raw_env, predator = _predator_env(predator_fov=360)
    distances = [150, 30, 190, 90, 60]
    for prey, (distance, degrees) in zip(
        raw_env.prey, zip(distances, [0, 90, 180, -90, 45])
    ):
        prey.position = _at(distance, degrees)
    slots = _prey_slots(raw_env, predator)
    assert len(slots) == raw_env.prey_observe_count == 3
    np.testing.assert_allclose(
        slots[:, DISTANCE], np.array([30, 60, 90]) / 200, atol=1e-9
    )


def test_predator_pads_when_nothing_in_view():
    raw_env, predator = _predator_env()
    assert not _prey_slots(raw_env, predator).any()


@pytest.mark.parametrize("respawn", [True, False])
def test_rollout_keeps_observation_shapes(respawn):
    env = make_env(
        {
            "predator_count": 1,
            "prey_count": 8,
            "max_time_steps": 300,
            "predator_fov": 360,
            "keep_prey_count_constant": respawn,
        }
    )
    obs, _ = env.reset(seed=0)
    rng = np.random.default_rng(0)
    raw_env = env.par_env.aec_env.unwrapped
    for step in range(600):
        obs, _, terminateds, truncateds, _ = env.step(
            {agent: int(rng.integers(16)) for agent in obs}
        )
        for agent, agent_obs in obs.items():
            assert np.asarray(agent_obs).shape == (29,), agent
        if terminateds.get("__all__") or truncateds.get("__all__"):
            obs, _ = env.reset(seed=step)
    assert raw_env.torus._view_patched
    assert raw_env._predator_vision_patched
