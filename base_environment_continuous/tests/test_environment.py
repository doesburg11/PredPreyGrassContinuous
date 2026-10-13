"""Behaviour of the environment that the equivalence test does not reach."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from environment import PredPreyGrassContinuous  # noqa: E402
from geometry import Vector  # noqa: E402


def test_spaces_cover_the_id_pool():
    env = PredPreyGrassContinuous()
    assert len(env.possible_agents) == 4000
    assert env.get_agent_ids() == set(env.possible_agents)
    assert env.observation_spaces["prey_1999"].shape == (33,)
    assert env.action_spaces["predator_0"].n == 48


def test_reset_is_reproducible_and_in_bounds():
    env = PredPreyGrassContinuous()
    first, infos = env.reset(seed=7)
    again, _ = PredPreyGrassContinuous().reset(seed=7)
    assert sorted(first) == sorted(again) == sorted(infos)
    assert len(first) == 14
    for agent, obs in first.items():
        assert obs.dtype == np.float32 and obs.shape == (33,)
        assert np.array_equal(obs, again[agent])
        assert np.all(np.abs(obs) <= 1)


def test_missing_actions_raise_and_finished_agents_are_ignored():
    env = PredPreyGrassContinuous()
    obs, _ = env.reset(seed=0)
    with pytest.raises(ValueError):
        env.step({agent: 0 for agent in list(obs)[1:]})
    actions = {agent: 47 for agent in obs}  # everybody stops
    actions["prey_1999"] = 0  # not in the episode: ignored
    env.step(actions)
    with pytest.raises(ValueError):
        env.step({agent: 48 for agent in env.active})


def test_equally_near_animals_keep_list_order():
    """Slots are sorted by distance with a stable sort: animals at exactly
    the same distance keep the order of the animal list, as in add_walls."""
    env = PredPreyGrassContinuous({"prey_count": 3, "predator_count": 1})
    env.reset(seed=0)
    observer, left, right = env.prey
    for animal in env.prey + env.predators:
        animal.velocity = Vector(0.0, 0.0)
    env.predators[0].position = Vector(700.0, 700.0)
    observer.position = Vector(400.0, 400.0)
    observer.orientation = 90.0  # facing -y (up)
    left.position = Vector(350.0, 300.0)
    right.position = Vector(450.0, 300.0)
    for first, second, dx in ((left, right, -50.0), (right, left, 50.0)):
        env.prey = [observer, first, second]
        obs = env._observe(observer)
        slots = obs[4 + 6 :][:18].reshape(3, 6)
        assert slots[0][0] == 1 and slots[1][0] == 1 and slots[2][0] == 0
        assert slots[0][1] == np.float32(dx / 267)


@pytest.mark.parametrize(
    "setting",
    [
        {"unknown": 1},
        {"action_repeat": 0},
        {"prey_count": 101},
        {"reproduction_prey_threshold": 3.0},
        {"energy_prey_initial": 20.0},
        {"prey_view_distance": 401},
        {"energy_catch_efficiency_predator": 1.5},
        {"grass_dispersal": True, "grass_random_respawn": True},
    ],
)
def test_invalid_settings_are_rejected(setting):
    with pytest.raises(ValueError):
        PredPreyGrassContinuous(setting)
