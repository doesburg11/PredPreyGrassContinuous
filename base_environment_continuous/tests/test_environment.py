"""Behaviour of the environment that the equivalence test does not reach."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

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


def test_watching_does_not_change_an_episode(monkeypatch):
    """The viewer only reads the state (SPEC.md, section 8)."""
    pytest.importorskip("pygame")
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    from compare import digest, random_actions, snapshot
    from render import Viewer

    results = []
    for watch in (False, True):
        env = PredPreyGrassContinuous()
        viewer = Viewer(env, window_size=None) if watch else None
        rng = np.random.default_rng(5)
        obs, _ = env.reset(seed=5)
        trace = []
        actions = random_actions(rng, obs)
        for _ in range(60):
            obs, rewards, terms, truncs, infos = env.step(actions)
            if viewer is not None:
                frame = viewer.draw()
                assert frame.shape == (env.height, 2 * env.width, 3)
            trace.append(digest((obs, rewards, terms, truncs, infos, snapshot(env))))
            actions = random_actions(rng, obs, terms, truncs)
        if viewer is not None:
            viewer.close()
        results.append(trace)
    assert results[0] == results[1]


def test_viewer_draws_narrow_cones_and_forgets_old_episodes(monkeypatch):
    pytest.importorskip("pygame")
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    from render import Viewer

    env = PredPreyGrassContinuous({"predator_fov": 1, "prey_fov": 360})
    viewer = Viewer(env, window_size=None)
    env.reset(seed=0)
    viewer.reset()
    viewer.draw()
    env.reset(seed=1)
    viewer.reset()
    viewer.draw()
    assert viewer.history == [(0, 6, 8)]
    viewer.close()


def test_checkpoint_settings_translate_from_add_walls(tmp_path):
    import pickle

    from checkpoints import checkpoint_env_config

    def saved(env_config):
        with open(tmp_path / "class_and_ctor_args.pkl", "wb") as f:
            pickle.dump({"ctor_args_and_kwargs": (({"env_config": env_config},), {})}, f)
        return tmp_path

    milestone = {"prey_max_velocity": 6.0, "walls_layout": None,
                 "predator_max_age": 10**9, "render_mode": None}
    assert checkpoint_env_config(saved(milestone)) == {"prey_max_velocity": 6.0}
    for change in ({"predator_max_age": 100}, {"walls_layout": "blocks"},
                   {"obs_mode": "aquarium"}):
        with pytest.raises(ValueError):
            checkpoint_env_config(saved(dict(milestone, **change)))
