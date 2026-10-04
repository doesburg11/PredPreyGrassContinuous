"""Grass lifecycle, toroidal visibility, rewards, and observation integration."""

from types import SimpleNamespace

import numpy as np
import pytest
from marl_aquarium.env.vector import Vector

from env_wrapper import _patch_torus_view, make_env
from grass import GrassLayer, GrassPatch


def prey(agent, x, y, alive=True):
    return SimpleNamespace(
        position=Vector(x, y), orientation_angle=0, alive=alive, id=lambda: agent
    )


@pytest.fixture
def raw():
    from marl_aquarium.env.utils import Torus

    value = SimpleNamespace(
        width=800,
        height=800,
        torus=Torus(800, 800),
        prey_view_distance=200,
        prey_fov=120,
        prey=[],
    )
    _patch_torus_view(value)
    return value


def test_consume_once_and_respawn_after_exact_delay(raw):
    raw.prey = [prey("prey_1", 100, 100), prey("prey_0", 100, 100)]
    layer = GrassLayer(raw, count=1, respawn_delay=3)
    layer.patches = [GrassPatch(Vector(100, 100))]
    assert layer.advance() == {"prey_0": 1}
    assert layer.advance() == {}
    assert layer.advance() == {}
    assert layer.advance() == {"prey_0": 1}
    assert layer.total_consumed == 2
    assert layer.patches[0].position.x == 100


def test_nearest_living_prey_gets_patch_across_edge(raw):
    raw.prey = [
        prey("prey_0", 799, 100, alive=False),
        prey("prey_1", 795, 100),
        prey("prey_2", 790, 100),
    ]
    layer = GrassLayer(raw, count=1)
    layer.patches = [GrassPatch(Vector(2, 100))]
    assert layer.advance() == {"prey_1": 1}


def test_nearest_visible_available_patch_encoding(raw):
    observer = prey("prey_0", 790, 100)
    layer = GrassLayer(raw, count=0)
    layer.patches = [
        GrassPatch(Vector(795, 100), ready_at=10),
        GrassPatch(Vector(2, 100)),
        GrassPatch(Vector(40, 100)),
        GrassPatch(Vector(780, 100)),
    ]
    np.testing.assert_allclose(layer.observation(observer), [1, 0.06, 0, 0.06])
    layer.patches = [GrassPatch(Vector(780, 100)), GrassPatch(Vector(300, 100))]
    np.testing.assert_array_equal(layer.observation(observer), np.zeros(4))


def test_seeded_reset_and_episode_reset(raw):
    layer = GrassLayer(raw, count=3)
    layer.reset(7)
    positions = [(p.position.x, p.position.y) for p in layer.patches]
    layer.tick = 90
    layer.total_consumed = 20
    layer.reset(7)
    assert positions == [(p.position.x, p.position.y) for p in layer.patches]
    assert layer.tick == layer.total_consumed == 0
    layer.reset(8)
    assert positions != [(p.position.x, p.position.y) for p in layer.patches]


@pytest.mark.parametrize(
    "settings",
    [
        {"count": -1},
        {"count": True},
        {"respawn_delay": 0},
        {"respawn_delay": 1.5},
        {"consume_radius": 0},
        {"consume_radius": float("nan")},
        {"food_reward": -1},
        {"food_reward": float("inf")},
    ],
)
def test_invalid_grass_settings(raw, settings):
    with pytest.raises(ValueError):
        GrassLayer(raw, **settings)


def test_real_environment_food_reward_spaces_and_next_observation():
    env = make_env(
        {
            "predator_count": 1,
            "prey_count": 1,
            "obs_mode": "egocentric",
            "keep_prey_count_constant": False,
            "prey_view_distance": 200,
            "prey_fov": 360,
            "grass_count": 2,
            "grass_food_reward": 7,
            "grass_consume_radius": 20,
            "reward_scale": 0.1,
        }
    )
    try:
        obs, _ = env.reset(seed=7)
        raw = env.par_env.aec_env.unwrapped
        assert obs["prey_0"].shape == (32,)
        assert obs["predator_0"].shape == (28,)
        raw.prey[0].position = Vector(100, 100)
        raw.prey[0].velocity = Vector(0, 0)
        raw.predators[0].position = Vector(500, 500)
        raw.grass.patches = [GrassPatch(Vector(100, 100)), GrassPatch(Vector(105, 100))]
        obs, rewards, _, _, infos = env.step({"prey_0": 0, "predator_0": 0})
        assert rewards["prey_0"] == pytest.approx((1 + 2 * 7) * 0.1)
        assert infos["prey_0"]["grass_eaten"] == 2
        assert infos["prey_0"]["grass_eaten_total"] == 2
        np.testing.assert_array_equal(obs["prey_0"][-4:], np.zeros(4))
        for agent, value in obs.items():
            assert env.observation_space[agent].contains(value)
    finally:
        env.close()


def test_disabled_grass_keeps_ablation_observation_shape():
    env = make_env({"obs_mode": "egocentric", "grass_count": 0, "obs_stack": 2})
    try:
        obs, _ = env.reset(seed=3)
        assert obs["prey_0"].shape == (64,)
        np.testing.assert_array_equal(obs["prey_0"][-4:], np.zeros(4))
        assert env.par_env.aec_env.unwrapped.grass.patches == []
    finally:
        env.close()


def test_food_counts_accumulate_across_action_repeat():
    env = make_env(
        {
            "predator_count": 1,
            "prey_count": 1,
            "obs_mode": "egocentric",
            "grass_count": 1,
            "grass_respawn_delay": 1,
            "grass_consume_radius": 100,
            "action_repeat": 3,
        }
    )
    try:
        env.reset(seed=7)
        raw = env.par_env.aec_env.unwrapped
        raw.prey[0].position = Vector(100, 100)
        raw.predators[0].position = Vector(500, 500)
        raw.grass.patches = [GrassPatch(Vector(100, 100))]
        _, _, _, _, infos = env.step({"prey_0": 0, "predator_0": 0})
        assert raw.grass.tick == 3
        assert infos["prey_0"]["grass_eaten"] == 3
    finally:
        env.close()


def test_render_contains_available_grass_and_hides_consumed_patch(monkeypatch):
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    env = make_env(
        {"obs_mode": "egocentric", "grass_count": 1, "render_mode": "rgb_array"}
    )
    try:
        env.reset(seed=2)
        raw = env.par_env.aec_env.unwrapped
        raw.grass.patches = [GrassPatch(Vector(400, 400))]
        frame = env.par_env.render()
        visible_patch = frame[388:412, 388:412].copy()
        raw.grass.patches[0].ready_at = 10
        frame = env.par_env.render()
        assert not np.array_equal(visible_patch, frame[388:412, 388:412])
        # Transparent sprite corners preserve the arena background.
        np.testing.assert_array_equal(visible_patch[0, 0], frame[388, 388])
    finally:
        env.close()


def test_render_draws_grass_under_animals(monkeypatch):
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    env = make_env(
        {"obs_mode": "egocentric", "grass_count": 1, "render_mode": "rgb_array"}
    )
    try:
        env.reset(seed=2)
        raw = env.par_env.aec_env.unwrapped
        env.par_env.render()  # creates the view
        calls = []
        available = raw.grass.available
        draw_animal = raw.view.draw_animal
        monkeypatch.setattr(
            raw.grass, "available", lambda p: calls.append("grass") or available(p)
        )
        monkeypatch.setattr(
            raw.view,
            "draw_animal",
            lambda *a: calls.append("animal") or draw_animal(*a),
        )
        env.par_env.render()
        assert "grass" in calls and "animal" in calls
        assert calls.index("animal") > max(
            i for i, c in enumerate(calls) if c == "grass"
        )
    finally:
        env.close()
