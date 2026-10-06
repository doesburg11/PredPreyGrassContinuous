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


def positions(layer):
    return [(p.position.x, p.position.y) for p in layer.patches]


def test_clustered_placement_spreads_clusters_and_splits_patches(raw):
    layer = GrassLayer(raw, count=100, clustered=True, cluster_count=10)
    layer.reset(seed=1)
    assert len(layer.centres) == 10
    # Each centre sits in its own cell of a 4 x 3 grid over the arena.
    cells = {(int(c.x // 200), int(c.y // (800 / 3))) for c in layer.centres}
    assert len(cells) == 10
    sizes = [sum(p.cluster == k for p in layer.patches) for k in range(10)]
    assert sizes == [10] * 10
    for patch in layer.patches:
        assert patch.cluster is not None
        centre = layer.centres[patch.cluster]
        distance = raw.torus.get_distance_in_torus(patch.position, centre)
        assert distance < 5 * layer.cluster_spread
        assert 0 <= patch.position.x < 800 and 0 <= patch.position.y < 800
    layer.reset(seed=1)
    again = positions(layer)
    layer.reset(seed=2)
    assert positions(layer) != again


def test_default_regrowth_keeps_the_spot(raw):
    raw.prey = [prey("prey_0", 100, 100)]
    layer = GrassLayer(raw, count=1, respawn_delay=2)
    layer.reset(seed=0)
    layer.patches[0].position = Vector(100, 100)
    assert layer.advance() == {"prey_0": 1}
    assert positions(layer) == [(100, 100)]


def test_random_respawn_moves_the_patch_and_keeps_it_in_its_cluster(raw):
    raw.prey = []
    layer = GrassLayer(
        raw, count=10, clustered=True, cluster_count=2, random_respawn=True
    )
    layer.reset(seed=3)
    patch = layer.patches[0]
    start = (patch.position.x, patch.position.y)
    raw.prey = [prey("prey_0", *start)]
    assert layer.advance() == {"prey_0": 1}
    assert (patch.position.x, patch.position.y) != start
    assert not layer.available(patch)  # hidden until it regrows
    assert patch.cluster is not None
    centre = layer.centres[patch.cluster]
    assert raw.torus.get_distance_in_torus(patch.position, centre) < 5 * 48


def test_random_respawn_without_clusters_goes_anywhere(raw):
    layer = GrassLayer(raw, count=1, random_respawn=True, respawn_delay=1)
    layer.reset(seed=4)
    spots = set()
    for _ in range(20):
        patch = layer.patches[0]
        raw.prey = [prey("prey_0", patch.position.x, patch.position.y)]
        layer.advance()
        spots.add((round(patch.position.x), round(patch.position.y)))
        raw.prey = []
        layer.advance()  # regrows
    assert len(spots) > 15


@pytest.mark.parametrize(
    "settings",
    [
        {"clustered": 1},
        {"random_respawn": "yes"},
        {"cluster_count": 0},
        {"cluster_spread": 0},
    ],
)
def test_invalid_cluster_settings(raw, settings):
    with pytest.raises(ValueError):
        GrassLayer(raw, **settings)


def test_cluster_settings_reach_the_environment():
    env = make_env(
        {
            "obs_mode": "egocentric",
            "grass_count": 20,
            "grass_clustered": True,
            "grass_cluster_count": 4,
            "grass_random_respawn": True,
        }
    )
    try:
        env.reset(seed=0)
        layer = env.par_env.aec_env.unwrapped.grass
        assert layer.clustered and layer.random_respawn
        assert len(layer.centres) == 4
    finally:
        env.close()


def eat(layer, raw, patch):
    """One step in which a prey stands on this patch (and nowhere else)."""
    raw.prey = [prey("prey_0", patch.position.x, patch.position.y)]
    layer.advance()
    raw.prey = []


def test_dispersal_regrows_next_to_a_living_patch_of_the_cluster(raw):
    layer = GrassLayer(
        raw,
        count=4,
        clustered=True,
        cluster_count=1,
        respawn_delay=2,
        dispersal=True,
        dispersal_distance=5.0,
    )
    layer.reset(seed=0)
    eaten, *living = layer.patches
    # Push the living patches far from the eaten one, close together.
    for i, patch in enumerate(living):
        patch.position = Vector(600 + i, 600)
    raw.prey = [prey("prey_0", eaten.position.x, eaten.position.y)]
    layer.advance()
    assert eaten.regrow == "disperse" and not layer.available(eaten)
    raw.prey = []
    layer.advance()
    assert not layer.available(eaten)
    layer.advance()  # respawn_delay after eating: regrows next to a living patch
    assert layer.available(eaten) and eaten.regrow is None
    nearest = min(
        raw.torus.get_distance_in_torus(eaten.position, p.position) for p in living
    )
    assert nearest < 5 * 5.0


def test_dispersal_without_living_patches_uses_the_seed_bank(raw):
    layer = GrassLayer(
        raw, count=1, clustered=True, cluster_count=1, respawn_delay=2, dispersal=True
    )
    layer.reset(seed=0)
    patch = layer.patches[0]
    eat(layer, raw, patch)
    layer.advance()
    layer.advance()
    assert layer.available(patch)
    centre = layer.centres[0]
    assert raw.torus.get_distance_in_torus(patch.position, centre) < 5 * 48


def test_overgrazing_collapses_and_relocates_the_cluster(raw):
    layer = GrassLayer(
        raw,
        count=30,
        consume_radius=0.01,  # one patch per step, however close they are
        clustered=True,
        cluster_count=3,  # a 2 x 2 grid: one free cell to move to
        respawn_delay=1000,
        overgrazing=True,
        overgrazing_threshold=0.2,
        overgrazing_delay=50,
    )
    layer.reset(seed=5)
    members = [p for p in layer.patches if p.cluster == 0]
    old_cell = layer.cell_of(layer.centres[0])
    for patch in members[:7]:  # 3 of 10 left: above the threshold
        eat(layer, raw, patch)
    assert layer.collapsed == [False, False, False]
    eat(layer, raw, members[7])  # 2 of 10 left: collapse
    assert layer.collapsed == [True, False, False] and layer.total_collapses == 1
    assert not any(layer.available(p) for p in members)  # leftovers die back too
    new_cell = layer.cell_of(layer.centres[0])
    others = {layer.cell_of(layer.centres[k]) for k in (1, 2)}
    assert new_cell != old_cell and new_cell not in others
    # The other clusters are untouched.
    assert all(layer.available(p) for p in layer.patches if p.cluster != 0)
    for _ in range(49):
        layer.advance()
    assert not any(layer.available(p) for p in members)
    layer.advance()  # overgrazing_delay later: regrows around the new centre
    assert all(layer.available(p) for p in members)
    for patch in members:
        assert layer.cell_of(patch.position) == new_cell or (
            raw.torus.get_distance_in_torus(patch.position, layer.centres[0]) < 5 * 48
        )
    assert layer.collapsed == [False, False, False]  # recovered: can collapse again


def test_collapsed_cluster_does_not_collapse_again_before_recovering(raw):
    layer = GrassLayer(
        raw,
        count=5,
        consume_radius=0.01,
        clustered=True,
        cluster_count=1,
        respawn_delay=1000,
        overgrazing=True,
        overgrazing_threshold=0.5,
        overgrazing_delay=100,
    )
    layer.reset(seed=1)
    for patch in layer.patches[:3]:
        eat(layer, raw, patch)
    assert layer.total_collapses == 1
    for _ in range(50):
        layer.advance()
    assert layer.total_collapses == 1


@pytest.mark.parametrize(
    "settings",
    [
        {"dispersal": True},  # needs clustered
        {"overgrazing": True},  # needs clustered
        {"clustered": True, "dispersal": True, "random_respawn": True},
        {"clustered": True, "overgrazing": True, "overgrazing_threshold": 1.0},
        {"clustered": True, "overgrazing": True, "overgrazing_delay": 0},
        {"clustered": True, "dispersal": True, "dispersal_distance": 0},
        {"clustered": True, "dispersal": "yes"},
    ],
)
def test_invalid_dispersal_and_overgrazing_settings(raw, settings):
    with pytest.raises(ValueError):
        GrassLayer(raw, **settings)


def test_dispersal_and_overgrazing_run_in_the_environment():
    env = make_env(
        {
            "obs_mode": "egocentric",
            "prey_count": 12,
            "grass_count": 30,
            "grass_consume_radius": 40,
            "grass_respawn_delay": 20,
            "grass_clustered": True,
            "grass_cluster_count": 3,
            "grass_dispersal": True,
            "grass_overgrazing": True,
            "grass_overgrazing_delay": 30,
            "max_time_steps": 400,
        }
    )
    try:
        obs, _ = env.reset(seed=0)
        layer = env.par_env.aec_env.unwrapped.grass
        for _ in range(400):
            obs, _, terms, truncs, _ = env.step(
                {agent: env.action_space[agent].sample() for agent in obs}
            )
            if terms["__all__"] or truncs["__all__"]:
                break
        assert layer.total_consumed > 0
        for patch in layer.patches:
            assert 0 <= patch.position.x < 800 and 0 <= patch.position.y < 800
    finally:
        env.close()
