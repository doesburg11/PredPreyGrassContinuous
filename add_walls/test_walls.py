"""Walls: layouts, blocked movement, occlusion, wall sensing, integration."""

from types import SimpleNamespace
import random

import numpy as np
import pytest
from marl_aquarium.env.vector import Vector

from env_wrapper import make_env
from walls import Walls, layout_rectangles


def arena():
    return SimpleNamespace(width=800, height=800)


def animal(x, y, vx=0.0, vy=0.0, radius=16):
    return SimpleNamespace(
        position=Vector(x, y), velocity=Vector(vx, vy), radius=radius
    )


ONE_WALL = [(380, 100, 40, 600)]  # a tall wall in the middle


def test_layouts():
    blocks = layout_rectangles("blocks", 800, 800, 32)
    assert len(blocks) == 16 and all(r[2] == r[3] == 64 for r in blocks)
    chambers = layout_rectangles("chambers", 800, 800, 32)
    assert len(chambers) == 16
    # Each chamber wall leaves a doorway: no wall covers the door's centre.
    walls = Walls(arena(), layout="chambers", occlusion=False)
    # The top wall's left half (x 0-400) has its doorway at x 150-250.
    assert not walls.inside(200, 16)
    assert walls.inside(100, 16) and walls.inside(300, 16)
    assert layout_rectangles("custom", 800, 800, 32, ONE_WALL) == [
        (380.0, 100.0, 40.0, 600.0)
    ]


@pytest.mark.parametrize(
    "settings",
    [
        {"layout": "maze"},
        {"layout": "custom"},  # needs rectangles
        {"thickness": 8},
        {"rays": -1},
        {"occlusion": "yes"},
        {"visibility_cell": 0},
        {"layout": "custom", "rectangles": [(0, 0, 0, 10)]},
    ],
)
def test_invalid_settings(settings):
    with pytest.raises(ValueError):
        Walls(arena(), **settings)


def test_animals_are_pushed_out_and_slide_along_walls():
    walls = Walls(arena(), layout="custom", rectangles=ONE_WALL, occlusion=False)
    # Moving right into the wall's left face, and down along it.
    body = animal(370, 300, vx=5, vy=3)
    walls.push_out(body)
    assert body.position.x == pytest.approx(380 - 16)  # touching, not inside
    assert body.velocity.x == 0 and body.velocity.y == 3  # slides
    # A centre inside the wall leaves by the nearest side.
    deep = animal(385, 300)
    walls.push_out(deep)
    assert deep.position.x == pytest.approx(380 - 16)
    assert walls.overlapping([animal(100, 100), animal(370, 300)]) == [1]


def test_walls_wrap_across_the_arena_edges():
    walls = Walls(
        arena(), layout="custom", rectangles=[(0, 300, 20, 200)], occlusion=False
    )
    body = animal(795, 400, vx=5)  # moving right, across the edge into the wall
    walls.push_out(body)
    assert body.position.x == pytest.approx(800 - 16)
    assert body.velocity.x == 0


def test_occlusion_hides_what_is_behind_a_wall():
    walls = Walls(arena(), layout="custom", rectangles=ONE_WALL)
    left, right = animal(200, 400), animal(600, 400)
    assert not walls.can_see(left, right)
    assert walls.can_see(left, animal(250, 400))  # same side
    assert walls.can_see(animal(200, 50), animal(600, 50))  # above the wall
    # The line across the arena edge avoids the wall.
    assert walls.can_see(animal(50, 400), animal(750, 400))


def test_targets_next_to_a_wall_stay_visible():
    # Grass has no body radius, so a patch can sit right next to a wall, in
    # a cell whose centre (368) lies inside the wall (360-400).
    walls = Walls(arena(), layout="custom", rectangles=[(360, 100, 40, 600)])
    patch = SimpleNamespace(position=Vector(358, 400))
    assert walls.cell_of(358, 400) == walls.cell_of(368, 400)
    assert walls.inside(368, 400) and not walls.inside(358, 400)
    assert walls.can_see(animal(200, 400), patch)
    assert not walls.can_see(animal(600, 400), patch)  # still hidden behind it


def test_ray_distances():
    walls = Walls(arena(), layout="custom", rectangles=ONE_WALL, rays=4)
    # Rays go up, right, down, left (action directions 0, 4, 8, 12).
    rays = walls.ray_distances(animal(280, 400), view_distance=200)
    np.testing.assert_allclose(rays, [1.0, 0.5, 1.0, 1.0])  # wall 100 to the right
    assert walls.ray_distances(animal(280, 400), view_distance=50)[1] == 1.0


def test_walls_in_the_environment():
    env = make_env(
        {
            "obs_mode": "egocentric",
            "predator_count": 3,
            "prey_count": 6,
            "grass_count": 40,
            "walls_layout": "blocks",
            "walls_rays": 8,
            "action_repeat": 4,
            "max_time_steps": 400,
        }
    )
    try:
        obs, _ = env.reset(seed=0)
        raw = env.par_env.aec_env.unwrapped
        walls = raw.walls
        assert env.get_observation_space("predator_0").shape == (28 + 8,)
        assert env.get_observation_space("prey_0").shape == (28 + 8 + 4,)
        rng = np.random.default_rng(0)
        for _ in range(60):
            for a in raw.predators + raw.prey:
                assert not walls.inside(a.position.x, a.position.y, margin=-1)
            for patch in raw.grass.patches:
                assert not walls.inside(patch.position.x, patch.position.y)
            for agent, value in obs.items():
                assert env.observation_space[agent].contains(
                    np.asarray(value, dtype=np.float32)
                )
            obs, _, terms, truncs, _ = env.step(
                {agent: int(rng.integers(16)) for agent in obs}
            )
            if terms["__all__"] or truncs["__all__"]:
                break
    finally:
        env.close()


def test_walls_off_by_default_and_ignored_when_off():
    env = make_env({"obs_mode": "egocentric", "walls_rays": 8})
    try:
        env.reset(seed=0)
        assert not hasattr(env.par_env.aec_env.unwrapped, "walls")
        assert env.get_observation_space("predator_0").shape == (28,)
    finally:
        env.close()


def test_render_draws_walls(monkeypatch):
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    env = make_env(
        {"obs_mode": "egocentric", "walls_layout": "blocks", "render_mode": "rgb_array"}
    )
    try:
        env.reset(seed=0)
        frame = env.par_env.render()
        wall_colour = (frame == np.array([110, 100, 90])).all(axis=-1)
        assert wall_colour.sum() > 16 * 64 * 64 * 0.5
    finally:
        env.close()


def test_overlapping_walls_do_not_trap_animals():
    # Two overlapping rectangles act as one wall from x 100 to 280.
    walls = Walls(
        arena(),
        layout="custom",
        rectangles=[(100, 100, 100, 100), (180, 100, 100, 100)],
        occlusion=False,
    )
    body = animal(190, 150)
    walls.push_out(body)
    assert not walls.overlapping([body])


def test_animals_never_stay_inside_chamber_walls():
    # Corners and the central cross are made of overlapping rectangles.
    walls = Walls(arena(), layout="chambers", occlusion=False)
    rng = np.random.default_rng(0)
    for x, y in rng.uniform(0, 800, size=(2000, 2)):
        body = animal(x, y)
        walls.push_out(body)
        assert not walls.overlapping([body]), (x, y)


def test_thin_custom_walls_are_rejected():
    with pytest.raises(ValueError, match="16 units"):
        Walls(arena(), layout="custom", rectangles=[(110, 0, 2, 800)])


def test_walls_reject_speeds_that_could_cross_them():
    config = {
        "obs_mode": "egocentric",
        "walls_layout": "custom",
        "walls_rectangles": [(380, 100, 16, 600)],
        "prey_max_velocity": 30,
    }
    with pytest.raises(ValueError, match="too fast"):
        make_env(config)


def test_observations_are_taken_after_walls_move_animals():
    env = make_env(
        {
            "obs_mode": "egocentric",
            "predator_count": 1,
            "prey_count": 1,
            "walls_layout": "custom",
            "walls_rectangles": [(380, 100, 40, 600)],
            "action_repeat": 1,
        }
    )
    try:
        obs, _ = env.reset(seed=0)
        raw = env.par_env.aec_env.unwrapped
        prey = raw.prey[0]
        prey.position = Vector(362, 400)
        prey.velocity = Vector(4, 0)  # running into the wall
        obs, *_ = env.step({agent: 4 for agent in obs})  # action 4: right
        np.testing.assert_allclose(obs["prey_0"], raw.get_obs()["prey_0"])
        assert obs["prey_0"][0] == pytest.approx(0.0)  # no speed into the wall
    finally:
        env.close()


@pytest.mark.parametrize("field", range(4))
@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_custom_rectangles_reject_nonfinite_values(field, value):
    rect = [380, 100, 40, 600]
    rect[field] = value
    with pytest.raises(ValueError, match="finite"):
        Walls(arena(), layout="custom", rectangles=[rect])


@pytest.mark.parametrize("size", [(801, 40), (40, 801)])
def test_custom_rectangles_reject_sizes_larger_than_arena(size):
    with pytest.raises(ValueError, match="larger than the arena"):
        Walls(arena(), layout="custom", rectangles=[(0, 0, *size)])


@pytest.mark.parametrize("offset", [-1600, 1600])
def test_custom_rectangles_wrap_consistently(offset):
    reference = Walls(arena(), layout="custom", rectangles=ONE_WALL)
    wrapped = Walls(arena(), layout="custom", rectangles=[(380 + offset, 100 + offset, 40, 600)])
    np.testing.assert_array_equal(wrapped.rects, reference.rects)
    np.testing.assert_array_equal(wrapped.visible, reference.visible)
    body = animal(370, 300, vx=5)
    assert wrapped.inside(390, 300)
    assert wrapped.overlapping([body]) == [0]
    np.testing.assert_array_equal(wrapped.ray_distances(body, 200), reference.ray_distances(body, 200))
    wrapped.push_out(body)
    assert body.position.x == pytest.approx(364)


def test_edge_crossing_custom_rectangle_wraps():
    walls = Walls(arena(), layout="custom", rectangles=[(790, 300, 40, 200)], occlusion=False)
    assert walls.inside(5, 400)
    body = animal(35, 400, vx=-5)
    assert walls.overlapping([body]) == [0]
    assert walls.ray_distances(body, 200)[6] == pytest.approx(5 / 200)
    walls.push_out(body)
    assert body.position.x == pytest.approx(46)


@pytest.mark.parametrize("rect,bounds", [
    ((100, 100, 40, 100), (100, 100, 180, 200)),
    ((790, 300, 40, 200), (0, 300, 80, 500)),
])
def test_sample_open_in_partially_blocked_bounded_region(rect, bounds):
    walls = Walls(arena(), layout="custom", rectangles=[rect], occlusion=False)
    rng = random.Random(0)
    for _ in range(100):
        x, y = walls.sample_open(rng, bounds)
        assert bounds[0] <= x <= bounds[2]
        assert bounds[1] <= y <= bounds[3]
        assert not walls.inside(x, y)


def test_sample_open_rejects_fully_blocked_bounded_region():
    walls = Walls(arena(), layout="custom", rectangles=ONE_WALL, occlusion=False)
    with pytest.raises(ValueError, match="no usable space"):
        walls.sample_open(random.Random(0), (385, 200, 415, 300))
