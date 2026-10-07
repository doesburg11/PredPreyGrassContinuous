"""Walls: obstacles that block movement and sight, so animals can hide.

The continuous counterpart of PredPreyGrass's walls_occlusion experiments.
Walls are axis-aligned rectangles. They

- block movement: after every physics step, an animal overlapping a wall is
  pushed back out and loses its speed into the wall (it slides along it);
- block sight (occlusion): an animal or grass patch behind a wall is not
  seen. Visibility is decided between cells of a grid over the arena
  (visibility_cell units, 32 = one PredPreyGrass cell), from a table
  computed once per environment, so a check during the episode is a lookup.
  Within one cell everything is visible, so a wall narrower than a cell can
  leave a thin strip on its far side visible;
- are sensed: every agent observes `rays` distances to the nearest wall, in
  the absolute directions of the actions (0 = up, clockwise), each divided
  by its view distance (1.0: no wall within view).

The arena wraps around, and so do walls: a wall at an edge also blocks
animals and lines of sight that cross that edge.
"""

import math

import numpy as np
from gymnasium.spaces import Box

LAYOUTS = ("blocks", "chambers", "custom")
EPSILON = 1e-6  # Extra push-out distance, so rounding leaves no overlap.


def layout_rectangles(layout, width, height, thickness, custom=()):
    """Rectangles (x, y, w, h) for a layout.

    - blocks: a "forest" of 4 x 4 square blocks, each two wall thicknesses
      wide (64 units for thickness 32), evenly spread over the arena.
    - chambers: walls along the arena's edges (shared by opposite edges, as
      the arena wraps around) and a cross through the middle, making four
      rooms. Each room side has a doorway of a quarter of its length, in the
      middle.
    - custom: the given rectangles.
    """
    t = thickness
    if layout == "custom":
        return [tuple(float(v) for v in rect) for rect in custom]
    if layout == "blocks":
        size = 2 * t
        return [
            (
                (i + 0.5) * width / 4 - size / 2,
                (j + 0.5) * height / 4 - size / 2,
                size,
                size,
            )
            for i in range(4)
            for j in range(4)
        ]
    if layout == "chambers":
        rects = []
        for y in (0.0, height / 2 - t / 2):  # horizontal walls, doors per room
            half = width / 2
            door = half / 4
            for x0 in (0.0, half):
                left = (half - door) / 2
                rects.append((x0, y, left, t))
                rects.append((x0 + left + door, y, half - left - door, t))
        for x in (0.0, width / 2 - t / 2):  # vertical walls, doors per room
            half = height / 2
            door = half / 4
            for y0 in (0.0, half):
                top = (half - door) / 2
                rects.append((x, y0, t, top))
                rects.append((x, y0 + top + door, t, half - top - door))
        return rects
    raise ValueError(f"walls_layout must be one of {LAYOUTS}")


class Walls:
    """Wall geometry, collision, visibility table and ray sensing."""

    def __init__(
        self,
        raw_env,
        layout="blocks",
        thickness=32.0,
        rectangles=(),
        occlusion=True,
        rays=8,
        visibility_cell=32.0,
    ):
        for name, value, positive in (
            ("thickness", thickness, True),
            ("visibility_cell", visibility_cell, True),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value <= 0
            ):
                raise ValueError(f"walls_{name} must be finite and positive")
        if thickness < 16:
            # Animals move up to ~5 units per step; thinner walls could be
            # crossed between two collision checks.
            raise ValueError("walls_thickness must be at least 16")
        if type(occlusion) is not bool:
            raise ValueError("walls_occlusion must be True or False")
        if type(rays) is not int or rays < 0:
            raise ValueError("walls_rays must be a nonnegative integer")
        if layout == "custom" and not rectangles:
            raise ValueError('walls_layout "custom" needs walls_rectangles')
        self.env = raw_env
        self.width, self.height = raw_env.width, raw_env.height
        self.rects = np.asarray(
            layout_rectangles(layout, self.width, self.height, thickness, rectangles),
            dtype=np.float64,
        ).reshape(-1, 4)
        if np.any(self.rects[:, 2:] < 16):
            # Thinner walls could be crossed between two collision checks.
            raise ValueError("wall rectangles must be at least 16 units wide and high")
        self.occlusion = occlusion
        self.rays = rays
        self.cell = visibility_cell
        self.columns = max(1, round(self.width / visibility_cell))
        self.rows = max(1, round(self.height / visibility_cell))
        # Every wall and its copies one arena away, for wrapped geometry.
        shifts = [
            (dx, dy)
            for dx in (-self.width, 0.0, self.width)
            for dy in (-self.height, 0.0, self.height)
        ]
        self.images = np.concatenate(
            [self.rects + np.array([dx, dy, 0.0, 0.0]) for dx, dy in shifts]
        )
        self.visible = self.visibility_table() if occlusion else None

    # Geometry ---------------------------------------------------------------

    def inside(self, x, y, margin=0.0):
        """Whether a point lies in (or within margin of) a wall."""
        x0, y0, w, h = self.images.T
        return bool(
            np.any(
                (x > x0 - margin)
                & (x < x0 + w + margin)
                & (y > y0 - margin)
                & (y < y0 + h + margin)
            )
        )

    def _segment_hits(self, px, py, dx, dy):
        """Smallest fraction t in [0, 1] at which segments (px, py) + t (dx, dy)
        enter a wall (any image), inf where they enter none. Arrays broadcast."""
        x0, y0, w, h = (a[:, None] for a in self.images.T)
        px, py, dx, dy = (
            np.asarray(a, dtype=np.float64)[None, :] for a in (px, py, dx, dy)
        )
        with np.errstate(divide="ignore", invalid="ignore"):
            tx1, tx2 = (x0 - px) / dx, (x0 + w - px) / dx
            ty1, ty2 = (y0 - py) / dy, (y0 + h - py) / dy
        # A segment parallel to an axis is inside that slab or never.
        inside_x = (px > x0) & (px < x0 + w)
        inside_y = (py > y0) & (py < y0 + h)
        tx_lo = np.where(
            dx == 0, np.where(inside_x, -np.inf, np.inf), np.minimum(tx1, tx2)
        )
        tx_hi = np.where(
            dx == 0, np.where(inside_x, np.inf, -np.inf), np.maximum(tx1, tx2)
        )
        ty_lo = np.where(
            dy == 0, np.where(inside_y, -np.inf, np.inf), np.minimum(ty1, ty2)
        )
        ty_hi = np.where(
            dy == 0, np.where(inside_y, np.inf, -np.inf), np.maximum(ty1, ty2)
        )
        enter = np.maximum(tx_lo, ty_lo)
        leave = np.minimum(tx_hi, ty_hi)
        hit = (enter <= leave) & (leave >= 0) & (enter <= 1)
        return np.where(hit, np.maximum(enter, 0.0), np.inf).min(axis=0)

    def representatives(self):
        """One point per cell to judge its visibility from: the cell centre,
        or, if a wall covers it, the open point of the cell nearest to it (on
        a 10 x 10 sub-grid). Cells entirely inside walls keep their centre."""
        cells = self.columns * self.rows
        index = np.arange(cells)
        cell_w, cell_h = self.width / self.columns, self.height / self.rows
        cx = (index % self.columns + 0.5) * cell_w
        cy = (index // self.columns + 0.5) * cell_h
        offsets = np.linspace(-0.45, 0.45, 10)
        ox, oy = (a.ravel() for a in np.meshgrid(offsets * cell_w, offsets * cell_h))
        order = np.argsort(np.hypot(ox, oy))  # nearest to the centre first
        ox, oy = ox[order], oy[order]
        x0, y0, w, h = (self.images[:, i][None, :] for i in range(4))
        for cell in range(cells):
            px, py = cx[cell] + ox[:, None], cy[cell] + oy[:, None]
            m = 1.0  # strictly clear of walls, not on an edge
            covered = (
                (px > x0 - m) & (px < x0 + w + m) & (py > y0 - m) & (py < y0 + h + m)
            ).any(axis=1)
            centre_covered = self.inside(cx[cell], cy[cell], margin=1.0)
            if centre_covered and not covered.all():
                first_open = int(np.argmin(covered))
                cx[cell] += ox[first_open]
                cy[cell] += oy[first_open]
        return cx, cy

    def visibility_table(self):
        """visible[a, b]: whether cell b's representative point can be seen
        from cell a's along the shortest (wrapped) line, i.e. no wall lies on
        it. Representatives are open points (see representatives), so a
        target next to a wall is seen from its own side only."""
        cells = self.columns * self.rows
        cx, cy = self.representatives()
        visible = np.ones((cells, cells), dtype=bool)
        for a in range(cells):
            dx = (cx - cx[a] + self.width / 2) % self.width - self.width / 2
            dy = (cy - cy[a] + self.height / 2) % self.height - self.height / 2
            hits = self._segment_hits(
                np.full(cells, cx[a]), np.full(cells, cy[a]), dx, dy
            )
            visible[a] = ~np.isfinite(hits)
        np.fill_diagonal(visible, True)
        return visible

    def cell_of(self, x, y):
        column = int(x % self.width / self.width * self.columns) % self.columns
        row = int(y % self.height / self.height * self.rows) % self.rows
        return row * self.columns + column

    def can_see(self, observer, target):
        if self.visible is None:
            return True
        return bool(
            self.visible[
                self.cell_of(observer.position.x, observer.position.y),
                self.cell_of(target.position.x, target.position.y),
            ]
        )

    def ray_distances(self, animal, view_distance):
        """Distances to the nearest wall in the `rays` action directions, each
        divided by view_distance and capped at 1."""
        angles = np.radians(np.arange(self.rays) * 360.0 / self.rays)
        dx = np.sin(angles) * view_distance
        dy = -np.cos(angles) * view_distance
        hits = self._segment_hits(
            np.full(self.rays, animal.position.x),
            np.full(self.rays, animal.position.y),
            dx,
            dy,
        )
        return np.minimum(hits, 1.0).astype(np.float32)

    def overlapping(self, animals):
        """Indices of the animals that touch any wall (one NumPy pass)."""
        if not animals:
            return []
        position = np.array([[a.position.x, a.position.y] for a in animals])
        radius = np.array([a.radius for a in animals])[:, None]
        x0, y0, w, h = (self.rects[:, i][None, :] for i in range(4))
        cx, cy = x0 + w / 2, y0 + h / 2
        px = (position[:, :1] - cx + self.width / 2) % self.width - self.width / 2
        py = (position[:, 1:] - cy + self.height / 2) % self.height - self.height / 2
        ox = px - np.clip(px, -w / 2, w / 2)
        oy = py - np.clip(py, -h / 2, h / 2)
        touching = np.hypot(ox, oy) < radius
        return np.flatnonzero(touching.any(axis=1)).tolist()

    def push_out_all(self, animals):
        for index in self.overlapping(animals):
            self.push_out(animals[index])

    def push_out(self, animal):
        """Move an animal that overlaps walls back out, and remove its speed
        into them. A centre inside a wall leaves by the nearest side whose
        exit point is clear of every wall, so overlapping walls (corners,
        crossings) act as one shape. Repeats until nothing overlaps, at most
        8 times (a gap narrower than the animal cannot be resolved)."""
        radius = animal.radius
        for _ in range(8):
            moved = False
            for x0, y0, w, h in self.rects:
                cx, cy = x0 + w / 2, y0 + h / 2
                # Animal position relative to the wall centre, wrapped.
                px = (
                    animal.position.x - cx + self.width / 2
                ) % self.width - self.width / 2
                py = (
                    animal.position.y - cy + self.height / 2
                ) % self.height - self.height / 2
                qx = min(max(px, -w / 2), w / 2)
                qy = min(max(py, -h / 2), h / 2)
                ox, oy = px - qx, py - qy
                distance = math.hypot(ox, oy)
                if distance >= radius:
                    continue
                if distance > 0:
                    nx, ny = ox / distance, oy / distance
                    depth = radius - distance + EPSILON
                else:  # centre inside the wall
                    exits = sorted(
                        (
                            (w / 2 - px, (1.0, 0.0)),
                            (px + w / 2, (-1.0, 0.0)),
                            (h / 2 - py, (0.0, 1.0)),
                            (py + h / 2, (0.0, -1.0)),
                        ),
                    )
                    for gap, (nx, ny) in exits:
                        depth = gap + radius + EPSILON
                        if not self.inside(
                            animal.position.x + nx * depth,
                            animal.position.y + ny * depth,
                        ):
                            break
                    else:  # every side leads into another wall: nearest one
                        gap, (nx, ny) = exits[0]
                        depth = gap + radius + EPSILON
                animal.position.x = (animal.position.x + nx * depth) % self.width
                animal.position.y = (animal.position.y + ny * depth) % self.height
                into = animal.velocity.x * nx + animal.velocity.y * ny
                if into < 0:
                    animal.velocity.x -= into * nx
                    animal.velocity.y -= into * ny
                moved = True
            if not moved:
                return


def patch_walls(raw_env, **settings):
    """Attach walls to one Aquarium instance (egocentric observations) before
    RLlib snapshots its spaces: blocked movement, occluded sight and `rays`
    wall-distance inputs for every agent."""
    if not getattr(raw_env, "_egocentric_obs_patched", False):
        raise ValueError("Walls require obs_mode='egocentric'")
    walls = Walls(raw_env, **settings)
    fastest = max(raw_env.predator_max_velocity, raw_env.prey_max_velocity)
    smallest = min(raw_env.predator_radius, raw_env.prey_radius)
    if fastest >= smallest + walls.rects[:, 2:].min() / 2:
        # A step this long could carry an animal's centre past the middle of
        # the thinnest wall, and the push-out would then finish the crossing.
        raise ValueError(
            "animals move too fast for these walls: max velocity must stay "
            "below the smallest radius plus half the thinnest wall"
        )
    raw_env.walls = walls
    original_reset = raw_env.reset
    original_step = raw_env.step
    original_obs = raw_env.get_obs
    original_space = raw_env.observation_space
    original_render = raw_env.render
    torus = raw_env.torus
    original_in_view = torus.check_if_entity_is_in_view_in_torus

    def in_view(observer, target, view_distance, fov):
        return original_in_view(observer, target, view_distance, fov) and walls.can_see(
            observer, target
        )

    torus.check_if_entity_is_in_view_in_torus = in_view

    spaces = {}
    for species, agent in (("predator", "predator_0"), ("prey", "prey_0")):
        size = original_space(agent).shape[0] + walls.rays
        spaces[species] = Box(-1.0, 1.0, shape=(size,), dtype=np.float32)
    raw_env.number_of_predator_observations += walls.rays
    raw_env.number_of_fish_observations += walls.rays

    def push_all():
        walls.push_out_all(raw_env.predators + raw_env.prey)

    def get_obs():
        obs = original_obs()
        for animal in raw_env.predators + raw_env.prey:
            agent = animal.id()
            if agent not in obs:
                continue
            view = (
                raw_env.predator_view_distance
                if agent.startswith("predator")
                else raw_env.prey_view_distance
            )
            obs[agent] = np.concatenate(
                (np.asarray(obs[agent], np.float32), walls.ray_distances(animal, view))
            )
        return obs

    def reset(seed=None, options=None):
        _, infos = original_reset(seed=seed, options=options)
        push_all()  # Aquarium places animals without knowing about walls
        # Observe again after moving them, with every layer's inputs.
        return raw_env.get_obs(), infos

    def step(actions):
        obs, rewards, terms, truncs, infos = original_step(actions)
        push_all()
        # Aquarium observed before the walls moved anyone back; observe again
        # (cheap inside a repeated decision, where observations are skipped).
        obs.update(raw_env.get_obs())
        return obs, rewards, terms, truncs, infos

    hooked_view = None

    def render(mode=None):
        nonlocal hooked_view
        from grass import ensure_view

        view = ensure_view(raw_env)
        if view is not hooked_view:
            draw_background = view.draw_background

            def draw_background_and_walls():
                draw_background()
                draw_walls(view)

            view.draw_background = draw_background_and_walls
            hooked_view = view
        return original_render(mode)

    def draw_walls(view):
        import pygame

        for x0, y0, w, h in walls.images:
            if x0 + w < 0 or y0 + h < 0 or x0 > walls.width or y0 > walls.height:
                continue
            pygame.draw.rect(view.screen, (110, 100, 90), (x0, y0, w, h))

    raw_env.reset = reset
    raw_env.step = step
    raw_env.get_obs = get_obs
    raw_env.observation_space = lambda agent: (
        spaces["predator"] if agent.startswith("predator") else spaces["prey"]
    )
    raw_env.render = render
