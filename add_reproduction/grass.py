"""Renewable food patches: placed at random or in clusters, regrowing in
place or at a new random spot."""

import math
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from gymnasium.spaces import Box
from marl_aquarium.env.vector import Vector


@dataclass
class GrassPatch:
    position: Vector
    ready_at: int = 0
    cluster: int | None = None  # Index into GrassLayer.centres when clustered.
    # Where the patch regrows, chosen when it does: None keeps its spot,
    # "disperse" near a living patch of its cluster, "centre" around the
    # cluster's centre.
    regrow: str | None = None


class GrassLayer:
    """Grass patches with their own seeded RNG and physics-step clock.

    Placement at reset: uniform over the arena, or (clustered) in
    cluster_count clusters whose centres each get their own cell of a grid
    over the arena, with patches split evenly over the clusters and scattered
    around each centre (normal offsets with sd cluster_spread, wrapped across
    the edges). An eaten patch regrows after respawn_delay physics steps: in
    the same spot, or (random_respawn) at a new random spot, within its own
    cluster when clustered, so clusters persist.

    Two clustered-only dynamics let the prey move the grass:

    - dispersal (seed dispersal): an eaten patch regrows next to a living
      patch of its cluster, chosen at random when it regrows (normal offsets
      with sd dispersal_distance), or around the cluster's centre if none is
      left. Clusters creep away from where grazing removes their edge.
    - overgrazing: when a cluster is grazed down to overgrazing_threshold or
      less of its patches, it collapses: its remaining patches die back, its
      centre moves to a grid cell no other cluster uses, and after
      overgrazing_delay physics steps the whole cluster regrows around the
      new centre. It cannot collapse again until it has recovered above the
      threshold.
    """

    def __init__(
        self,
        raw_env,
        count=24,
        consume_radius=12.0,
        food_reward=10.0,
        respawn_delay=100,
        clustered=False,
        cluster_count=10,
        cluster_spread=48.0,
        random_respawn=False,
        dispersal=False,
        dispersal_distance=24.0,
        overgrazing=False,
        overgrazing_threshold=0.0,
        overgrazing_delay=400,
    ):
        for name, value, minimum in (
            ("count", count, 0),
            ("respawn_delay", respawn_delay, 1),
            ("cluster_count", cluster_count, 1),
            ("overgrazing_delay", overgrazing_delay, 1),
        ):
            if type(value) is not int or value < minimum:
                raise ValueError(f"grass_{name} must be an integer >= {minimum}")
        for name, value in (
            ("clustered", clustered),
            ("random_respawn", random_respawn),
            ("dispersal", dispersal),
            ("overgrazing", overgrazing),
        ):
            if type(value) is not bool:
                raise ValueError(f"grass_{name} must be True or False")
        for name, value, positive in (
            ("consume_radius", consume_radius, True),
            ("food_reward", food_reward, False),
            ("cluster_spread", cluster_spread, True),
            ("dispersal_distance", dispersal_distance, True),
            ("overgrazing_threshold", overgrazing_threshold, False),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value < 0
                or (positive and value == 0)
            ):
                raise ValueError(
                    f"grass_{name} must be finite and "
                    f"{'positive' if positive else 'nonnegative'}"
                )
        if not 0 <= overgrazing_threshold < 1:
            raise ValueError("grass_overgrazing_threshold must be in [0, 1)")
        if (dispersal or overgrazing) and not clustered:
            raise ValueError(
                "grass_dispersal and grass_overgrazing require grass_clustered"
            )
        if dispersal and random_respawn:
            raise ValueError(
                "grass_dispersal and grass_random_respawn both choose where an "
                "eaten patch regrows; enable only one"
            )
        self.env = raw_env
        self.count = count
        self.consume_radius = consume_radius
        self.food_reward = food_reward
        self.respawn_delay = respawn_delay
        self.clustered = clustered
        self.cluster_count = cluster_count
        self.cluster_spread = cluster_spread
        self.random_respawn = random_respawn
        self.dispersal = dispersal
        self.dispersal_distance = dispersal_distance
        self.overgrazing = overgrazing
        self.overgrazing_threshold = overgrazing_threshold
        self.overgrazing_delay = overgrazing_delay
        self.rng = random.Random()
        self.centres = []
        self.collapsed = []
        self.total_collapses = 0
        self.patches = []
        self.tick = 0
        self.total_consumed = 0

    def reset(self, seed=None):
        self.rng = random.Random(seed)
        if self.clustered:
            self.centres = self.cluster_centres()
            self.patches = []
            for i in range(self.count):
                cluster = i % self.cluster_count
                self.patches.append(GrassPatch(self.spot(cluster), cluster=cluster))
        else:
            self.centres = []
            self.patches = [GrassPatch(self.spot()) for _ in range(self.count)]
        self.collapsed = [False] * len(self.centres)
        self.tick = 0
        self.total_consumed = 0
        self.total_collapses = 0

    def grid(self):
        """Columns and rows of the grid that spreads clusters over the arena
        (4 x 3 for 10 clusters on a square arena)."""
        width, height = self.env.width, self.env.height
        columns = math.ceil(math.sqrt(self.cluster_count * width / height))
        return columns, math.ceil(self.cluster_count / columns)

    def cell_of(self, position):
        columns, rows = self.grid()
        column = min(int(position.x / (self.env.width / columns)), columns - 1)
        row = min(int(position.y / (self.env.height / rows)), rows - 1)
        return row * columns + column

    def centre_in(self, cell):
        """A random point in the middle half of a grid cell."""
        columns, rows = self.grid()
        cell_w, cell_h = self.env.width / columns, self.env.height / rows
        return Vector(
            (cell % columns + self.rng.uniform(0.25, 0.75)) * cell_w,
            (cell // columns + self.rng.uniform(0.25, 0.75)) * cell_h,
        )

    def cluster_centres(self):
        """One centre per cluster, each in its own grid cell, so clusters are
        spread out."""
        columns, rows = self.grid()
        cells = self.rng.sample(range(columns * rows), self.cluster_count)
        return [self.centre_in(cell) for cell in cells]

    def near(self, position, spread):
        """A point scattered around position (wrapped across the edges)."""
        return Vector(
            (position.x + self.rng.gauss(0, spread)) % self.env.width,
            (position.y + self.rng.gauss(0, spread)) % self.env.height,
        )

    def spot(self, cluster=None):
        """A random patch position: uniform over the arena, or scattered
        around a cluster's centre (wrapped across the arena's edges)."""
        width, height = self.env.width, self.env.height
        if cluster is None:
            return Vector(self.rng.uniform(0, width), self.rng.uniform(0, height))
        return self.near(self.centres[cluster], self.cluster_spread)

    def available(self, patch):
        return patch.ready_at <= self.tick

    def advance(self):
        """After movement/captures, award each patch to its closest living prey.

        Ties use agent ID, so each patch pays once per step. Consumption can
        occur across arena edges. A prey may eat multiple patches in a step.
        """
        self.tick += 1
        for patch in self.patches:
            if patch.regrow is not None and self.available(patch):
                self.place(patch)
        eaten = {}
        for patch in self.patches:
            if not self.available(patch):
                continue
            candidates = []
            for prey in self.env.prey:
                if not prey.alive:
                    continue
                distance = self.env.torus.get_distance_in_torus(
                    prey.position, patch.position
                )
                if distance <= self.consume_radius:
                    candidates.append((distance, prey.id()))
            if candidates:
                _, agent = min(candidates)
                eaten[agent] = eaten.get(agent, 0) + 1
                patch.ready_at = self.tick + self.respawn_delay
                if self.random_respawn:
                    # Moves now, but stays hidden until it has regrown.
                    patch.position = self.spot(patch.cluster)
                elif self.dispersal:
                    patch.regrow = "disperse"
                self.total_consumed += 1
        if self.overgrazing:
            self.check_overgrazing()
        return eaten

    def place(self, patch):
        """Position a patch that regrows now (it stays hidden until then)."""
        if patch.regrow == "disperse":
            parents = [
                other
                for other in self.patches
                if other is not patch
                and other.cluster == patch.cluster
                and other.regrow is None
                and self.available(other)
            ]
            if parents:
                parent = self.rng.choice(parents)
                patch.position = self.near(parent.position, self.dispersal_distance)
            else:  # nothing left to seed from: regrow from the seed bank
                patch.position = self.spot(patch.cluster)
        elif patch.regrow == "centre":
            patch.position = self.spot(patch.cluster)
        patch.regrow = None

    def check_overgrazing(self):
        """Collapse clusters grazed down to the threshold; re-arm recovered
        ones."""
        for cluster in range(len(self.centres)):
            members = [p for p in self.patches if p.cluster == cluster]
            living = sum(self.available(p) for p in members)
            limit = self.overgrazing_threshold * len(members)
            if self.collapsed[cluster]:
                if living > limit:
                    self.collapsed[cluster] = False
            elif living <= limit:
                self.collapse(cluster, members)

    def collapse(self, cluster, members):
        """The cluster dies back and regrows elsewhere after a delay."""
        self.collapsed[cluster] = True
        self.total_collapses += 1
        used = {self.cell_of(c) for i, c in enumerate(self.centres) if i != cluster}
        used.add(self.cell_of(self.centres[cluster]))
        columns, rows = self.grid()
        free = [cell for cell in range(columns * rows) if cell not in used]
        if not free:  # every cell taken: any cell but its own
            own = self.cell_of(self.centres[cluster])
            free = [cell for cell in range(columns * rows) if cell != own] or [own]
        self.centres[cluster] = self.centre_in(self.rng.choice(free))
        # The whole cluster regrows together, also patches eaten earlier.
        regrows = self.tick + self.overgrazing_delay
        for patch in members:
            patch.ready_at = regrows
            patch.regrow = "centre"

    def observation(self, prey):
        """[seen, wrapped dx/range, wrapped dy/range, distance/range]."""
        visible = []
        torus = self.env.torus
        for patch in self.patches:
            if not self.available(patch):
                continue
            if not torus.check_if_entity_is_in_view_in_torus(
                prey, patch, self.env.prey_view_distance, self.env.prey_fov
            ):
                continue
            dx = (
                patch.position.x - prey.position.x + torus.width / 2
            ) % torus.width - torus.width / 2
            dy = (
                patch.position.y - prey.position.y + torus.height / 2
            ) % torus.height - torus.height / 2
            visible.append((math.hypot(dx, dy), dx, dy))
        if not visible:
            return np.zeros(4, dtype=np.float32)
        distance, dx, dy = min(visible)
        scale = self.env.prey_view_distance
        return np.asarray(
            [1.0, dx / scale, dy / scale, distance / scale], dtype=np.float32
        )


TITLE_BAR = 40  # Room left for the window's title bar, in pixels.


def ensure_view(raw_env):
    """Create Aquarium's pygame view now, as its render() would, so a layer
    can hook the view's draw methods before the first frame is drawn.

    With raw_env.window_size set (see env_wrapper._patch_scaled_view),
    Aquarium draws at one pixel per arena unit onto an off-screen canvas, and
    show_view scales each finished frame into the left of the window, with a
    population time series to its right. The window is placed at the top of
    the screen, centred horizontally."""
    if raw_env.view is None:
        import pygame
        from marl_aquarium.env.view import View

        pygame.init()
        headless = pygame.display.get_driver() in ("dummy", "offscreen")
        area = None if headless else work_area(pygame)  # before any window
        scaled = getattr(raw_env, "window_size", None) is not None
        arena_px = window_size(raw_env, area)
        panel = arena_px[0] if scaled else 0  # time series as wide as the arena
        total = (arena_px[0] + panel, arena_px[1])
        if area is not None and scaled:
            place_window(area, total)
        view = View(raw_env.width, raw_env.height, raw_env.caption, raw_env.fps)
        view.shark_image = sprite_for_radius(
            pygame, view.shark_image, raw_env.predator_radius
        )
        view.fish_image = sprite_for_radius(
            pygame, view.fish_image, raw_env.prey_radius
        )
        if scaled:
            raw_env.window = pygame.display.set_mode(total)
            raw_env.arena_px = arena_px
            raw_env.population = []  # (physics step, predators, prey)
            view.screen = pygame.Surface((raw_env.width, raw_env.height))
        raw_env.view = view
    return raw_env.view


SPRITE_PER_DIAMETER = 1.25  # Sprite size relative to the body diameter.
MIN_SPRITE = 36  # Arena units; keeps small animals recognizable.


def sprite_for_radius(pygame, image, radius):
    """Aquarium's animal image, cropped to its visible part and scaled so its
    longer side is SPRITE_PER_DIAMETER body diameters (at least MIN_SPRITE
    arena units): close to the body that catches and collides, yet large
    enough to tell a predator from a prey. Display only."""
    image = image.subsurface(image.get_bounding_rect(min_alpha=1))
    target = max(SPRITE_PER_DIAMETER * 2 * radius, MIN_SPRITE)
    scale = target / max(image.get_size())
    size = (max(1, round(image.get_width() * scale)), round(image.get_height() * scale))
    return pygame.transform.smoothscale(image, (size[0], max(1, size[1])))


def work_area(pygame):
    """The screen area not taken by panels or task bars, as (x, y, width,
    height): the window manager's _NET_WORKAREA on X11, else the desktop
    minus a typical task bar."""
    import subprocess

    try:
        out = subprocess.run(
            ["xprop", "-root", "_NET_WORKAREA"],
            check=False,
            capture_output=True,
            text=True,
            timeout=2,
        ).stdout
        x, y, width, height = (int(v) for v in out.split("=")[1].split(",")[:4])
        if width > 0 and height > 0:
            return x, y, width, height
    except (OSError, ValueError, IndexError, subprocess.SubprocessError):
        pass
    desktop = pygame.display.Info()
    if desktop.current_w <= 0 or desktop.current_h <= 0:
        return None
    return 0, 0, desktop.current_w, desktop.current_h - 60


def window_size(raw_env, area):
    """The arena's size on screen for raw_env.window_size: "fit" is the
    largest size that fits the work area below a title bar, with room beside
    it for an equally wide time series; an int is the arena's longer side in
    pixels; None keeps Aquarium's 1:1 window. The arena's aspect ratio is
    kept. "fit" without a screen (headless) also keeps 1:1."""
    setting = getattr(raw_env, "window_size", None)
    arena = (raw_env.width, raw_env.height)
    if setting is None or (setting == "fit" and area is None):
        return arena
    if setting == "fit":
        _, _, width, height = area
        scale = min(width / (2 * arena[0]), (height - TITLE_BAR) / arena[1])
    else:
        scale = setting / max(arena)
    return (round(arena[0] * scale), round(arena[1] * scale))


def place_window(area, size):
    """Ask SDL to open the window at the top of the work area, centred
    horizontally; the window manager keeps the title bar on screen, so the
    drawing area ends up just below it. Must run before the window exists.

    Uses SDL_VIDEO_WINDOW_POS, not pygame._sdl2's Window: a Window object
    registers itself on the SDL window, and every later window event points
    at it, so letting it be freed crashed pygame (segfault in event cleanup).
    An SDL_VIDEO_WINDOW_POS set by the user is left alone."""
    import os

    x, y, width, _ = area
    left = x + max(0, (width - size[0]) // 2)
    os.environ.setdefault("SDL_VIDEO_WINDOW_POS", f"{left},{y}")


def show_view(raw_env):
    """Scale the finished canvas into the left of the window, draw the
    population time series to its right, and display it. Returns the
    window's pixels as Aquarium's get_frame does, (width, height, 3)."""
    import pygame

    window = getattr(raw_env, "window", None)
    if window is None:
        return None
    width, height = raw_env.arena_px
    arena = window.subsurface((0, 0, width, height))
    pygame.transform.smoothscale(raw_env.view.screen, (width, height), arena)
    history = raw_env.population
    if history and raw_env.time_step < history[-1][0]:
        history.clear()  # a new episode
    history.append((raw_env.time_step, len(raw_env.predators), len(raw_env.prey)))
    panel = window.subsurface((width, 0, window.get_width() - width, height))
    draw_population(pygame, panel, history, raw_env.max_time_steps)
    pygame.display.update()
    return pygame.surfarray.array3d(window)


PREDATOR_COLOR = (200, 60, 50)
PREY_COLOR = (150, 100, 40)


def draw_population(pygame, surface, history, max_steps):
    """Predator and prey counts over the episode: physics steps 0 to
    max_steps on the x axis, animals on the y axis."""
    width, height = surface.get_size()
    surface.fill((250, 250, 248))
    if width < 100:
        return
    font = pygame.font.Font(None, max(16, height // 40))
    title = pygame.font.Font(None, max(20, height // 28))
    ink, grid = (60, 60, 60), (225, 225, 220)
    left, right = int(width * 0.11), int(width * 0.04)
    top, bottom = int(height * 0.09), int(height * 0.08)
    plot_w, plot_h = width - left - right, height - top - bottom
    peak = max([1] + [max(pred, prey) for _, pred, prey in history])
    y_max = next(m for m in (10, 20, 25, 40, 50, 75, 100, 150, 200, 500) if m >= peak)
    y_max = max(y_max, peak)

    def to_xy(step, count):
        x = left + plot_w * min(step, max_steps) / max_steps
        return x, top + plot_h * (1 - count / y_max)

    for i in range(6):  # gridlines and axis labels
        count = y_max * i / 5
        _, y = to_xy(0, count)
        pygame.draw.line(surface, grid, (left, y), (left + plot_w, y))
        label = font.render(f"{count:g}", True, ink)
        surface.blit(label, (left - label.get_width() - 6, y - label.get_height() / 2))
        step = max_steps * i / 5
        x, _ = to_xy(step, 0)
        label = font.render(f"{step:g}", True, ink)
        surface.blit(label, (x - label.get_width() / 2, top + plot_h + 6))
    pygame.draw.rect(surface, ink, (left, top, plot_w, plot_h), 1)
    for index, color in (
        (1, PREDATOR_COLOR),
        (2, PREY_COLOR),
    ):  # rows: step, pred, prey
        points = [to_xy(row[0], row[index]) for row in history]
        if len(points) > 1:
            pygame.draw.lines(surface, color, False, points, 3)
    pred, prey = (history[-1][1], history[-1][2]) if history else (0, 0)
    surface.blit(title.render("Population", True, ink), (left, top * 0.3))
    legend_x = left + plot_w
    for text, color in (
        (f"prey {prey}", PREY_COLOR),
        (f"predators {pred}", PREDATOR_COLOR),
    ):
        label = font.render(text, True, color)
        legend_x -= label.get_width()
        surface.blit(label, (legend_x, top * 0.45))
        legend_x -= 24
    caption = font.render("physics step", True, ink)
    surface.blit(
        caption, (left + plot_w / 2 - caption.get_width() / 2, height - bottom * 0.45)
    )


def patch_grass(raw_env, **settings):
    """Attach food to one Aquarium instance before RLlib snapshots its spaces.

    Only prey receive four additional inputs. Predator observations retain
    their original shape, allowing later experiments with a frozen hunter.
    """
    layer = GrassLayer(raw_env, **settings)
    raw_env.grass = layer
    original_reset = raw_env.reset
    original_step = raw_env.step
    original_obs = raw_env.get_obs
    original_space = raw_env.observation_space
    original_render = raw_env.render
    prey_space = Box(
        -1.0, 1.0, shape=(raw_env.number_of_fish_observations + 4,), dtype=np.float32
    )
    raw_env.number_of_fish_observations += 4

    def get_obs():
        obs = original_obs()
        for prey in raw_env.prey:
            agent = prey.id()
            if agent in obs:
                obs[agent] = np.concatenate((obs[agent], layer.observation(prey)))
        return obs

    def reset(seed=None, options=None):
        layer.reset(seed)
        return original_reset(seed=seed, options=options)

    def step(actions):
        obs, rewards, terms, truncs, infos = original_step(actions)
        eaten = layer.advance()
        # Aquarium observes before rewards; refresh after food consumption so
        # consumed patches are absent from this transition's next observation.
        obs.update(get_obs())
        for agent in rewards:
            count = eaten.get(agent, 0)
            if count:
                rewards[agent] += count * layer.food_reward
            infos.setdefault(agent, {}).update(
                grass_eaten=count, grass_eaten_total=layer.total_consumed
            )
        return obs, rewards, terms, truncs, infos

    sprite = None
    hooked_view = None

    def draw_grass():
        nonlocal sprite
        import pygame

        if sprite is None:
            asset = Path(__file__).resolve().parents[1] / "assets/grass-transparent.png"
            image = pygame.image.load(str(asset)).convert_alpha()
            bounds = image.get_bounding_rect(min_alpha=1)
            image = image.subsurface(bounds)
            width = 24
            height = max(1, round(width * image.get_height() / image.get_width()))
            sprite = pygame.transform.smoothscale(image, (width, height))
        for patch in layer.patches:
            if layer.available(patch):
                # Draw wrapped copies where a patch straddles the boundary.
                for dx in (-raw_env.width, 0, raw_env.width):
                    for dy in (-raw_env.height, 0, raw_env.height):
                        rect = sprite.get_rect(
                            center=(
                                round(patch.position.x + dx),
                                round(patch.position.y + dy),
                            )
                        )
                        raw_env.view.screen.blit(sprite, rect)

    def render(mode=None):
        # Aquarium draws background then animals in one call; hook the view's
        # draw_background so grass lands between them, under the animals.
        view = ensure_view(raw_env)
        nonlocal hooked_view
        if view is not hooked_view:

            def draw_background_and_grass():
                # Aquarium's draw_background, without its FPS counter in the
                # top-left corner: clear the frame and keep the frame rate.
                view.screen.blit(view.background, (0, 0))
                view.clock.tick(view.fps)
                draw_grass()

            view.draw_background = draw_background_and_grass
            hooked_view = view
        return original_render(mode)

    raw_env.reset = reset
    raw_env.step = step
    raw_env.get_obs = get_obs
    raw_env.observation_space = lambda agent: (
        prey_space if agent.startswith("prey_") else original_space(agent)
    )
    raw_env.render = render
