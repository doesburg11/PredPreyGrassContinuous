"""Stationary, renewable food patches for the minimal grass experiment."""

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


class GrassLayer:
    """Use a separate RNG and physics-step clock; preserve each patch location."""

    def __init__(
        self,
        raw_env,
        count=24,
        consume_radius=12.0,
        food_reward=10.0,
        respawn_delay=100,
    ):
        for name, value, minimum in (
            ("count", count, 0),
            ("respawn_delay", respawn_delay, 1),
        ):
            if type(value) is not int or value < minimum:
                raise ValueError(f"grass_{name} must be an integer >= {minimum}")
        for name, value, positive in (
            ("consume_radius", consume_radius, True),
            ("food_reward", food_reward, False),
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
        self.env = raw_env
        self.count = count
        self.consume_radius = consume_radius
        self.food_reward = food_reward
        self.respawn_delay = respawn_delay
        self.patches = []
        self.tick = 0
        self.total_consumed = 0

    def reset(self, seed=None):
        rng = random.Random(seed)
        self.patches = [
            GrassPatch(
                Vector(rng.uniform(0, self.env.width), rng.uniform(0, self.env.height))
            )
            for _ in range(self.count)
        ]
        self.tick = 0
        self.total_consumed = 0

    def available(self, patch):
        return patch.ready_at <= self.tick

    def advance(self):
        """After movement/captures, award each patch to its closest living prey.

        Ties use agent ID, so each patch pays once per step. Consumption can
        occur across arena edges. A prey may eat multiple patches in a step.
        """
        self.tick += 1
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
                self.total_consumed += 1
        return eaten

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
        if raw_env.view is None:
            from marl_aquarium.env.view import View

            raw_env.view = View(
                raw_env.width, raw_env.height, raw_env.caption, raw_env.fps
            )
        view = raw_env.view
        nonlocal hooked_view
        if view is not hooked_view:
            draw_background = view.draw_background

            def draw_background_and_grass():
                draw_background()
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
