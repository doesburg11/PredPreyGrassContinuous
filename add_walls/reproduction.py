"""Energy-threshold reproduction, as in PredPreyGrass's base_environment."""

import math
import random
from typing import Any

import numpy as np
from marl_aquarium.env.predator import Predator
from marl_aquarium.env.prey import Prey
from marl_aquarium.env.vector import Vector

from energy import SPECIES, species_of


class ReproductionLayer:
    """Settings and bookkeeping for births.

    Agent IDs come from a fixed pool per species (predator_0 .. predator_<pool-1>)
    and are never reused within an episode: RLlib maps each agent ID to one
    trajectory per episode. Newborns are numbered by this layer's own
    counters, continuing after the initial animals. Aquarium's class-level
    counters are shared by every environment in the process (and set back
    to 0 by each one's reset), so they can't number births.
    """

    def __init__(
        self,
        energy_layer,
        predator_threshold=12.0,
        prey_threshold=8.0,
        predator_reward=10.0,
        prey_reward=10.0,
        max_predators=50,
        max_prey=100,
        predator_pool=2000,
        prey_pool=2000,
    ):
        settings = {
            "predator_threshold": (predator_threshold, True),
            "prey_threshold": (prey_threshold, True),
            "predator_reward": (predator_reward, False),
            "prey_reward": (prey_reward, False),
        }
        for name, (value, positive) in settings.items():
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value < 0
                or (positive and value == 0)
            ):
                raise ValueError(
                    f"reproduction_{name} must be finite and "
                    f"{'positive' if positive else 'nonnegative'}"
                )
        for name, value in (
            ("max_predators", max_predators),
            ("max_prey", max_prey),
            ("predator_pool", predator_pool),
            ("prey_pool", prey_pool),
        ):
            if type(value) is not int or value < 1:
                raise ValueError(f"reproduction_{name} must be a positive integer")
        self.threshold = {"predator": predator_threshold, "prey": prey_threshold}
        for species in SPECIES:
            initial = energy_layer.initial[species]
            cap = energy_layer.max[species]
            # A parent pays the offspring's initial energy; at or below it a
            # parent could breed while left with nothing.
            if not initial < self.threshold[species] <= cap:
                raise ValueError(
                    f"reproduction_{species}_threshold must exceed "
                    f"energy_{species}_initial and not exceed energy_{species}_max"
                )
        self.reward = {"predator": predator_reward, "prey": prey_reward}
        self.max_count = {"predator": max_predators, "prey": max_prey}
        self.pool = {"predator": predator_pool, "prey": prey_pool}
        self.rng = random.Random()
        self.births = {"predator": 0, "prey": 0}
        self.next_id = {"predator": 0, "prey": 0}

    def agent_ids(self):
        return [
            f"{species}_{i}" for species in SPECIES for i in range(self.pool[species])
        ]

    def reset(self, initial_counts, seed=None):
        # Own RNG for spawn offsets, so births don't shift Aquarium's stream.
        self.rng = random.Random(None if seed is None else seed + 1)
        self.births = {"predator": 0, "prey": 0}
        self.next_id = dict(initial_counts)


def patch_reproduction(raw_env, **settings):
    """Let animals reproduce once their energy reaches a threshold.

    Runs after the energy layer each step (so after movement, catches,
    feeding, metabolic costs and starvation). Each living animal at or above its
    species' threshold has one offspring, while its species is below
    max_<species> and the ID pool has IDs left: the offspring gets the
    species' initial energy, the parent loses that amount and earns the
    species' reproduction reward. Offspring appear two body radii from the
    parent, in a random direction, and act from the next step. No births
    happen on a step that ends the episode.
    """
    if not hasattr(raw_env, "energy"):
        raise ValueError("Reproduction requires energy settings")
    layer = ReproductionLayer(raw_env.energy, **settings)
    raw_env.reproduction = layer
    energy = raw_env.energy
    for species, count in (
        ("predator", raw_env.predator_count),
        ("prey", raw_env.prey_count),
    ):
        if count > layer.max_count[species] or count > layer.pool[species]:
            raise ValueError(
                f"{species}_count must not exceed reproduction_max_"
                f"{'predators' if species == 'predator' else 'prey'} or the pool"
            )
    original_reset = raw_env.reset
    original_step = raw_env.step

    def reset(seed=None, options=None):
        layer.reset(
            {"predator": raw_env.predator_count, "prey": raw_env.prey_count}, seed
        )
        return original_reset(seed=seed, options=options)

    def spawn(parent):
        species = species_of(parent)
        distance = 2 * parent.radius
        angle = layer.rng.uniform(0, 2 * math.pi)
        position = Vector(
            (parent.position.x + distance * math.cos(angle)) % raw_env.width,
            (parent.position.y + distance * math.sin(angle)) % raw_env.height,
        )
        # Same initial velocity/acceleration as Aquarium's create_random_*.
        child: Any  # carries the energy attribute added by the energy layer
        if species == "predator":
            child = Predator(
                position,
                Vector(0, -1),
                Vector(0, 0),
                raw_env.predator_radius,
                raw_env.predator_view_distance,
                raw_env.predator_max_velocity,
                raw_env.predator_max_acceleration,
            )
            raw_env.predators.append(child)
        else:
            # Prey() draws an (unused) replication_age from the global random
            # module; restore its state so births leave Aquarium's stream alone.
            state = random.getstate()
            child = Prey(
                position,
                Vector(-1, 0),
                Vector(0, 0),
                raw_env.prey_radius,
                raw_env.prey_view_distance,
                raw_env.prey_max_velocity,
                raw_env.prey_max_acceleration,
            )
            random.setstate(state)
            raw_env.prey.append(child)
            raw_env.current_prey_count += 1
        child.identifier = layer.next_id[species]
        layer.next_id[species] += 1
        walls = getattr(raw_env, "walls", None)
        if walls is not None:  # born next to a wall: move it out
            walls.push_out(child)
        raw_env.all_entities.append(child)
        raw_env.agents.append(child.id())
        child.energy = energy.initial[species]
        parent.energy -= energy.initial[species]
        return child

    def step(actions):
        obs, rewards, terms, truncs, infos = original_step(actions)
        if not raw_env.agents:
            return obs, rewards, terms, truncs, infos  # episode ended this step
        population = {"predator": len(raw_env.predators), "prey": len(raw_env.prey)}
        born = []
        for parent in raw_env.predators + raw_env.prey:
            species = species_of(parent)
            if (
                parent.energy < layer.threshold[species]
                or parent.id() not in raw_env.agents
                or population[species] >= layer.max_count[species]
                or layer.next_id[species] >= layer.pool[species]
            ):
                continue
            child = spawn(parent)
            born.append((child, parent.id()))
            population[species] += 1
            layer.births[species] += 1
            parent_id = parent.id()
            rewards[parent_id] = rewards.get(parent_id, 0.0) + layer.reward[species]
            infos.setdefault(parent_id, {})["births"] = 1
            infos[parent_id]["energy"] = parent.energy
        if born:
            obs.update(raw_env.get_obs())
            for child, parent_id in born:
                agent = child.id()
                if agent not in obs:  # observations skipped mid-decision
                    shape = raw_env.observation_space(agent).shape
                    obs[agent] = np.zeros(shape, dtype=np.float32)
                rewards[agent] = 0.0
                terms[agent] = False
                truncs[agent] = False
                infos[agent] = {
                    "born": True,
                    "parent": parent_id,
                    "energy": child.energy,
                }
        return obs, rewards, terms, truncs, infos

    raw_env.reset = reset
    raw_env.step = step
