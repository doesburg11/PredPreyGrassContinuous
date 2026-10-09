"""Metabolic energy for the energy experiment: costs, food, and starvation."""

import math

import numpy as np
from gymnasium.spaces import Box

from grass import ensure_view

SPECIES = ("predator", "prey")


def species_of(entity_or_agent) -> str:
    agent = (
        entity_or_agent if isinstance(entity_or_agent, str) else entity_or_agent.id()
    )
    return "predator" if agent.startswith("predator_") else "prey"


class EnergyLayer:
    """Per-species energy settings and bookkeeping.

    Energy lives on each Aquarium entity as `entity.energy`. Entities that
    have not been assigned energy yet (Aquarium's reset observes before this
    layer can set it) read as their species' initial energy.
    """

    def __init__(
        self,
        predator_initial=5.0,
        prey_initial=3.0,
        predator_max=12.0,
        prey_max=8.0,
        predator_resting_metabolic_cost=0.15 / 8,
        prey_resting_metabolic_cost=0.05 / 8,
        predator_speed_cost=0.0,
        prey_speed_cost=0.0,
        predator_acceleration_cost=0.0,
        prey_acceleration_cost=0.0,
        grass_gain=2.0,
        catch_efficiency_predator=1.0,
        observe_others=False,
    ):
        settings = {
            "predator_initial": (predator_initial, True),
            "prey_initial": (prey_initial, True),
            "predator_max": (predator_max, True),
            "prey_max": (prey_max, True),
            "predator_resting_metabolic_cost": (predator_resting_metabolic_cost, False),
            "prey_resting_metabolic_cost": (prey_resting_metabolic_cost, False),
            "predator_speed_cost": (predator_speed_cost, False),
            "prey_speed_cost": (prey_speed_cost, False),
            "predator_acceleration_cost": (predator_acceleration_cost, False),
            "prey_acceleration_cost": (prey_acceleration_cost, False),
            "grass_gain": (grass_gain, False),
            "catch_efficiency_predator": (catch_efficiency_predator, False),
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
                    f"energy_{name} must be finite and "
                    f"{'positive' if positive else 'nonnegative'}"
                )
        for species in SPECIES:
            if settings[f"{species}_initial"][0] > settings[f"{species}_max"][0]:
                raise ValueError(
                    f"energy_{species}_initial must not exceed energy_{species}_max"
                )
        if type(observe_others) is not bool:
            raise ValueError("energy_observe_others must be True or False")
        if catch_efficiency_predator > 1:
            raise ValueError("energy_catch_efficiency_predator must be in [0, 1]")
        self.initial = {"predator": predator_initial, "prey": prey_initial}
        self.max = {"predator": predator_max, "prey": prey_max}
        self.resting_metabolic_cost = {
            "predator": predator_resting_metabolic_cost,
            "prey": prey_resting_metabolic_cost,
        }
        self.speed_cost = {"predator": predator_speed_cost, "prey": prey_speed_cost}
        self.acceleration_cost = {
            "predator": predator_acceleration_cost,
            "prey": prey_acceleration_cost,
        }
        self.grass_gain = grass_gain
        self.catch_efficiency_predator = catch_efficiency_predator
        self.observe_others = observe_others

    def energy(self, entity):
        return getattr(entity, "energy", self.initial[species_of(entity)])

    def gain(self, entity, amount):
        species = species_of(entity)
        entity.energy = min(self.max[species], self.energy(entity) + amount)

    def fraction(self, entity):
        return self.energy(entity) / self.max[species_of(entity)]


def patch_energy(raw_env, **settings):
    """Attach energy to one Aquarium instance before RLlib snapshots its spaces.

    Each step, after Aquarium's movement and captures and after grass is
    eaten: catching predators gain catch_efficiency_predator times the prey's
    energy, prey gain grass_gain per patch eaten, every living animal loses
    its species' resting metabolic cost plus any movement costs, and animals
    at or below zero energy starve: they are removed like a caught prey and
    terminated. The episode ends once no predator or no prey is left. Every
    agent observes one more input, its own energy as a fraction of its
    species' maximum. With observe_others, every animal slot of the
    egocentric observation also shows that animal's energy fraction (one
    more input per slot; empty slots show 0).

    Requires keep_prey_count_constant=False: a respawned prey would need a
    separate rule for its energy, and starvation needs permanent death.
    """
    if raw_env.keep_prey_count_constant:
        raise ValueError("Energy requires keep_prey_count_constant=False")
    layer = EnergyLayer(**settings)
    raw_env.energy = layer
    original_reset = raw_env.reset
    original_step = raw_env.step
    original_obs = raw_env.get_obs
    original_space = raw_env.observation_space
    original_update_prey = raw_env.update_prey
    original_render = raw_env.render
    extra = 1  # own energy
    if layer.observe_others:
        if not getattr(raw_env, "_egocentric_obs_patched", False):
            raise ValueError("energy_observe_others needs obs_mode='egocentric'")
        extra += raw_env.predator_observe_count + raw_env.prey_observe_count
        raw_env._ego_slot_energy = lambda animal: float(
            np.clip(layer.fraction(animal), 0.0, 1.0)
        )
    spaces = {}
    for species, agent in (("predator", "predator_0"), ("prey", "prey_0")):
        size = original_space(agent).shape[0] + extra
        spaces[species] = Box(-1.0, 1.0, shape=(size,), dtype=np.float32)
    raw_env.number_of_predator_observations += extra
    raw_env.number_of_fish_observations += extra
    catches = []  # (catching predator, prey energy) this step

    def get_obs():
        obs = original_obs()
        for entity in raw_env.predators + raw_env.prey:
            agent = entity.id()
            if agent in obs:
                value = np.float32(np.clip(layer.fraction(entity), 0.0, 1.0))
                obs[agent] = np.append(np.asarray(obs[agent], np.float32), value)
        return obs

    def reset(seed=None, options=None):
        catches.clear()
        obs, infos = original_reset(seed=seed, options=options)
        for entity in raw_env.predators + raw_env.prey:
            entity.energy = layer.initial[species_of(entity)]
        return obs, infos

    def update_prey(prey, predators, desired_velocity):
        # Aquarium's own collision test, on the state it is about to use.
        catcher = raw_env.torus.get_colliding_animal(prey, predators)
        result = original_update_prey(prey, predators, desired_velocity)
        if not prey.alive and catcher is not None:
            catches.append((catcher, layer.energy(prey)))
        return result

    def starve(entity, obs, terms, truncs, infos):
        agent = entity.id()
        entity.alive = False
        if entity in raw_env.prey:
            raw_env.prey.remove(entity)
            raw_env.current_prey_count -= 1
            raw_env.dead_animals[agent] = raw_env.time_step
        else:
            raw_env.predators.remove(entity)
        raw_env.all_entities.remove(entity)
        if agent in raw_env.agents:
            raw_env.agents.remove(agent)
        size = spaces[species_of(entity)].shape[0]
        obs[agent] = np.zeros(size, dtype=np.float32)
        terms[agent] = True
        truncs[agent] = False
        infos.setdefault(agent, {})["starved"] = True

    def step(actions):
        previous_velocity = {
            entity.id(): entity.velocity.copy()
            for entity in raw_env.predators + raw_env.prey
        }
        obs, rewards, terms, truncs, infos = original_step(actions)
        for predator, prey_energy in catches:
            if predator.alive:
                layer.gain(predator, layer.catch_efficiency_predator * prey_energy)
        catches.clear()
        for prey in raw_env.prey:
            eaten = infos.get(prey.id(), {}).get("grass_eaten", 0)
            if eaten:
                layer.gain(prey, eaten * layer.grass_gain)
        living = raw_env.predators + raw_env.prey
        for entity in living:
            species = species_of(entity)
            delta = entity.velocity.copy()
            delta.sub(previous_velocity[entity.id()])
            cost = (
                layer.resting_metabolic_cost[species]
                + layer.speed_cost[species] * entity.velocity.mag()
                + layer.acceleration_cost[species] * delta.mag() ** 2
            )
            entity.energy = layer.energy(entity) - cost
        starved = [entity for entity in living if entity.energy <= 0]
        for entity in starved:
            starve(entity, obs, terms, truncs, infos)
        if starved and (not raw_env.prey or not raw_env.predators):
            # Aquarium checks for an empty species before this layer runs.
            for agent in raw_env.agents:
                truncs[agent] = True
            raw_env.agents.clear()
        # Observations were taken before this step's energy changes.
        obs.update(get_obs())
        for entity in raw_env.predators + raw_env.prey:
            infos.setdefault(entity.id(), {})["energy"] = entity.energy
        return obs, rewards, terms, truncs, infos

    def draw_energy_bar(position, animal):
        import pygame

        view = raw_env.view
        image = (
            view.shark_image if species_of(animal) == "predator" else view.fish_image
        )
        width, height = 30, 4
        left = round(position.x - width / 2)
        top = round(position.y - image.get_height() / 2 - height - 2)
        fill = round(width * min(1.0, max(0.0, layer.fraction(animal))))
        pygame.draw.rect(view.screen, (60, 60, 60), (left, top, width, height))
        pygame.draw.rect(view.screen, (40, 170, 60), (left, top, fill, height))

    hooked_view = None

    def render(mode=None):
        nonlocal hooked_view
        # Hook the view so each animal (and its wrapped copies) gets a bar
        # drawn right after its sprite.
        view = ensure_view(raw_env)
        if view is not hooked_view:
            draw_animal = view.draw_animal

            def draw_animal_and_energy(position, animal):
                draw_animal(position, animal)
                if animal.alive:
                    draw_energy_bar(position, animal)

            view.draw_animal = draw_animal_and_energy
            hooked_view = view
        return original_render(mode)

    raw_env.reset = reset
    raw_env.step = step
    raw_env.get_obs = get_obs
    raw_env.update_prey = update_prey
    raw_env.observation_space = lambda agent: spaces[species_of(agent)]
    raw_env.render = render
