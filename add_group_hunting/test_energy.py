"""Metabolic costs, food, catches, starvation, observations, and rendering."""

import math

import numpy as np
import pytest
from marl_aquarium.env.vector import Vector

from energy import EnergyLayer
from env_wrapper import make_env
from grass import GrassPatch

BASE = {
    "predator_count": 1,
    "prey_count": 2,
    "obs_mode": "egocentric",
    "keep_prey_count_constant": False,
    "grass_count": 0,
    "energy_predator_initial": 100.0,
    "energy_prey_initial": 50.0,
    "energy_predator_max": 200.0,
    "energy_prey_max": 100.0,
    "energy_predator_resting_metabolic_cost": 1.0,
    "energy_prey_resting_metabolic_cost": 0.5,
    "energy_grass_gain": 20.0,
    "energy_catch_efficiency_predator": 0.5,
}


@pytest.fixture
def env():
    env = make_env(dict(BASE))
    yield env
    env.close()


def setup(env, seed=0):
    """Reset, then spread the animals out and stop them, so nothing is
    caught or eaten unless a test arranges it."""
    obs, _ = env.reset(seed=seed)
    raw = env.par_env.aec_env.unwrapped
    for entity, x in zip(raw.predators + raw.prey, (100, 400, 700)):
        entity.position = Vector(x, 400)
        entity.velocity = Vector(0, 0)
    return obs, raw


def idle(env, obs):
    # Action 0 moves "up" (towards -y); animals sit 300 apart along x.
    return {agent: 0 for agent in obs}


@pytest.mark.parametrize(
    "settings",
    [
        {"prey_initial": 0},
        {"predator_resting_metabolic_cost": -1},
        {"grass_gain": float("nan")},
        {"catch_efficiency_predator": 1.5},
        {"prey_initial": 150, "prey_max": 100},
        {"predator_max": True},
        {"observe_others": 1},
    ],
)
def test_invalid_energy_settings(settings):
    with pytest.raises(ValueError):
        EnergyLayer(**settings)


def test_energy_requires_permanent_prey_death_and_grass():
    with pytest.raises(ValueError, match="keep_prey_count_constant"):
        make_env({**BASE, "keep_prey_count_constant": True})
    config = {k: v for k, v in BASE.items() if not k.startswith("grass_")}
    with pytest.raises(ValueError, match="grass"):
        make_env(config)


def test_energy_disables_aquarium_age_starvation(env):
    assert env.par_env.aec_env.unwrapped.predator_max_age == 10**9


def test_observations_end_with_own_energy_fraction(env):
    obs, _ = setup(env)
    assert env.observation_space["predator_0"].shape == (29,)
    assert env.observation_space["prey_0"].shape == (33,)
    assert obs["predator_0"][-1] == pytest.approx(100 / 200)
    assert obs["prey_0"][-1] == pytest.approx(50 / 100)
    obs, _, _, _, infos = env.step(idle(env, obs))
    assert obs["predator_0"][-1] == pytest.approx(99 / 200)
    assert obs["prey_1"][-1] == pytest.approx(49.5 / 100)
    assert infos["prey_1"]["energy"] == pytest.approx(49.5)
    for agent, value in obs.items():
        assert env.observation_space[agent].contains(value)


def test_observe_others_adds_each_seen_animals_energy():
    env = make_env(
        {**BASE, "energy_observe_others": True, "predator_fov": 360, "prey_fov": 360}
    )
    try:
        _, raw = setup(env)
        predator = raw.predators[0]
        near, far = raw.prey
        predator.position = Vector(400, 400)
        near.position = Vector(460, 400)
        far.position = Vector(400, 470)
        predator.energy, near.energy, far.energy = 150.0, 30.0, 80.0
        obs = raw.get_obs()
        # One more input per animal slot (1 predator + 3 prey slots).
        assert env.observation_space["predator_0"].shape == (29 + 4,)
        assert env.observation_space["prey_0"].shape == (33 + 4,)
        o = obs["predator_0"]
        # Own 4, then a 7-value predator slot (empty: no other predator), then
        # prey slots nearest first, each ending in that prey's energy fraction.
        assert not o[4:11].any()
        assert o[11] == 1 and o[17] == pytest.approx(30 / 100)
        assert o[18] == 1 and o[24] == pytest.approx(80 / 100)
        assert not o[25:32].any()  # third prey slot empty
        assert o[-1] == pytest.approx(150 / 200)  # own energy, as before
        p = obs[near.id()]
        assert p[4] == 1 and p[10] == pytest.approx(150 / 200)  # the predator
        assert p[11] == 1 and p[17] == pytest.approx(80 / 100)  # the other prey
        for agent, value in obs.items():
            assert env.observation_space[agent].contains(value)
    finally:
        env.close()


@pytest.mark.parametrize("action_repeat", [1, 3])
def test_observed_energies_are_current_after_a_step(action_repeat):
    env = make_env(
        {
            **BASE,
            "energy_observe_others": True,
            "predator_fov": 360,
            "prey_fov": 360,
            "action_repeat": action_repeat,
        }
    )
    try:
        _, raw = setup(env)
        predator = raw.predators[0]
        near, far = raw.prey
        predator.position = Vector(400, 400)
        near.position = Vector(460, 400)
        far.position = Vector(400, 470)
        obs, _, _, _, _ = env.step({agent: 0 for agent in env.agents})
        o = obs["predator_0"]
        # Prey slots are nearest first; each shows that prey's current energy,
        # after this decision's resting costs.
        slots = sorted(
            raw.prey,
            key=lambda prey: math.hypot(
                prey.position.x - predator.position.x,
                prey.position.y - predator.position.y,
            ),
        )
        for index, prey in enumerate(slots):
            assert o[11 + 7 * index] == 1
            assert o[17 + 7 * index] == pytest.approx(prey.energy / 100, abs=1e-6)
        assert prey.energy < 50.0
        for agent, value in obs.items():
            assert env.observation_space[agent].contains(value)
    finally:
        env.close()


def test_observe_others_needs_egocentric_observations():
    with pytest.raises(ValueError, match="egocentric"):
        make_env({**BASE, "obs_mode": "aquarium", "energy_observe_others": True})


def test_grass_feeds_prey_up_to_max(env):
    obs, raw = setup(env)
    prey = raw.prey[0]
    raw.grass.patches = [GrassPatch(prey.position.copy())]
    env.step(idle(env, obs))
    assert prey.energy == pytest.approx(50 + 20 - 0.5)
    prey.energy = 95.0
    raw.grass.patches = [GrassPatch(prey.position.copy())]
    env.step(idle(env, obs))
    assert prey.energy == pytest.approx(100 - 0.5)


def test_catch_feeds_the_catching_predator(env):
    obs, raw = setup(env)
    predator, prey = raw.predators[0], raw.prey[0]
    prey.energy = 40.0
    prey.position = predator.position.copy()
    _, _, terms, _, infos = env.step(idle(env, obs))
    assert terms[prey.id()] and not infos[prey.id()].get("starved")
    assert predator.energy == pytest.approx(100 + 0.5 * 40 - 1)


def test_simultaneous_catches_feed_their_own_catchers():
    env = make_env({**BASE, "predator_count": 2, "prey_count": 3})
    try:
        obs, _ = env.reset(seed=0)
        raw = env.par_env.aec_env.unwrapped
        first, second = raw.predators
        near_second, near_first, bystander = raw.prey
        for entity, x in zip((first, second, bystander), (100, 400, 700)):
            entity.position = Vector(x, 400)
        for entity in raw.predators + raw.prey:
            entity.velocity = Vector(0, 0)
        # Different energies so a swapped or misattributed catch shows.
        near_second.energy, near_first.energy = 40.0, 10.0
        near_second.position = second.position.copy()
        near_first.position = first.position.copy()
        _, _, terms, _, _ = env.step(idle(env, obs))
        assert terms[near_second.id()] and terms[near_first.id()]
        assert first.energy == pytest.approx(100 + 0.5 * 10 - 1)
        assert second.energy == pytest.approx(100 + 0.5 * 40 - 1)
        assert bystander.energy == pytest.approx(50 - 0.5)
    finally:
        env.close()


def test_prey_starvation_terminates_only_that_prey(env):
    obs, raw = setup(env)
    hungry, other = raw.prey
    hungry.energy = 0.5
    obs, _, terms, truncs, infos = env.step(idle(env, obs))
    assert terms[hungry.id()] and not truncs[hungry.id()]
    assert infos[hungry.id()]["starved"]
    assert not hungry.alive and hungry not in raw.prey
    assert hungry not in raw.all_entities
    np.testing.assert_array_equal(obs[hungry.id()], np.zeros(33))
    assert not terms["__all__"] and not truncs["__all__"]
    assert other.id() in raw.agents and hungry.id() not in raw.agents
    # Later steps go on without the starved prey.
    obs, _, terms, _, _ = env.step({agent: 0 for agent in obs if agent != hungry.id()})
    assert hungry.id() not in obs and not terms["__all__"]


def test_last_prey_starving_terminates_everyone(env):
    obs, raw = setup(env)
    for prey in raw.prey:
        prey.energy = 0.1
    _, _, terms, truncs, _ = env.step(idle(env, obs))
    assert terms["__all__"] and not truncs["__all__"]
    assert raw.agents == []


def test_last_predator_starving_truncates_surviving_prey(env):
    obs, raw = setup(env)
    raw.predators[0].energy = 0.5
    _, _, terms, truncs, infos = env.step(idle(env, obs))
    assert terms["predator_0"] and infos["predator_0"]["starved"]
    for prey in ("prey_0", "prey_1"):
        assert truncs[prey] and not terms[prey]
    assert truncs["__all__"] and not terms["__all__"]
    assert raw.agents == []


def test_starvation_survives_action_repeat():
    env = make_env({**BASE, "action_repeat": 3})
    try:
        obs, raw = setup(env)
        hungry = raw.prey[0]
        hungry.energy = 1.2  # starves on the third sub-step
        _, _, terms, _, infos = env.step(idle(env, obs))
        assert terms[hungry.id()] and infos[hungry.id()]["starved"]
        assert raw.prey[0].energy == pytest.approx(50 - 3 * 0.5)
    finally:
        env.close()


def test_render_draws_energy_bar_over_animals(monkeypatch):
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    env = make_env({**BASE, "render_mode": "rgb_array"})
    try:
        _, raw = setup(env)
        prey = raw.prey[0]
        x, y = round(prey.position.x), round(prey.position.y)
        env.par_env.render()  # creates the view
        bar_y = round(y - raw.view.fish_image.get_height() / 2 - 4)
        prey.energy = 100.0
        full = env.par_env.render()[x - 15 : x + 15, bar_y]
        prey.energy = 1.0
        low = env.par_env.render()[x - 15 : x + 15, bar_y]
        green = np.array([40, 170, 60])
        assert (full == green).all(axis=-1).sum() >= 28
        assert (low == green).all(axis=-1).sum() == 0
    finally:
        env.close()
