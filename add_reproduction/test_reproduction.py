"""Births, ID pool, caps, rewards, and RLlib-facing bookkeeping."""

import numpy as np
import pytest
from gymnasium.spaces import Discrete
from marl_aquarium.env.vector import Vector

from env_wrapper import make_env

BASE = {
    "predator_count": 2,
    "prey_count": 2,
    "max_time_steps": 1000,
    "obs_mode": "egocentric",
    "keep_prey_count_constant": False,
    "prey_reward": 0,
    "prey_punishment": 0,
    "predator_reward": 0,
    "grass_count": 0,
    "grass_food_reward": 0.0,
    "energy_predator_initial": 5.0,
    "energy_prey_initial": 3.0,
    "energy_predator_max": 24.0,
    "energy_prey_max": 16.0,
    "energy_predator_decay": 1.0,
    "energy_prey_decay": 0.5,
    "reproduction_predator_threshold": 12.0,
    "reproduction_prey_threshold": 8.0,
    "reproduction_predator_reward": 10.0,
    "reproduction_prey_reward": 7.0,
    "reproduction_max_predators": 4,
    "reproduction_max_prey": 4,
    "reproduction_predator_pool": 10,
    "reproduction_prey_pool": 10,
}


def make(**overrides):
    return make_env({**BASE, **overrides})


def setup(env, seed=0):
    """Reset, then spread the animals along a line and stop them, so nothing
    is caught unless a test arranges it."""
    obs, _ = env.reset(seed=seed)
    raw = env.par_env.aec_env.unwrapped
    for i, entity in enumerate(raw.predators + raw.prey):
        entity.position = Vector(80 + 160 * i, 400)
        entity.velocity = Vector(0, 0)
    return obs, raw


def act(obs):
    return {agent: 0 for agent in obs}


@pytest.mark.parametrize(
    "overrides",
    [
        {"reproduction_predator_threshold": 5.0},  # not above initial energy
        {"reproduction_prey_threshold": 20.0},  # above the energy cap
        {"reproduction_prey_reward": -1.0},
        {"reproduction_max_prey": 0},
        {"reproduction_predator_pool": 1},  # smaller than predator_count
        {"reproduction_max_predators": 1.5},
    ],
)
def test_invalid_reproduction_settings(overrides):
    with pytest.raises(ValueError):
        make(**overrides)


def test_reproduction_requires_energy():
    config = {k: v for k, v in BASE.items() if not k.startswith("energy_")}
    with pytest.raises(ValueError, match="energy"):
        make_env(config)


def test_whole_id_pool_is_declared_with_species_spaces():
    env = make(obs_stack=2)
    try:
        assert len(env.possible_agents) == 20
        assert env.get_observation_space("predator_9").shape == (2 * 29,)
        assert env.get_observation_space("prey_9").shape == (2 * 33,)
        assert env.get_action_space("prey_9") == Discrete(16)
        obs, _ = env.reset(seed=0)
        assert set(obs) == {"predator_0", "predator_1", "prey_0", "prey_1"}
    finally:
        env.close()


def test_birth_creates_offspring_and_charges_and_rewards_parent():
    env = make()
    try:
        obs, raw = setup(env)
        parent = raw.predators[1]
        parent.energy = 13.0  # 12 after this step's decay of 1
        obs, rewards, terms, truncs, infos = env.step(act(obs))
        child = raw.predators[-1]
        assert child.id() == "predator_2"  # next unused ID
        assert child.energy == pytest.approx(5.0)
        assert parent.energy == pytest.approx(12.0 - 5.0)
        assert rewards["predator_1"] == pytest.approx(10.0)
        assert rewards["predator_2"] == 0.0
        assert rewards["predator_0"] == 0.0
        assert infos["predator_1"]["births"] == 1
        assert infos["predator_2"] == {
            "born": True,
            "parent": "predator_1",
            "energy": 5.0,
        }
        assert not terms["predator_2"] and not truncs["predator_2"]
        assert "predator_2" in raw.agents and child in raw.all_entities
        assert obs["predator_2"].shape == (29,)
        assert obs["predator_2"][-1] == pytest.approx(5.0 / 24.0)
        distance = raw.torus.get_distance_in_torus(parent.position, child.position)
        assert distance == pytest.approx(2 * parent.radius, abs=1e-6)
        # The newborn acts from the next step on.
        before = child.position.copy()
        obs, _, _, _, _ = env.step(act(obs))
        assert child.position.x != before.x or child.position.y != before.y
        assert "predator_2" in obs
    finally:
        env.close()


def test_prey_birth_uses_prey_settings():
    env = make()
    try:
        obs, raw = setup(env)
        raw.prey[0].energy = 9.0  # 8.5 after decay
        _, rewards, _, _, infos = env.step(act(obs))
        assert infos["prey_2"]["parent"] == "prey_0"
        assert rewards["prey_0"] == pytest.approx(7.0)
        assert raw.prey[0].energy == pytest.approx(8.5 - 3.0)
        assert raw.prey[-1].energy == pytest.approx(3.0)
        assert raw.current_prey_count == 3
    finally:
        env.close()


def test_no_birth_below_threshold_and_one_birth_per_parent_per_step():
    env = make(energy_predator_max=24.0)
    try:
        obs, raw = setup(env)
        raw.predators[0].energy = 12.9  # 11.9 after decay: no birth
        raw.predators[1].energy = 23.0  # 22 after decay: one birth only
        obs, _, _, _, _ = env.step(act(obs))
        assert len(raw.predators) == 3
        assert raw.predators[1].energy == pytest.approx(17.0)
        # Still above the threshold, so it breeds again next step.
        env.step(act(obs))
        assert len(raw.predators) == 4
    finally:
        env.close()


def test_population_cap_blocks_births():
    env = make(reproduction_max_prey=2)
    try:
        obs, raw = setup(env)
        raw.prey[0].energy = 9.0
        _, rewards, _, _, infos = env.step(act(obs))
        assert len(raw.prey) == 2 and "births" not in infos["prey_0"]
        assert rewards["prey_0"] == 0.0
        assert raw.prey[0].energy == pytest.approx(8.5)  # nothing paid
    finally:
        env.close()


def test_ids_are_never_reused_and_births_stop_when_the_pool_runs_out():
    env = make(reproduction_prey_pool=3, reproduction_max_prey=4)
    try:
        obs, raw = setup(env)
        doomed = raw.prey[1]
        doomed.energy = 0.1  # starves this step
        raw.prey[0].energy = 9.0
        obs, _, terms, _, infos = env.step(act(obs))
        assert terms["prey_1"] and infos["prey_1"]["starved"]
        assert [p.id() for p in raw.prey] == ["prey_0", "prey_2"]  # prey_1 not reused
        raw.prey[0].energy = 9.0
        _, _, _, _, infos = env.step(act(obs))
        assert len(raw.prey) == 2 and "births" not in infos["prey_0"]
    finally:
        env.close()


def test_no_births_on_the_final_step():
    env = make(max_time_steps=1, reproduction_max_predators=10)
    try:
        obs, raw = setup(env)
        raw.predators[0].energy = 20.0
        while True:
            obs, _, terms, truncs, _ = env.step(act(obs))
            if terms["__all__"] or truncs["__all__"]:
                break
            raw.predators[0].energy = 20.0  # breeds on every non-final step
        assert truncs["__all__"]
        assert all(truncs[agent] for agent in truncs if agent != "__all__")
        assert raw.agents == []
    finally:
        env.close()


def test_reset_restores_initial_population_and_ids():
    env = make()
    try:
        obs, raw = setup(env)
        raw.predators[0].energy = 20.0
        env.step(act(obs))
        assert len(raw.predators) == 3
        obs, infos = env.reset(seed=1)
        assert (
            set(obs)
            == set(infos)
            == {
                "predator_0",
                "predator_1",
                "prey_0",
                "prey_1",
            }
        )
        assert raw.reproduction.births == {"predator": 0, "prey": 0}
        raw.predators[0].energy = 20.0
        env.step(act(obs))
        assert raw.predators[-1].id() == "predator_2"
    finally:
        env.close()


def test_birth_ends_a_repeated_decision_early():
    env = make(action_repeat=3)
    try:
        obs, raw = setup(env)
        raw.prey[0].energy = 3.0  # grows no further, so only one birth
        raw.predators[0].energy = 13.0  # breeds on the first sub-step
        obs, rewards, _, _, infos = env.step(act(obs))
        assert raw.time_step == 1  # stopped after the birth sub-step
        child = raw.predators[-1]
        assert infos[child.id()]["born"] and infos[child.id()]["parent"] == "predator_0"
        assert infos["predator_0"]["births"] == 1
        assert rewards["predator_0"] == pytest.approx(10.0)
        assert obs[child.id()].shape == (29,)
        # Without a birth the next decision runs all three sub-steps.
        env.step(act(obs))
        assert raw.time_step == 4
    finally:
        env.close()


def test_birth_on_a_later_sub_step_keeps_earlier_rewards_and_flags():
    from grass import GrassPatch

    env = make(action_repeat=3)
    try:
        obs, raw = setup(env)
        parent, hungry = raw.prey
        parent.energy = 7.0  # 6.5 after sub-step 1: no birth yet
        # Edible from sub-step 2: 6.5 + 2 - 0.5 = 8.0 reaches the threshold.
        raw.grass.patches = [GrassPatch(parent.position.copy(), ready_at=2)]
        hungry.energy = 0.4  # starves on sub-step 1
        obs, rewards, terms, _, infos = env.step(act(obs))
        assert raw.time_step == 2  # ended after the birth on sub-step 2
        assert terms[hungry.id()] and infos[hungry.id()]["starved"]
        assert infos[parent.id()]["grass_eaten"] == 1
        assert infos[parent.id()]["births"] == 1
        assert rewards[parent.id()] == pytest.approx(7.0)
        assert infos["prey_2"]["born"] and infos["prey_2"]["parent"] == parent.id()
        assert parent.energy == pytest.approx(8.0 - 3.0)
    finally:
        env.close()


def test_births_leave_aquarium_random_stream_alone():
    import random

    env = make()
    try:
        obs, raw = setup(env)
        raw.prey[0].energy = 9.0
        before = random.getstate()
        env.step(act(obs))
        assert len(raw.prey) == 3
        assert random.getstate() == before  # Aquarium itself draws nothing here
    finally:
        env.close()


def test_rllib_registers_newborns_when_all_known_agents_then_die():
    """RLlib's MultiAgentEpisode ends an episode once every agent it knows is
    done, before adding newcomers. Births must reach it while parents live."""
    from ray.rllib.env.multi_agent_episode import MultiAgentEpisode

    # Thresholds just above initial energy, so parents can starve right
    # after giving birth.
    env = make(
        action_repeat=2,
        reproduction_predator_threshold=5.5,
        reproduction_prey_threshold=3.2,
    )

    def live_actions(obs, terms, truncs):
        return {a: 0 for a in obs if not (terms.get(a) or truncs.get(a))}

    try:
        obs, raw = setup(env)
        episode = MultiAgentEpisode(
            observation_space=env.observation_space,
            action_space=env.action_space,
        )
        episode.add_env_reset(observations=obs, infos={a: {} for a in obs})
        originals = list(raw.predators + raw.prey)
        # Every original survives the first sub-step and starves on the second.
        for entity in raw.predators:
            entity.energy = 1.5  # decay 1.0
        for entity in raw.prey:
            entity.energy = 0.8  # decay 0.5
        # These two give birth on the first sub-step, keeping 0.5 and 0.2.
        raw.predators[0].energy = 6.5
        raw.prey[0].energy = 3.7
        terms, truncs = {}, {}
        for _ in range(3):
            actions = live_actions(obs, terms, truncs)
            obs, rewards, terms, truncs, infos = env.step(actions)
            episode.add_env_step(
                obs, actions, rewards, infos, terminateds=terms, truncateds=truncs
            )
        assert not any(entity.alive for entity in originals)
        assert {"predator_2", "prey_2"} <= set(episode.agent_ids)
        assert not episode.is_done
        assert not terms["__all__"] and not truncs["__all__"]
    finally:
        env.close()


def test_birth_ids_are_independent_per_environment():
    first, second = make(), make()
    try:
        obs, raw = setup(first)
        setup(second)
        raw.predators[0].energy = 20.0
        raw.prey[0].energy = 12.0
        obs, *_ = first.step(act(obs))
        setup(second)  # Aquarium's reset sets its shared class counters to 0
        raw.predators[0].energy = 20.0
        raw.prey[0].energy = 12.0
        first.step(act(obs))
        ids = [p.id() for p in raw.predators]
        assert ids == ["predator_0", "predator_1", "predator_2", "predator_3"]
        assert [p.id() for p in raw.prey] == ["prey_0", "prey_1", "prey_2", "prey_3"]
        assert len(set(raw.agents)) == len(raw.agents)
    finally:
        first.close()
        second.close()


def test_spawn_positions_are_reproducible_for_a_seed():
    positions = []
    for _ in range(2):
        env = make()
        try:
            obs, raw = setup(env, seed=5)
            raw.predators[0].energy = 20.0
            env.step(act(obs))
            child = raw.predators[-1]
            positions.append((child.position.x, child.position.y))
        finally:
            env.close()
    assert positions[0] == positions[1]


def test_newborn_observation_is_stacked():
    env = make(obs_stack=2)
    try:
        obs, raw = setup(env)
        raw.prey[0].energy = 9.0
        obs, _, _, _, _ = env.step(act(obs))
        newborn = obs["prey_2"]
        assert newborn.shape == (66,)
        np.testing.assert_array_equal(newborn[:33], newborn[33:])
        assert env.get_observation_space("prey_2").contains(newborn)
    finally:
        env.close()


@pytest.mark.parametrize("size", [None, 1000, "fit"])
def test_render_window_scales_the_picture_not_the_arena(monkeypatch, size):
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    env = make(render_mode="rgb_array", render_window_size=size)
    try:
        obs, raw = setup(env)
        frame = env.par_env.render()
        # frame is (width, height, 3). A scaled window adds an equally wide
        # population panel; "fit" has no desktop to fit under the dummy driver.
        expected = {None: (800, 800), 1000: (2000, 1000), "fit": (1600, 800)}
        assert frame.shape == (*expected[size], 3)
        if size is not None:
            assert raw.view.screen.get_size() == (800, 800)  # drawn 1:1
            panel = frame[expected[size][1] :]
            # Predator and prey lines are drawn in the panel.
            for color in ((200, 60, 50), (150, 100, 40)):
                assert (panel == np.array(color)).all(axis=-1).sum() > 0
        assert (raw.width, raw.height) == (800, 800)
        # Energy bars are drawn and scaled with the rest of the frame.
        green = (frame == np.array([40, 170, 60])).all(axis=-1).sum()
        assert green > 0
        env.step(act(obs))
        assert env.par_env.render().shape == frame.shape
    finally:
        env.close()


def test_render_window_size_is_validated():
    with pytest.raises(ValueError, match="render_window_size"):
        make(render_window_size="big")


def test_window_is_placed_through_sdl_env_var_not_a_window_object(monkeypatch):
    # A pygame._sdl2 Window object, once freed, is still referenced by every
    # window event and crashed pygame; placement must not create one.
    from grass import place_window

    monkeypatch.delenv("SDL_VIDEO_WINDOW_POS", raising=False)
    place_window((0, 0, 2560, 1020), (980, 980))
    import os

    assert os.environ["SDL_VIDEO_WINDOW_POS"] == "790,0"
    monkeypatch.setenv("SDL_VIDEO_WINDOW_POS", "5,5")  # a user's own choice wins
    place_window((0, 0, 2560, 1020), (980, 980))
    assert os.environ["SDL_VIDEO_WINDOW_POS"] == "5,5"


def test_population_lines_use_each_species_color():
    import pygame

    from grass import PREDATOR_COLOR, PREY_COLOR, draw_population

    pygame.init()
    surface = pygame.Surface((400, 400))
    # 2 predators throughout, prey climbing to 9: prey must be the higher line.
    history = [(0, 2, 1), (500, 2, 9), (1000, 2, 9)]
    draw_population(pygame, surface, history, 1000)
    pixels = pygame.surfarray.array3d(surface)  # (x, y, 3)

    def mean_row(color):
        return np.argwhere((pixels == np.array(color)).all(axis=-1))[:, 1].mean()

    assert mean_row(PREY_COLOR) < mean_row(PREDATOR_COLOR)  # smaller y is higher


def test_env_agents_follow_births_and_deaths_for_rllib():
    from ray.rllib.utils.pre_checks.env import check_multiagent_environments

    env = make()
    try:
        obs, raw = setup(env)
        assert set(env.agents) == set(obs)
        raw.predators[0].energy = 20.0  # gives birth this step
        raw.prey[1].energy = 0.1  # starves this step
        obs, _, terms, _, _ = env.step(act(obs))
        assert "predator_2" in env.agents and "prey_1" in env.agents
        assert set(obs) <= set(env.agents)
        assert set(terms) - {"__all__"} == set(env.agents)
        obs, _, _, _, _ = env.step({agent: 0 for agent in obs if agent != "prey_1"})
        assert "prey_1" not in env.agents  # gone after its final step
        # RLlib's own startup check, on an env whose first step has a birth.
        setup(env)
        raw.prey[0].energy = 9.0
        original_reset = env.reset

        def reset_with_birth_ready(*, seed=None, options=None):
            result = original_reset(seed=seed, options=options)
            raw.prey[0].energy = 9.0
            return result

        env.reset = reset_with_birth_ready
        check_multiagent_environments(env)
    finally:
        env.close()
