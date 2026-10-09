"""Group hunting: catch chances, failed attacks, cooldowns and shared catches."""

import pytest
from marl_aquarium.env.vector import Vector

from env_wrapper import make_env
from group_hunting import GroupHunting

BASE = {
    "predator_count": 2,
    "prey_count": 2,
    "obs_mode": "egocentric",
    "keep_prey_count_constant": False,
    "grass_count": 0,
    "energy_predator_initial": 6.0,
    "energy_prey_initial": 6.0,
    "energy_predator_max": 24.0,
    "energy_prey_max": 16.0,
    "energy_predator_resting_metabolic_cost": 0.0,
    "energy_prey_resting_metabolic_cost": 0.0,
    "energy_catch_efficiency_predator": 1.0,
    "group_hunting_radius": 64.0,
    "group_hunting_prey_strength": 2.0,
    "group_hunting_steepness": 2.0,
    "group_hunting_cooldown": 16,
}


def arrange(env, helper=True, seed=0):
    """predator_0 touches prey_0 (Aquarium's radii 30 + 20 = 50 apart at most);
    predator_1 is a helper within 64 of prey_0 but not touching, or far away;
    prey_1 is far from everyone."""
    env.reset(seed=seed)
    raw = env.par_env.aec_env.unwrapped
    attacker, other = raw.predators
    prey, bystander = raw.prey
    attacker.position = Vector(400, 400)
    prey.position = Vector(430, 400)
    other.position = Vector(430, 455) if helper else Vector(100, 100)
    bystander.position = Vector(700, 700)
    for entity in raw.predators + raw.prey:
        entity.velocity = Vector(0, 0)
    return raw, attacker, other, prey


def idle(env):
    return {agent: 0 for agent in env.agents}


@pytest.mark.parametrize(
    "settings",
    [
        {"radius": 0},
        {"prey_strength": -1},
        {"steepness": float("nan")},
        {"cooldown": 1.5},
        {"cooldown": -1},
        {"radius": True},
    ],
)
def test_invalid_settings(settings):
    with pytest.raises(ValueError):
        GroupHunting(**settings)


def test_group_hunting_requires_energy():
    config = {k: v for k, v in BASE.items() if not k.startswith("energy_")}
    with pytest.raises(ValueError, match="energy"):
        make_env(config)


def test_chance_is_a_contest_between_hunters_and_prey():
    hunt = GroupHunting(prey_strength=2.0, steepness=2.0)
    assert hunt.chance(6, 6) == pytest.approx(36 / (36 + 144))  # alone: 0.2
    assert hunt.chance(12, 6) == pytest.approx(0.5)  # a pair
    assert hunt.chance(6, 10) == pytest.approx(36 / (36 + 400))
    # Steepness 2 and strong prey: a pair member's share beats hunting alone.
    assert hunt.chance(12, 10) / 2 > hunt.chance(6, 10)
    # Steepness 1: never.
    flat = GroupHunting(prey_strength=2.0, steepness=1.0)
    assert flat.chance(12, 10) / 2 < flat.chance(6, 10)
    assert GroupHunting(prey_strength=0).chance(1, 100) == 1.0
    assert hunt.chance(6, 0) == 1.0
    assert hunt.chance(0, 6) == 0.0
    assert GroupHunting(steepness=50).chance(1e6, 1.0) == pytest.approx(1.0)
    # Large steepness in either direction: no overflow.
    assert GroupHunting(steepness=1024).chance(6, 6) == pytest.approx(0.0)
    assert GroupHunting(steepness=1024).chance(600, 6) == pytest.approx(1.0)
    assert GroupHunting(prey_strength=1, steepness=1024).chance(6, 6) == 0.5


def test_every_attack_succeeds_without_prey_strength():
    env = make_env({**BASE, "group_hunting_prey_strength": 0.0})
    try:
        raw, attacker, other, prey = arrange(env, helper=False)
        env.step(idle(env))
        assert not prey.alive
        assert attacker.energy == pytest.approx(6.0 + 6.0)  # alone: all of it
        assert other.energy == pytest.approx(6.0)
        assert raw.group_hunting.stats["catches_by_hunters"] == {1: 1}
    finally:
        env.close()


def test_catch_is_shared_equally_among_hunters():
    env = make_env({**BASE, "group_hunting_prey_strength": 0.0})
    try:
        raw, attacker, other, prey = arrange(env, helper=True)
        env.step(idle(env))
        assert not prey.alive
        assert attacker.energy == pytest.approx(6.0 + 3.0)
        assert other.energy == pytest.approx(6.0 + 3.0)
        assert raw.group_hunting.stats["catches_by_hunters"] == {2: 1}
    finally:
        env.close()


def test_only_predators_within_the_radius_share_the_catch():
    # Radius 1: below the contact distance, so only the attacker hunts, even
    # though the other predator touches the prey too.
    env = make_env(
        {**BASE, "group_hunting_prey_strength": 0.0, "group_hunting_radius": 1.0}
    )
    try:
        raw, attacker, other, prey = arrange(env, helper=False)
        other.position = Vector(460, 400)  # touches the prey from the other side
        env.step(idle(env))
        assert not prey.alive
        gains = sorted([attacker.energy - 6.0, other.energy - 6.0])
        assert gains == pytest.approx([0.0, 6.0])  # one hunter, all of it
        assert raw.group_hunting.stats["catches_by_hunters"] == {1: 1}
    finally:
        env.close()


def test_hunting_with_target_speed_and_action_repeat():
    env = make_env(
        {
            **BASE,
            "group_hunting_prey_strength": 0.0,
            "target_speed_actions": True,
            "action_repeat": 4,
        }
    )
    try:
        raw, attacker, other, prey = arrange(env, helper=True)
        env.step({agent: 32 for agent in env.agents})  # everyone stops
        assert not prey.alive
        assert attacker.energy == pytest.approx(9.0)
        assert other.energy == pytest.approx(9.0)
    finally:
        env.close()


def test_failed_attack_spares_the_prey_and_starts_a_cooldown():
    # An overwhelming prey: every attack fails.
    env = make_env({**BASE, "group_hunting_prey_strength": 1e6})
    try:
        raw, attacker, _, prey = arrange(env, helper=False)
        env.step(idle(env))
        assert prey.alive
        assert raw.group_hunting.stats == {
            "attacks": 1,
            "failed": 1,
            "catches_by_hunters": {},
        }
        # Still touching, but blocked for the cooldown: no new attack.
        for _ in range(16):
            prey.position = Vector(attacker.position.x + 30, attacker.position.y)
            env.step(idle(env))
        assert raw.group_hunting.stats["attacks"] == 1
        prey.position = Vector(attacker.position.x + 30, attacker.position.y)
        env.step(idle(env))
        assert raw.group_hunting.stats["attacks"] == 2
    finally:
        env.close()


def test_failed_attack_lets_another_touching_predator_try():
    env = make_env({**BASE, "group_hunting_prey_strength": 1e6})
    try:
        raw, attacker, other, prey = arrange(env, helper=False)
        other.position = Vector(460, 400)  # touches the prey from the other side
        env.step(idle(env))
        assert prey.alive
        assert raw.group_hunting.stats["attacks"] == 2
    finally:
        env.close()


def test_chance_uses_the_hunters_summed_energy():
    # Equal chance per attack with the helper: summed energy 12 against prey 6
    # with prey_strength 1 and steepness 1 gives 2/3; alone 1/2.
    rates = {}
    for helper in (False, True):
        caught = 0
        env = make_env(
            {
                **BASE,
                "group_hunting_prey_strength": 1.0,
                "group_hunting_steepness": 1.0,
            }
        )
        try:
            for seed in range(400):
                _, _, _, prey = arrange(env, helper=helper, seed=seed)
                env.step(idle(env))
                caught += not prey.alive
        finally:
            env.close()
        rates[helper] = caught / 400
    assert rates[False] == pytest.approx(1 / 2, abs=0.07)
    assert rates[True] == pytest.approx(2 / 3, abs=0.07)


def test_same_seed_same_outcome():
    outcomes = []
    for _ in range(2):
        env = make_env({**BASE, "group_hunting_prey_strength": 1.0})
        try:
            result = []
            for seed in range(20):
                _, _, _, prey = arrange(env, seed=seed)
                env.step(idle(env))
                result.append(prey.alive)
            outcomes.append(result)
        finally:
            env.close()
    assert outcomes[0] == outcomes[1]
    assert any(outcomes[0]) and not all(outcomes[0])


def test_reset_clears_cooldowns_and_statistics():
    env = make_env({**BASE, "group_hunting_prey_strength": 1e6})
    try:
        raw, *_ = arrange(env, helper=False)
        env.step(idle(env))
        assert raw.group_hunting.blocked and raw.group_hunting.stats["attacks"]
        env.reset(seed=1)
        assert not raw.group_hunting.blocked
        assert raw.group_hunting.stats["attacks"] == 0
    finally:
        env.close()
