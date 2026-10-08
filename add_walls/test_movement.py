"""Direction/target-speed control, movement costs and legacy action behavior."""

import numpy as np
import pytest
from marl_aquarium.env.vector import Vector

from energy import EnergyLayer
from env_wrapper import make_env


def environment(**overrides):
    settings = {
        "predator_count": 1,
        "prey_count": 1,
        "obs_mode": "egocentric",
        "keep_prey_count_constant": False,
        "target_speed_actions": True,
        "predator_max_velocity": 5,
        "prey_max_velocity": 5,
        "predator_max_acceleration": 0.6,
        "prey_max_acceleration": 1,
        "grass_count": 0,
        "energy_predator_initial": 10,
        "energy_prey_initial": 10,
        "energy_predator_max": 20,
        "energy_prey_max": 20,
        "energy_predator_resting_metabolic_cost": 0,
        "energy_prey_resting_metabolic_cost": 0,
        "action_repeat": 1,
        "max_time_steps": 200,
    }
    settings.update(overrides)
    env = make_env(settings)
    obs, _ = env.reset(seed=0)
    raw = env.par_env.aec_env.unwrapped
    for animal, x in zip(raw.predators + raw.prey, (100, 500)):
        animal.position = Vector(x, 400)
        animal.velocity = Vector(0, 0)
        animal.acceleration = Vector(0, 0)
    return env, obs, raw


@pytest.mark.parametrize("index,speed", [(0, 5), (1, 2.5), (2, 0)])
def test_target_speeds_and_braking(index, speed):
    env, obs, raw = environment()
    try:
        assert env.get_action_space("predator_0").n == 48
        for animal in raw.predators + raw.prey:
            animal.velocity = Vector(5, 0)
        for _ in range(15):
            obs, *_ = env.step({a: 4 + 16 * index for a in obs})
        for animal in raw.predators + raw.prey:
            assert animal.velocity.x == pytest.approx(speed)
            assert animal.velocity.y == pytest.approx(0)
    finally:
        env.close()


def test_initial_acceleration_and_turning_are_limited():
    env, obs, raw = environment()
    try:
        obs, *_ = env.step({a: 4 for a in obs})
        assert raw.predators[0].velocity.x == pytest.approx(0.6)
        assert raw.prey[0].velocity.x == pytest.approx(1)
        before = raw.predators[0].velocity.copy()
        env.step({a: 0 for a in obs})
        delta = raw.predators[0].velocity.copy()
        delta.sub(before)
        assert delta.mag() == pytest.approx(0.6)
        assert delta.x < 0 and delta.y < 0
    finally:
        env.close()


def test_energy_uses_actual_speed_and_velocity_change():
    env, obs, raw = environment(
        energy_predator_resting_metabolic_cost=0.1,
        energy_predator_speed_cost=0.2,
        energy_predator_acceleration_cost=0.3,
    )
    try:
        env.step({a: 4 for a in obs})
        assert raw.predators[0].energy == pytest.approx(
            10 - 0.1 - 0.2 * 0.6**2 - 0.3 * 0.6**2
        )
    finally:
        env.close()


def test_stationary_agent_pays_only_resting_cost():
    env, obs, raw = environment(
        energy_predator_resting_metabolic_cost=0.1,
        energy_predator_speed_cost=1,
        energy_predator_acceleration_cost=1,
    )
    try:
        env.step({a: 32 for a in obs})
        assert raw.predators[0].velocity.mag() == 0
        assert raw.predators[0].energy == pytest.approx(9.9)
    finally:
        env.close()


def test_movement_cost_is_charged_per_repeated_physics_step():
    env, obs, raw = environment(
        action_repeat=4,
        energy_predator_speed_cost=0.2,
        energy_predator_acceleration_cost=0.3,
    )
    try:
        env.step({a: 4 for a in obs})
        expected = 10 - sum(
            0.2 * (0.6 * step) ** 2 + 0.3 * 0.6**2 for step in range(1, 5)
        )
        assert raw.predators[0].energy == pytest.approx(expected)
    finally:
        env.close()


@pytest.mark.parametrize(
    "key",
    [
        "predator_speed_cost",
        "prey_speed_cost",
        "predator_acceleration_cost",
        "prey_acceleration_cost",
    ],
)
@pytest.mark.parametrize("value", [-1, np.nan, np.inf, True])
def test_invalid_movement_energy_costs(key, value):
    with pytest.raises(ValueError, match=key):
        EnergyLayer(**{key: value})


def test_legacy_action_space_and_zero_movement_cost_defaults():
    env = make_env({"predator_count": 1, "prey_count": 1})
    try:
        assert env.get_action_space("predator_0").n == 16
        assert EnergyLayer().speed_cost == {"predator": 0, "prey": 0}
    finally:
        env.close()


def test_newborns_receive_target_speed_action_space():
    env, obs, raw = environment(
        reproduction_predator_threshold=12,
        reproduction_prey_threshold=12,
        reproduction_max_predators=3,
        reproduction_max_prey=3,
        reproduction_predator_pool=4,
        reproduction_prey_pool=4,
    )
    try:
        raw.predators[0].energy = 13
        obs, *_ = env.step({a: 32 for a in obs})
        assert "predator_1" in obs
        assert env.get_action_space("predator_1").n == 48
        env.step({a: 32 for a in obs})
    finally:
        env.close()


@pytest.mark.parametrize("value", [1, "yes", None])
def test_invalid_target_speed_toggle(value):
    with pytest.raises(ValueError, match="target_speed_actions"):
        make_env({"target_speed_actions": value})


def test_ppo_module_supports_expanded_categorical_actions():
    import torch
    from ray.rllib.algorithms.ppo.torch.default_ppo_torch_rl_module import (
        DefaultPPOTorchRLModule,
    )
    from ray.rllib.core.columns import Columns
    from ray.rllib.core.rl_module.rl_module import RLModuleSpec

    env, obs, _raw = environment()
    try:
        module = RLModuleSpec(
            module_class=DefaultPPOTorchRLModule,
            observation_space=env.get_observation_space("predator_0"),
            action_space=env.get_action_space("predator_0"),
            model_config={},
        ).build()
        result = module.forward_train(
            {Columns.OBS: torch.as_tensor(obs["predator_0"]).unsqueeze(0)}
        )
        logits = result[Columns.ACTION_DIST_INPUTS]
        assert logits.shape == (1, 48)
        loss = (
            -torch.distributions.Categorical(logits=logits)
            .log_prob(torch.tensor([32]))
            .mean()
        )
        loss.backward()
        assert any(
            p.grad is not None and torch.isfinite(p.grad).all()
            for p in module.parameters()
        )
    finally:
        env.close()


def test_wall_blocked_command_does_not_charge_target_speed():
    env, obs, raw = environment(
        walls_layout="custom",
        walls_rectangles=[(380, 100, 40, 600)],
        energy_predator_resting_metabolic_cost=0.1,
        energy_predator_speed_cost=1,
        energy_predator_acceleration_cost=1,
    )
    try:
        raw.predators[0].position = Vector(364, 400)
        env.step({a: 4 for a in obs})
        assert raw.predators[0].velocity.mag() == pytest.approx(0)
        assert raw.predators[0].energy == pytest.approx(9.9)
    finally:
        env.close()
