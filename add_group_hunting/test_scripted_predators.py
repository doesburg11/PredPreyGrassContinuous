"""Scripted predator strategies for scripted_predators.py."""

from types import SimpleNamespace

import pytest
from marl_aquarium.env.utils import Torus
from marl_aquarium.env.vector import Vector

import scripted_predators as sp
from env_wrapper import _patch_torus_view


def animal(agent, x, y, vx=0.0, vy=0.0, heading=0.0):
    return SimpleNamespace(
        position=Vector(x, y),
        velocity=Vector(vx, vy),
        orientation_angle=heading,
        id=lambda: agent,
    )


def arena(predators, prey):
    raw = SimpleNamespace(
        width=800,
        height=800,
        torus=Torus(800, 800),
        predators=predators,
        prey=prey,
        predator_view_distance=200,
        predator_fov=150,
        predator_max_velocity=5,
    )
    _patch_torus_view(raw)
    return raw


@pytest.mark.parametrize(
    "dx, dy, action",
    [(0, -1, 0), (1, 0, 4), (0, 1, 8), (-1, 0, 12), (1, -1, 2)],
)
def test_action_towards_matches_aquarium_directions(dx, dy, action):
    assert sp.action_towards(dx, dy) == action


def test_chase_steers_at_the_nearest_visible_prey():
    # Aquarium heading 0 faces +x: the prey to the right is in view.
    predator = animal("predator_0", 400, 400, vx=1, heading=0)
    raw = arena([predator], [animal("prey_0", 500, 400), animal("prey_1", 300, 400)])
    actions = sp.ScriptedPredators("chase").act(raw, ["predator_0"])
    assert actions == {"predator_0": 4}  # right, towards prey_0


def test_chase_keeps_heading_without_visible_prey_but_chase_all_sees_everything():
    predator = animal("predator_0", 400, 400, vx=1, heading=0)  # facing +x
    raw = arena([predator], [animal("prey_0", 300, 400)])  # behind it
    assert sp.ScriptedPredators("chase").act(raw, ["predator_0"]) == {"predator_0": 4}
    assert sp.ScriptedPredators("chase_all").act(raw, ["predator_0"]) == {
        "predator_0": 12
    }


def test_intercept_leads_a_moving_prey():
    predator = animal("predator_0", 400, 400, vx=1, heading=0)
    # 100 units ahead, moving up (-y) at 5 units per step.
    prey = animal("prey_0", 500, 400, vy=-5)
    raw = arena([predator], [prey])
    target = sp.ScriptedPredators("intercept").target(raw, predator)
    assert target is not None
    dx, dy = target
    assert dx == pytest.approx(100) and dy == pytest.approx(-100)
    assert sp.ScriptedPredators("chase").target(raw, predator) == (100, 0)


def test_dead_predators_get_an_ignored_action():
    raw = arena([], [animal("prey_0", 300, 400)])
    assert sp.ScriptedPredators("chase").act(raw, ["predator_3"]) == {"predator_3": 0}


def test_unknown_strategy_is_rejected():
    with pytest.raises(ValueError):
        sp.ScriptedPredators("ambush")


def test_scripted_predators_play_a_real_episode():
    from ray.rllib.algorithms.ppo.torch.default_ppo_torch_rl_module import (
        DefaultPPOTorchRLModule,
    )
    from ray.rllib.core.rl_module.rl_module import RLModuleSpec

    import tournament
    from config.config_env import config_env
    from env_wrapper import make_env

    env = make_env({**config_env, "max_time_steps": 80, "render_mode": None})
    try:
        prey = RLModuleSpec(
            module_class=DefaultPPOTorchRLModule,
            observation_space=env.get_observation_space("prey_0"),
            action_space=env.get_action_space("prey_0"),
            model_config={},
        ).build()
        modules = {"predator": sp.ScriptedPredators("intercept_all"), "prey": prey}
        result = tournament.play_episode(env, modules, seed=1)
    finally:
        env.close()
    assert result["outcome"] in tournament.OUTCOMES
    assert int(result["physics_steps"]) > 0
    rows = [{"predator": "chase", **result}]
    assert sp.summary(rows)[0]["episodes"] == 1
