"""Checkpoint selection, summary matrices, and one real tournament episode."""

import numpy as np
import pytest

import tournament
from tournament import (
    CATCH_RISK,
    adaptation_trends,
    select_iterations,
    summarize,
)


def test_select_iterations_keeps_spacing_and_the_newest():
    available = [10, 20, 50, 60, 100, 150, 170]
    assert select_iterations(available, every=50) == [50, 100, 150, 170]
    assert select_iterations(available, every=None) == available
    assert select_iterations(available, iterations=[100, 20]) == [20, 100]
    with pytest.raises(ValueError, match="No checkpoint"):
        select_iterations(available, iterations=[30])
    with pytest.raises(ValueError, match="no checkpoints"):
        select_iterations([], every=50)


def test_checkpoint_iterations_reads_iter_directories(tmp_path):
    for name in ("iter_000010", "iter_000200", "not_a_checkpoint"):
        (tmp_path / "checkpoint" / name).mkdir(parents=True)
    assert tournament.checkpoint_iterations(tmp_path) == [10, 200]


def row(predator_iter, prey_iter, steps, caught, prey_exposure, outcome):
    return {
        "predator_iter": predator_iter,
        "prey_iter": prey_iter,
        "physics_steps": steps,
        "prey_caught": caught,
        "prey_exposure": prey_exposure,
        "outcome": outcome,
    }


def test_summary_catch_risk_is_per_prey_step_and_missing_is_nan():
    rows = [
        row(1, 1, 1000, 10, 10_000.0, "prey_extinct"),
        row(1, 1, 3000, 30, 30_000.0, "time_limit"),
        # Twice the catches, but twice the prey exposure: the same risk.
        row(1, 2, 1000, 20, 20_000.0, "both_extinct"),
        # (2, 2) has no episodes.
        row(2, 1, 2000, 0, 5_000.0, "predators_extinct"),
    ]
    matrices = summarize(rows, [1, 2])
    np.testing.assert_allclose(
        matrices["episode length (physics steps)"], [[2000, 1000], [2000, np.nan]]
    )
    np.testing.assert_allclose(matrices[CATCH_RISK], [[1.0, 1.0], [0.0, np.nan]])
    np.testing.assert_allclose(
        matrices["prey extinct (fraction)"], [[0.5, 0], [0, np.nan]]
    )
    np.testing.assert_allclose(
        matrices["both extinct (fraction)"], [[0, 1], [0, np.nan]]
    )
    np.testing.assert_allclose(
        matrices["predators extinct (fraction)"], [[0, 0], [1, np.nan]]
    )


def test_classify_outcomes_including_simultaneous_extinction():
    assert tournament.classify(predators=3, prey=5) == "time_limit"
    assert tournament.classify(predators=3, prey=0) == "prey_extinct"
    assert tournament.classify(predators=0, prey=5) == "predators_extinct"
    assert tournament.classify(predators=0, prey=0) == "both_extinct"


def test_adaptation_trends_need_both_species_to_improve():
    # Rows: predator checkpoints; columns: prey checkpoints.
    both = np.array([[2.0, 1.0, 0.5], [3.0, 2.0, 1.0], [4.0, 3.0, 2.0]])
    assert adaptation_trends(both) == {"prey": 1.0, "predator": 1.0}
    # Only predators improve: prey columns are identical, so no prey trend,
    # although newer predators beat older prey.
    predators_only = np.array([[1.0, 1.0, 1.0], [2.0, 2.0, 2.0], [3.0, 3.0, 3.0]])
    trends = adaptation_trends(predators_only)
    assert trends is not None
    assert trends["predator"] == 1.0 and np.isnan(trends["prey"])
    # A missing matchup is skipped, not counted as zero risk.
    gap = both.copy()
    gap[0, 2] = np.nan
    assert adaptation_trends(gap) == {"prey": 1.0, "predator": 1.0}
    assert adaptation_trends(np.array([[1.0]])) is None


def test_play_episode_with_untrained_policies():
    from ray.rllib.algorithms.ppo.torch.default_ppo_torch_rl_module import (
        DefaultPPOTorchRLModule,
    )
    from ray.rllib.core.rl_module.rl_module import RLModuleSpec

    from config.config_env import config_env
    from env_wrapper import make_env

    env = make_env(
        {
            **config_env,
            "max_time_steps": 40,
            "action_repeat": 4,
            "render_mode": None,
        }
    )
    try:
        modules = {
            species: RLModuleSpec(
                module_class=DefaultPPOTorchRLModule,
                observation_space=env.get_observation_space(f"{species}_0"),
                action_space=env.get_action_space(f"{species}_0"),
                model_config={},
            ).build()
            for species in ("predator", "prey")
        }
        first = tournament.play_episode(env, modules, seed=3)
        again = tournament.play_episode(env, modules, seed=3)
    finally:
        env.close()
    assert first["outcome"] in tournament.OUTCOMES
    assert int(first["physics_steps"]) > 0 and int(first["decisions"]) > 0
    assert set(tournament.FIELDS) - {
        "predator_iter",
        "prey_iter",
        "episode",
        "seed",
    } == set(first)
    assert first == again  # same seed, same episode
