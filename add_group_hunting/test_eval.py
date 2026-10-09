"""eval.py reads config/config_eval.py instead of command-line arguments."""

import subprocess
import sys
from pathlib import Path

import pytest

import eval as evaluation


def test_default_settings_are_valid():
    settings = evaluation.load_settings()
    assert settings.source in evaluation.SOURCES
    assert settings.render in evaluation.RENDERS


@pytest.mark.parametrize(
    "name, value",
    [
        ("source", "best"),
        ("render", "screen"),
        ("episodes", 0),
        ("window_fps", -8),
        ("prey_count", 2.5),
        ("prey_fov", 400),
        ("stochastic", "yes"),
        ("model_seed", 3),
        ("seed", 1.5),
        ("env_overrides", ["grass_clustered"]),
    ],
)
def test_invalid_settings_are_rejected(monkeypatch, name, value):
    monkeypatch.setitem(evaluation.config_eval, name, value)
    with pytest.raises(ValueError, match=name):
        evaluation.load_settings()


def test_checkpoint_source_needs_a_path(monkeypatch):
    monkeypatch.setitem(evaluation.config_eval, "source", "checkpoint")
    monkeypatch.setitem(evaluation.config_eval, "checkpoint", None)
    with pytest.raises(ValueError, match="checkpoint path"):
        evaluation.load_settings()


def test_settings_are_copied():
    settings = evaluation.load_settings()
    settings.episodes = 99
    assert evaluation.config_eval["episodes"] != 99


def test_command_line_arguments_are_refused():
    script = Path(evaluation.__file__)
    result = subprocess.run(
        [sys.executable, str(script), "--random"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode != 0
    assert "config/config_eval.py" in result.stderr


def test_random_run_from_settings(monkeypatch, tmp_path, capsys):
    csv = tmp_path / "population.csv"
    for name, value in {
        "source": "random",
        "render": "none",
        "episodes": 2,
        "max_time_steps": 40,
        "population_csv": str(csv),
    }.items():
        monkeypatch.setitem(evaluation.config_eval, name, value)
    evaluation.main()
    out = capsys.readouterr().out
    assert "episode 1/2" in out and "episode 2/2" in out
    assert csv.read_text().startswith("episode,step,predators,prey")


def test_env_overrides_replace_environment_settings(monkeypatch):
    built = []
    real_make_env = evaluation.make_env

    def spy(config):
        built.append(dict(config))
        return real_make_env(config)

    monkeypatch.setattr(evaluation, "make_env", spy)
    for name, value in {
        "source": "random",
        "render": "none",
        "max_time_steps": 16,
        "env_overrides": {"grass_clustered": True, "grass_cluster_count": 3},
    }.items():
        monkeypatch.setitem(evaluation.config_eval, name, value)
    evaluation.main()
    assert built[0]["grass_clustered"] is True
    assert built[0]["grass_cluster_count"] == 3


def test_view_cones_follow_one_survivor_per_species():
    from types import SimpleNamespace

    predators = [SimpleNamespace(alive=True) for _ in range(2)]
    prey = [SimpleNamespace(alive=True) for _ in range(2)]
    drawn = []
    raw = SimpleNamespace(
        predators=predators,
        prey=prey,
        draw_view_cone_in_torus=lambda animal, distance, fov: drawn.append(animal),
    )
    evaluation.limit_view_cones(raw)

    def frame():
        drawn.clear()
        for animal in raw.prey + raw.predators:
            raw.draw_view_cone_in_torus(animal, 200, 150)
        return [id(animal) for animal in drawn]

    assert frame() == [id(prey[0]), id(predators[0])]
    # Reordering or births must not switch a still-living selected animal.
    raw.predators = list(reversed(predators))
    assert frame() == [id(prey[0]), id(predators[0])]
    predators[0].alive = False
    prey[0].alive = False
    raw.prey = [prey[1]]
    raw.predators = [predators[1]]
    assert frame() == [id(prey[1]), id(predators[1])]
    raw.predators = []
    assert frame() == [id(prey[1])]
    # Reset can reuse IDs; selection follows the new objects.
    new_prey = SimpleNamespace(alive=True)
    raw.prey = [new_prey]
    assert frame() == [id(new_prey)]
