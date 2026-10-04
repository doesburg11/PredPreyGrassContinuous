import numpy as np
import pytest
import torch

from eval import select_actions
from learned_policies import SpeciesPolicies, recommended_paths, species_mapping


def test_paths_and_species():
    for seed in range(3):
        paths = recommended_paths(seed)
        assert f"seed_{seed}" in str(paths["predator"])
        assert "prey_predator_pool/pool" in str(paths["prey"])
    assert species_mapping("predator_0") == "predator"
    assert species_mapping("prey_3") == "prey"
    with pytest.raises(ValueError):
        species_mapping("fish_0")
    with pytest.raises(ValueError):
        recommended_paths(3)


def test_missing_checkpoint(tmp_path):
    with pytest.raises(FileNotFoundError, match="Missing predator model"):
        SpeciesPolicies(
            {"predator": tmp_path / "missing", "prey": tmp_path / "missing"}
        )


def test_separate_species_actions():
    class Module:
        def __init__(self, action):
            self.action = action

        def forward_inference(self, batch):
            logits = torch.full((1, 16), -1000.0)
            logits[0, self.action] = 1000.0
            return {"action_dist_inputs": logits}

    class Policies:
        def get_module(self, species):
            return Module(2 if species == "predator" else 11)

    obs = {"predator_0": np.zeros(28), "prey_0": np.zeros(28)}
    for stochastic in (False, True):
        assert select_actions(Policies(), obs, species_mapping, stochastic) == {
            "predator_0": 2,
            "prey_0": 11,
        }
