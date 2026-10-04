"""Run the recommended frozen species policies without starting Ray workers."""

from pathlib import Path

from ray.rllib.core.rl_module.rl_module import RLModule

PROJECT = Path(__file__).resolve().parents[1]


def recommended_paths(seed=0):
    if seed not in (0, 1, 2):
        raise ValueError("model seed must be 0, 1, or 2")
    suffix = f"seed_{seed}/checkpoint_200/learner_group/learner/rl_module"
    return {
        "predator": PROJECT
        / "runs/training_generalization/fixed/predator_plain"
        / suffix
        / "predator_policy",
        "prey": PROJECT / "runs/prey_predator_pool/pool" / suffix / "prey_policy",
    }


class SpeciesPolicies:
    def __init__(self, paths):
        self.modules = {}
        for species in ("predator", "prey"):
            path = Path(paths[species])
            if not path.is_dir():
                raise FileNotFoundError(
                    f"Missing {species} model: {path}. Trained runs must be present."
                )
            module = RLModule.from_checkpoint(str(path.resolve()))
            module.to("cpu")
            module.eval()
            module.requires_grad_(False)
            if (
                module.observation_space.shape != (28,)
                or module.action_space.n != 16
                or module.is_stateful()
            ):
                raise ValueError(
                    f"Incompatible {species} model: expected a stateless "
                    "28-input, 16-action policy"
                )
            self.modules[species] = module
            print(f"{species} policy: {path}")

    def get_module(self, species):
        return self.modules[species]


def species_mapping(agent_id):
    for species in ("predator", "prey"):
        if agent_id.startswith(species):
            return species
    raise ValueError(f"Unknown species for agent {agent_id!r}")
