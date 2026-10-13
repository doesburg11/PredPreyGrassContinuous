"""Reading trained checkpoints: their environment settings and policies.

Checkpoints of add_walls runs in the milestone world work too: this
environment reproduces add_walls exactly (SPEC.md, section 10), and their
settings are translated.
"""

import pickle
from pathlib import Path

from environment import DEFAULT_CONFIG

SPECIES = ("predator", "prey")

# add_walls settings without a counterpart here, with the only values the
# milestone world (and so this environment) allows.
ADD_WALLS_ONLY = {
    "reward_scale": 1.0,
    "prey_reward": 0,
    "prey_punishment": 0,
    "predator_reward": 0,
    "predator_shaping": 0.0,
    "obs_stack": 1,
    "target_speed_actions": True,
    "grass_food_reward": 0.0,
    "grass_predators_observe": True,
    "walls_layout": None,
    "energy_predator_acceleration_cost": 0.0,
    "energy_prey_acceleration_cost": 0.0,
}
IGNORED = {
    "render_mode", "render_window_size", "shaping_gamma",
    "walls_thickness", "walls_rectangles", "walls_occlusion", "walls_rays",
    "walls_visibility_cell",
}


def checkpoint_env_config(checkpoint):
    """The checkpoint's env_config, translated from add_walls if needed."""
    with open(Path(checkpoint) / "class_and_ctor_args.pkl", "rb") as f:
        saved = pickle.load(f)
    config = dict(saved["ctor_args_and_kwargs"][0][0]["env_config"])
    for key, value in ADD_WALLS_ONLY.items():
        if key in config and config.pop(key) != value:
            raise ValueError(f"{key} must be {value!r} for this environment")
    # Aquarium's predator age limit: only one that can never be reached
    # (energy is the only way to starve here).
    max_age = config.pop("predator_max_age", None)
    longest = config.get("max_time_steps", DEFAULT_CONFIG["max_time_steps"]) + 2
    if max_age is not None and max_age <= longest:
        raise ValueError(f"predator_max_age {max_age} would end predators' lives")
    for key in IGNORED:
        config.pop(key, None)
    unknown = set(config) - set(DEFAULT_CONFIG)
    if unknown:
        raise ValueError(f"settings without a counterpart: {sorted(unknown)}")
    return config


def load_modules(checkpoint):
    """Each species' RLModule, on the CPU, in evaluation mode."""
    from ray.rllib.core.rl_module.rl_module import RLModule

    modules = {}
    for species in SPECIES:
        path = (Path(checkpoint) / "learner_group" / "learner" / "rl_module"
                / f"{species}_policy")
        module = RLModule.from_checkpoint(str(path.resolve()))
        module.to("cpu")
        module.eval()
        modules[species] = module
    return modules
