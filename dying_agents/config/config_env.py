"""Environment settings. Edit this dictionary before running train.py."""

config_env = {
    "predator_count": 1,
    "prey_count": 4,
    "max_time_steps": 200,
    "render_mode": None,
    "keep_prey_count_constant": True,  # False: caught prey die permanently.
    "obs_mode": "egocentric",  # "aquarium" or "egocentric".
    "reward_scale": 0.01,
    "predator_shaping": 0.0,  # Potential-based shaping strength; 0 disables it.
    "action_repeat": 1,
    "obs_stack": 1,
    # Omit FOV keys to use Aquarium defaults (prey 120, predator 150 degrees).
    # "prey_fov": 120,
    # "predator_fov": 150,
}
