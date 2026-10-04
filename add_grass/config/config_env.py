"""Environment settings. Edit this dictionary before running train.py."""

config_env = {
    "predator_count": 1,
    "prey_count": 4,
    "max_time_steps": 1000,
    "render_mode": None,
    "keep_prey_count_constant": False,  # False: caught prey die permanently.
    "obs_mode": "egocentric",  # "aquarium" or "egocentric".
    "reward_scale": 0.01,
    "predator_shaping": 0.0,  # Potential-based shaping strength; 0 disables it.
    "action_repeat": 1,
    "obs_stack": 1,
    "prey_fov": 240,  # view cone in degrees
    "predator_fov": 150,
    "prey_view_distance": 200,  # View cone radius in simulation units.
    "predator_view_distance": 200,
    "grass_count": 24,  # Set to 0 for a grass-free comparison (same obs shape).
    "grass_consume_radius": 12.0,
    "grass_food_reward": 10.0,  # Raw reward, before reward_scale.
    "grass_respawn_delay": 100,  # Physics steps; patches regrow in place.
}
