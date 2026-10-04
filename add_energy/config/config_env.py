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
    # Energy: animals starve at 0. Untuned starting values.
    "energy_predator_initial": 100.0,
    "energy_prey_initial": 50.0,
    "energy_predator_max": 200.0,  # Cap; own-energy input is energy / max.
    "energy_prey_max": 100.0,
    "energy_predator_decay": 0.15,  # Lost per physics step.
    "energy_prey_decay": 0.1,
    "energy_grass_gain": 20.0,  # Per grass patch eaten.
    "energy_catch_efficiency": 1.0,  # Share of the prey's energy a predator gains.
}
