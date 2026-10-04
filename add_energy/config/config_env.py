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
    "grass_respawn_delay": 400,  # Physics steps; patches regrow in place.
    # Energy, in PredPreyGrass base_environment units. Its per-step rates are
    # divided by 8: crossing this arena takes ~8x as many steps as its 25x25
    # grid (800 units at 4-5 units/step vs 25 cells at 1 cell/step).
    "energy_predator_initial": 5.0,
    "energy_prey_initial": 3.0,
    # PPG has no cap, but reproduction at 12 / 8 drains energy; caps stand in
    # for those thresholds until reproduction exists. Input is energy / max.
    "energy_predator_max": 12.0,
    "energy_prey_max": 8.0,
    "energy_predator_decay": 0.15 / 8,  # Lost per physics step.
    "energy_prey_decay": 0.05 / 8,
    "energy_grass_gain": 2.0,  # A full PPG grass patch (regrowth 2.0 / 0.04 x 8 = 400).
    "energy_catch_efficiency": 1.0,  # PPG: the predator gains all prey energy.
    # Aquarium's own predator starvation clock (death after this many steps
    # without a catch). Kept off so energy is the only way to starve.
    "predator_max_age": 10**9,
}
