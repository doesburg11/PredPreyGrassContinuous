"""Environment settings. Edit this dictionary before running train.py."""

config_env = {
    # Initial populations (PredPreyGrass base_environment: 6 and 8).
    "predator_count": 6,
    "prey_count": 8,
    # Time scale. One PredPreyGrass grid step (1 cell, ~32 units here) takes
    # ~8 physics steps at 4-5 units/step, which is why the energy rates below
    # are divided by 8. Its 1000-step episodes therefore last 8000 physics
    # steps here. max_time_steps counts physics steps.
    "max_time_steps": 8000,
    "render_mode": None,
    # Viewer window: "fit" scales the 800x800 arena up to fit the screen, an
    # int is the arena's side in pixels, None draws 1:1 without the predator/
    # prey time series shown right of the arena. Display only.
    "render_window_size": "fit",
    # Rewards: PredPreyGrass's sparse scheme, +10 per birth and nothing else.
    # Aquarium's own rewards and the grass food reward are switched off.
    "reward_scale": 1.0,
    "prey_reward": 0,  # Aquarium: reward per step alive.
    "prey_punishment": 0,  # Aquarium: penalty when caught.
    "predator_reward": 0,  # Aquarium: reward per catch.
    "predator_shaping": 0.0,  # Potential-based shaping strength; 0 disables it.
    # Each decision is applied for 8 physics steps, so one decision equals one
    # PredPreyGrass step: ~1000 decisions per episode, and gamma (0.99 per
    # decision in config_ppo.py) discounts per grid step as PredPreyGrass
    # does. A sub-step with a birth ends its decision early (see README).
    "action_repeat": 8,
    "obs_stack": 1,
    # 48 actions: direction + 16 * speed index (0: full, 1: half, 2: stop).
    "target_speed_actions": True,
    # Movement. Prey speed raised to the predators' 5, as in PredPreyGrass.
    # Speeds are in units per physics step;
    # the energy time scale assumes ~4-5 (one ~32-unit grid cell per ~8 steps).
    "predator_max_velocity": 5,  # Top speed.
    "prey_max_velocity": 5,
    "predator_max_acceleration": 0.6,  # Speed gained per step.
    "prey_max_acceleration": 1,
    "predator_max_steer_force": 0.6,  # How sharply it can turn.
    "prey_max_steer_force": 0.6,
    # Body radii (Aquarium's defaults: 30 and 20). A predator catches a prey
    # when their centres are closer than the sum, now 32 units (~1 grid cell;
    # in PredPreyGrass it has to step onto the prey's cell). Same-species
    # animals bump apart at the sum of their radii. Sprites keep their size.
    "predator_radius": 16,
    "prey_radius": 16,
    "prey_fov": 240,  # view cone in degrees
    "predator_fov": 150,
    "prey_view_distance": 267,  # View cone radius in simulation units.
    "predator_view_distance": 200,
    # One 25x25 grid cell is ~32 units here, so PredPreyGrass's 100 grass
    # cells become 100 patches.
    "grass_count": 100,
    "grass_consume_radius": 12.0,
    "grass_food_reward": 0.0,  # Raw reward, before reward_scale.
    "grass_respawn_delay": 400,  # Physics steps until an eaten patch regrows.
    # False: patches placed uniformly at random. True: grass_cluster_count
    # clusters, each centred in its own cell of a grid over the arena, with the
    # patches split evenly over them and scattered around each centre (normal
    # offsets with standard deviation grass_cluster_spread, in arena units).
    "grass_clustered": True,
    "grass_cluster_count": 5,
    "grass_cluster_spread": 48.0,
    # False: an eaten patch regrows in the same spot. True: at a new random
    # spot (within its own cluster when clustered, so clusters persist).
    "grass_random_respawn": False,
    # Seed dispersal (needs grass_clustered; not with grass_random_respawn): an
    # eaten patch regrows next to a living patch of its cluster (normal offsets
    # with standard deviation grass_dispersal_distance), so clusters creep away
    # from where grazing removes their edge.
    "grass_dispersal": True,
    "grass_dispersal_distance": 24.0,
    # Overgrazing (needs grass_clustered): a cluster grazed down to this share
    # of its patches (0.0: grazed bare) dies back and, after
    # grass_overgrazing_delay physics steps, regrows around a new centre in a
    # grid cell no other cluster uses.
    "grass_overgrazing": True,
    "grass_overgrazing_threshold": 0.0,
    "grass_overgrazing_delay": 400,
    # Predators also observe the nearest available patch in their view cone
    # (4 inputs, like prey), as in PredPreyGrass, where both species see
    # grass. False keeps predators blind to grass (runs before this setting).
    "grass_predators_observe": True,
    # Walls (walls.py): obstacles that block movement and sight, so animals
    # can hide, as in PredPreyGrass's walls_occlusion. None switches them off;
    # "blocks" is a forest of 16 square blocks, "chambers" four rooms with
    # doorways, "custom" uses walls_rectangles [(x, y, width, height), ...].
    "walls_layout": "blocks",
    "walls_thickness": 32.0,  # Arena units; at least 16 (one cell is ~32).
    "walls_rectangles": [],
    "walls_occlusion": True,  # Walls block sight (animals and grass).
    "walls_rays": 8,  # Wall-distance inputs per agent, in action directions.
    "walls_visibility_cell": 32.0,  # Resolution of the line-of-sight table.
    # Energy, in PredPreyGrass base_environment units. Its per-step rates are
    # divided by 8: crossing this arena takes ~8x as many steps as its 25x25
    # grid (800 units at 4-5 units/step vs 25 cells at 1 cell/step).
    "energy_predator_initial": 5.0,
    "energy_prey_initial": 3.0,
    # PPG has no cap; these sit well above the reproduction thresholds and
    # only scale the own-energy input (energy / max).
    "energy_predator_max": 24.0,
    "energy_prey_max": 16.0,
    "energy_predator_decay": 0.15 / 8,  # Lost per physics step.
    "energy_prey_decay": 0.05 / 8,
    # Per physics step: resting decay + speed_cost * speed^2
    # + acceleration_cost * change_in_velocity^2 (actual arena units).
    # Full speed adds half the resting cost; these are initial tuning values.
    "energy_predator_speed_cost": 0.15 / 8 / (2 * 5**2),
    "energy_prey_speed_cost": 0.05 / 8 / (2 * 5**2),
    "energy_predator_acceleration_cost": 0.001875,
    "energy_prey_acceleration_cost": 0.000625,
    "energy_grass_gain": 2.0,  # A full PPG grass patch (regrowth 2.0 / 0.04 x 8 = 400).
    # Share of a caught prey's energy the catching predator gains (PPG: 1.0).
    "energy_catch_efficiency_predator": 0.5,
    # Aquarium's own predator starvation clock (death after this many steps
    # without a catch). Kept off so energy is the only way to starve.
    "predator_max_age": 10**9,
    # Reproduction (PredPreyGrass base_environment): at the threshold an
    # animal has one offspring with the initial energy, paid by the parent.
    "reproduction_predator_threshold": 12.0,
    "reproduction_prey_threshold": 8.0,
    "reproduction_predator_reward": 10.0,
    "reproduction_prey_reward": 10.0,
    # Population caps (Aquarium's observations scale with animals squared).
    "reproduction_max_predators": 50,
    "reproduction_max_prey": 100,
    # Agent IDs per species, never reused within an episode; births stop
    # when a pool runs out. PredPreyGrass uses 2000.
    "reproduction_predator_pool": 2000,
    "reproduction_prey_pool": 2000,
}
