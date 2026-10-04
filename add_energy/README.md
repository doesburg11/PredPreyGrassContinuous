# Add energy

A contained copy of `add_grass` that adds a metabolic energy model, the next
step towards [PredPreyGrass](https://github.com/doesburg11/PredPreyGrass).
Animals lose energy every step. Prey gain energy by eating grass, and predators
gain it by catching prey. An animal whose energy reaches zero starves and dies
for good. Rewards are unchanged from `add_grass`, so energy affects learning
only through death and the new observation input. There is no reproduction yet.

Edit `config/config_env.py` and `config/config_ppo.py`, then run from the
repository root:

```bash
.conda/bin/python add_energy/train.py
```

Training defaults to 1,000 iterations. Outputs are saved in
`add_energy/runs/add_energy_<Amsterdam timestamp>/`.

## Energy settings

| Setting | Default | Meaning |
|---|---:|---|
| `energy_predator_initial` | 100 | Predator energy at reset |
| `energy_prey_initial` | 50 | Prey energy at reset |
| `energy_predator_max` | 200 | Predator energy cap |
| `energy_prey_max` | 100 | Prey energy cap |
| `energy_predator_decay` | 0.15 | Predator energy lost per physics step |
| `energy_prey_decay` | 0.1 | Prey energy lost per physics step |
| `energy_grass_gain` | 20 | Prey energy per grass patch eaten |
| `energy_catch_efficiency` | 1.0 | Share of a caught prey's energy its catcher gains |

These defaults have not been tuned. Without food, a prey starves after 500
steps and a predator after about 667 steps.

Each step, energy is updated after Aquarium's movement and catches and after
grass is eaten:

1. Catching predators gain energy from their catches. The catcher is the
   predator Aquarium's collision test found.
2. Prey gain energy for each grass patch they ate.
3. Every living animal loses its species' decay.
4. Any animal at or below zero energy starves.

Energy is capped at the species maximum. A starved animal is removed and
terminated, with `infos[agent]["starved"] = True`. The episode ends once no
predators or no prey are left. If the prey are gone, every agent is
terminated. If the predators are gone, the surviving prey are truncated.
Aquarium's own predator starvation clock (`predator_max_age`) is disabled
whenever energy is configured. Energy requires grass settings and
`keep_prey_count_constant = False`.

Every agent gets one extra observation input at the end: its own energy as a
fraction of its species maximum. Prey inputs are 33 values and predator
inputs are 29 values, before observation stacking. `infos[agent]["energy"]`
reports the raw value. The viewer draws an energy bar above each animal.

## Grass settings

| Setting | Default | Meaning |
|---|---:|---|
| `grass_count` | 24 | Number of stationary patches; 0 disables grass |
| `grass_consume_radius` | 12 | Toroidal distance within which prey eat a patch |
| `grass_food_reward` | 10 | Reward per patch, before global reward scaling |
| `grass_respawn_delay` | 100 | Physics steps before a consumed patch regrows |

Consumption is checked after movement and predator captures. Dead prey cannot
consume grass. Each available patch rewards the closest living prey within its
radius; ties use agent ID. A prey can consume several nearby patches in one
step. Patches regrow at their original locations. Reset regenerates locations
using a separate seeded RNG, without changing Aquarium's initialization stream.
At reward scale 0.01, one food reward of 10 becomes 0.1 for training.

## Observations and display

Prey observations append four values for the nearest available patch inside
their view cone: `[seen, wrapped_dx/view_distance, wrapped_dy/view_distance,
distance/view_distance]`. These match the existing egocentric offset encoding.
Invisible or absent grass produces four zeros. Prey inputs are 32 values rather
than 28, and predator inputs stay at 28. The energy input then adds one
more to each, for 33 and 29.
Setting `grass_count = 0` retains the same observation shapes for ablation runs.
Grass requires `obs_mode = "egocentric"`.

Checkpoints from `add_grass` (32/28 inputs) cannot be used with the energy
input layout. New checkpoints retain their grass settings for evaluation.
Available patches use the transparent `assets/grass-transparent.png` sprite
in the Aquarium window/video, derived from `assets/grass.png`.

```bash
.conda/bin/python add_energy/eval.py --random --draw-view-cones
.conda/bin/python add_energy/eval.py --checkpoint add_energy/runs/<run>/checkpoint \
    --stochastic --draw-view-cones --episodes 3
```

Evaluation prints consumed grass (`grass_eaten`) and starvations per species
(`starved_predators`, `starved_prey`) alongside prey catches and returns.
Starved prey are not counted in `prey_eaten`. Rewards are reported unscaled; food rewards remain part of returns.
For a controlled comparison, measure food consumed, survival, and time until
capture against the same frozen hunters at matching initial seeds.

Tests:

```bash
.conda/bin/python -m pytest add_energy -q
```

Run each module's tests separately: these independent experiment
copies use the same flat module names. Dependencies are listed in
`requirements.txt`; the shared repository `.conda` environment can run both.
