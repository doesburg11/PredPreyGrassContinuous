# Add grass

A contained copy of `dying_prey` with stationary, renewable grass patches.
Existing models and runs were not copied. This module retains PPO training for
both species; fixed-hunter prey-only training is a separate future experiment.

Edit `config/config_env.py` and `config/config_ppo.py`, then run from the
repository root:

```bash
.conda/bin/python add_grass/train.py
```

Training defaults to 1,000 iterations. Outputs are saved in
`add_grass/runs/add_grass_<Amsterdam timestamp>/`. Permanent prey deaths are
enabled and prey view distance is 200. There is no energy or reproduction model.

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
than 28, before observation stacking. Predator inputs remain 28 values.
Setting `grass_count = 0` retains the same observation shapes for ablation runs.
Grass requires `obs_mode = "egocentric"`.

Old prey checkpoints with 28 inputs cannot train/evaluate the new grass input
layout directly. New checkpoints retain their grass settings for evaluation.
Available patches use the transparent `assets/grass-transparent.png` sprite
in the Aquarium window/video, derived from `assets/grass.png`.

```bash
.conda/bin/python add_grass/eval.py --random --draw-view-cones
.conda/bin/python add_grass/eval.py --checkpoint add_grass/runs/<run>/checkpoint \
    --stochastic --draw-view-cones --episodes 3
```

Evaluation prints consumed grass (`grass_eaten`) alongside prey catches and
returns. Rewards are reported unscaled; food rewards remain part of returns.
For a controlled comparison, measure food consumed, survival, and time until
capture against the same frozen hunters at matching initial seeds.

Tests:

```bash
.conda/bin/python -m pytest add_grass -q
```

Run `dying_prey` and `add_grass` tests separately: these independent experiment
copies use the same flat module names. Dependencies are listed in
`requirements.txt`; the shared repository `.conda` environment can run both.
