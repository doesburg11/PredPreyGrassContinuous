# PredPreyGrassContinuous

A continuous-space counterpart to the grid-based
[PredPreyGrass](https://github.com/doesburg11/PredPreyGrass): multi-agent
reinforcement learning for predators and prey, built on the physics-based
[Aquarium](https://github.com/michaelkoelle/marl-aquarium) environment and
trained with RLlib's new API stack (RLModule + Learner).

## Layout

- [`dying_prey/`](dying_prey/) — Aquarium PPO training with permanent prey deaths.
  See its [README](dying_prey/README.md) for installation and evaluation.
- [`add_grass/`](add_grass/) — a contained experiment adding stationary grass,
  food rewards, delayed regrowth, and nearest-visible-grass inputs for prey.
  See its [README](add_grass/README.md).
- [`add_energy/`](add_energy/) — `add_grass` plus metabolic energy: decay,
  food from grass and catches, starvation for both species, and an own-energy
  input. See its [README](add_energy/README.md).

All subprojects share one Conda environment at the repo root (`.conda/`).
Training settings live in each module's `config/config_env.py` and
`config/config_ppo.py`; run training without command-line arguments.
