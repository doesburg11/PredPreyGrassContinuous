# PredPreyGrassContinuous

A continuous-space counterpart to the grid-based
[PredPreyGrass](https://github.com/doesburg11/PredPreyGrass): multi-agent
reinforcement learning for predators and prey, built on the physics-based
[Aquarium](https://github.com/michaelkoelle/marl-aquarium) environment and
trained with RLlib's new API stack (RLModule + Learner).

There is no grass layer yet.

## Layout

- [`dying_agents/`](dying_agents/) — Aquarium under RLlib PPO (independent
  learning or parameter sharing, for both species), with `--no-respawn` so
  caught prey die for good instead of respawning. See its
  [README](dying_agents/README.md) for install, training and evaluation.

All subprojects share one Conda environment at the repo root (`.conda/`).
