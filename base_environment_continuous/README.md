# base_environment_continuous

A standalone continuous-space Predator-Prey-Grass environment for RLlib,
replacing the [Aquarium](https://github.com/michaelkoelle/marl-aquarium)
package (unmaintained since 2024) and the patches that
[`add_walls`](../add_walls/) applies to it. It contains only the world of
the milestone runs (AW-17 to AW-22): no walls, direction and speed actions,
energy with a movement cost, grass clusters that move as they are grazed,
energy-threshold reproduction and egocentric observations.

The first version reproduces `add_walls` in that configuration **exactly**,
including Aquarium's quirks, so all results obtained with `add_walls`
remain valid here. [SPEC.md](SPEC.md) records every rule, and in section 9
the quirks that are candidates for a later, deliberate cleanup.

## Files

- [`environment.py`](environment.py): `PredPreyGrassContinuous`, an RLlib
  `MultiAgentEnv` (animals, physics, energy, reproduction, observations,
  decisions).
- [`grass.py`](grass.py): grass patches, seed dispersal and overgrazing.
- [`geometry.py`](geometry.py): 2D vectors and torus distances, offsets and
  view cones.
- [`config/config_env.py`](config/config_env.py) and
  [`config/config_ppo.py`](config/config_ppo.py): settings for training.
- [`train.py`](train.py): PPO training, one policy per species (as in
  `add_walls`).
- [`tournament.py`](tournament.py): every predator checkpoint of a run
  against every prey checkpoint (as in `add_walls`).
- [`analysis/final_check.py`](analysis/final_check.py): final check of a
  trained checkpoint.
- [`tests/`](tests/): the equivalence test against `add_walls` and unit
  tests.

## Usage

From the repository root, with the shared Conda environment:

```
.conda/bin/python base_environment_continuous/train.py
cd base_environment_continuous && ../.conda/bin/python tournament.py runs/<run>
.conda/bin/python -m pytest base_environment_continuous/tests
```

Training takes no command-line arguments: edit the config files. Runs are
saved in `base_environment_continuous/runs/`.

## How the equivalence was checked

- `tests/test_equivalence.py` records trajectories of `add_walls`'s
  environment with random actions (in a separate process, since both
  modules use the same file names), replays the same actions here and
  compares every decision bit for bit: observations, rewards, flags, infos
  and the full state, including the random-number generators. 61 episodes,
  about 12,500 decisions, covering births, catches, starvation, grass
  collapse, extinction, the time limit, one physics step per decision,
  population caps and exhausted ID pools.
- Run AW-21's trained policies played the same 50 seeded episodes in both
  environments with identical results (55,298 decisions, 21,873 births,
  19,860 catches).

Two deliberate differences, neither affecting the dynamics: with
`reset(seed=None)` each environment has its own spawn stream instead of
Python's process-global one, and the keys of the returned dictionaries come
in a different order.

## Validation runs

BE-01 and BE-02 trained the milestone world from scratch with this
environment (3,000 iterations, 14 simulation workers each, side by side),
as AW-21 and AW-22 did with `add_walls`. They reached the same state: 72
of 100 evaluation episodes with both species surviving against 71 of 100
for AW-21 and AW-22, the same catch risk, populations and behaviour
(predators waiting near grass), and the same arms race. Details in
[RESULTS.md](RESULTS.md).

Training took 8–15% less wall time (about 9 hours instead of 10 for 3,000
iterations): the environment plays episodes about twice as fast as
`add_walls`, but the learner takes most of the time.

`analysis/final_check.py` evaluates a checkpoint (outcomes, catch risk,
populations, speed choices and time and catches near grass); it also
accepts checkpoints of `add_walls` runs in the milestone world.
