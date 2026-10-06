# Add reproduction

A contained copy of `add_energy` that adds energy-threshold reproduction, so
populations grow and shrink within an episode. This step brings the module
close to the [PredPreyGrass](https://github.com/doesburg11/PredPreyGrass)
discrete `base_environment`. It uses the same initial populations, energy
units, reproduction rule and sparse rewards, with the per-step rates rescaled
to continuous space.

Edit `config/config_env.py` and `config/config_ppo.py`, then run from the
repository root:

```bash
.conda/bin/python add_reproduction/train.py
```

Training defaults to 1,000 iterations. Outputs are saved in
`add_reproduction/runs/add_reproduction_<Amsterdam timestamp>/`. Training must
use `mode = "ps"` (one policy per species). `il` mode is rejected because a
policy per individual needs every agent ID in advance, and newborns get IDs
that no policy was built for.

Each run saves what it was started with in
`add_reproduction/runs/<run>/source_code/`, before training starts: the
module's code, `config/` and README (not the tests), `settings.json` with the
settings actually used (including changes made by a launcher script), a
launcher script started from outside the module, `git.txt` with the commit and
any uncommitted files plus `uncommitted.patch` with their changes, and
`versions.txt` with the Python version and `pip freeze`. To reproduce a run,
check out its commit, apply the patch if there is one, and train with its
`settings.json`, or run the copied `train.py` from `source_code/` itself.

## Reproduction

After movement, catches, grass, energy decay and starvation, every living
animal at or above its species' threshold has one offspring. A step can
produce at most one offspring per parent.

- The offspring starts with the species' initial energy, which the parent
  pays.
- The parent earns the species' reproduction reward.
- The offspring appears two body radii from the parent, in a random
  direction, and acts from the next step.
- No births happen on the step that ends an episode.
- Births stop while a species is at its population cap, and once its ID pool
  runs out.

Agent IDs are never reused within an episode, because RLlib maps each agent ID
to one trajectory per episode. Every ID in the pool (`predator_0` to
`predator_<pool - 1>`, and the same for prey) is declared up front, with its
species' observation and action spaces. A newborn takes the next unused ID.

| Setting | Default | Meaning |
|---|---:|---|
| `reproduction_predator_threshold` | 12.0 | Predator energy needed to reproduce |
| `reproduction_prey_threshold` | 8.0 | Prey energy needed to reproduce |
| `reproduction_predator_reward` | 10.0 | Reward to the predator parent per birth |
| `reproduction_prey_reward` | 10.0 | Reward to the prey parent per birth |
| `reproduction_max_predators` | 50 | Population cap for predators |
| `reproduction_max_prey` | 100 | Population cap for prey |
| `reproduction_predator_pool` | 2000 | Predator IDs available per episode |
| `reproduction_prey_pool` | 2000 | Prey IDs available per episode |

Each threshold must be above its species' initial energy and at or below its
energy cap. The caps keep Aquarium's observations affordable, because their
cost grows with the square of the number of animals.

Newborn IDs are counted per environment, so several environments in one
process don't hand out the same ID.

With `action_repeat > 1`, a sub-step with a birth ends the decision early.
RLlib's `MultiAgentEpisode` ends an episode as soon as every agent it already
knows is done, before it registers new agents. If births and the death of
every known animal fell in the same decision, the episode would end with the
newborns still alive. Ending the decision at the birth means RLlib sees each
newborn while its parent is alive. Per-agent infos are merged across
sub-steps, and `grass_eaten` and `births` are summed.

## Rewards

The defaults use PredPreyGrass's sparse scheme: +10 per birth and nothing
else. Your reward-shaping experiments found that this outperforms denser
rewards. Aquarium's own rewards and the grass food reward are set to 0 in
`config_env.py`:

| Setting | Default | Earlier modules |
|---|---:|---:|
| `prey_reward` (per step alive) | 0 | 1 |
| `prey_punishment` (when caught) | 0 | 1000 |
| `predator_reward` (per catch) | 0 | 10 |
| `grass_food_reward` (per patch) | 0 | 10 |
| `reward_scale` | 1.0 | 0.01 |

Restoring the earlier values gives the denser rewards of `add_energy`, for
comparison.

## Time scale

One PredPreyGrass grid step moves an animal one cell. Here, one cell
corresponds to about 32 units, and an animal covering that at 4 to 5 units per
step takes about 8 physics steps. The defaults use that factor throughout:

- `action_repeat = 8`: each decision is applied for 8 physics steps, so one
  decision equals one PredPreyGrass step. PPO's `gamma` (0.99 per decision)
  then discounts per grid step, as in PredPreyGrass.
- `max_time_steps = 8000` physics steps: an episode lasts about 1000
  decisions, matching PredPreyGrass's 1000-step episodes.
- The energy rates and the grass regrowth delay are rescaled by 8 (see
  Energy settings).

A sub-step with a birth ends its decision early, so some decisions are
shorter. With `--render video`, frames are captured once per decision, so use
a low `--fps` (for example 8) for real-time playback.

## Balance between predators and prey

As in PredPreyGrass, prey are as fast as predators and see farther. Both
species have a top speed of 5 units per step (`prey_max_velocity`,
`predator_max_velocity`). Prey see 267 units and predators see 200
(`prey_view_distance`, `predator_view_distance`), which keeps PredPreyGrass's
4:3 ratio between a 9x9 prey view and a 7x7 predator view. With Aquarium's
default prey speed of 4 and equal views, trained predators drove the prey
extinct in about 36 decisions. With these settings, episodes last about 90
decisions at the same training stage.

## Populations and grass

Episodes start with 6 predators and 8 prey, as in PredPreyGrass. One 25x25
grid cell corresponds to about 32 units of this 800-unit arena, so
PredPreyGrass's 100 grass cells become `grass_count = 100` patches.

## Energy settings

| Setting | Default | Meaning |
|---|---:|---|
| `energy_predator_initial` | 5.0 | Predator energy at reset and at birth |
| `energy_prey_initial` | 3.0 | Prey energy at reset and at birth |
| `energy_predator_max` | 24.0 | Predator energy cap |
| `energy_prey_max` | 16.0 | Prey energy cap |
| `energy_predator_decay` | 0.15 / 8 | Predator energy lost per physics step |
| `energy_prey_decay` | 0.05 / 8 | Prey energy lost per physics step |
| `energy_grass_gain` | 2.0 | Prey energy per grass patch eaten |
| `energy_catch_efficiency_predator` | 0.5 | Share of a caught prey's energy its catcher gains |
| `predator_max_age` | 10^9 | Aquarium's own starvation clock, kept off |

These use PredPreyGrass's energy units, with its per-step rates divided by 8,
because crossing this arena takes about 8 times as many steps as crossing
its grid. PredPreyGrass has no energy cap. Here the caps are set at twice the
reproduction thresholds, so they rarely bind and mainly scale each agent's
own-energy input (energy / max). Without food, a predator starves after 267
steps and a prey after 480 steps.

Each step, catching predators gain catch_efficiency_predator times the prey's energy,
prey gain energy per grass patch eaten, and every animal loses its decay.
Animals at or below zero energy starve: they are removed and terminated, with
`infos[agent]["starved"] = True`. Reproduction follows. The episode ends once
no predators or no prey are left. If the prey are gone, every agent is
terminated. If the predators are gone, the surviving prey are truncated.

## Grass

| Setting | Default | Meaning |
|---|---:|---|
| `grass_count` | 100 | Number of stationary patches; 0 disables grass |
| `grass_consume_radius` | 12 | Toroidal distance within which prey eat a patch |
| `grass_food_reward` | 0 | Reward per patch, before global reward scaling |
| `grass_respawn_delay` | 400 | Physics steps before a consumed patch regrows |

Each available patch is eaten by the closest living prey within its radius,
and regrows in place.

## Observations and display

Agents observe the egocentric layout from `dying_prey`. Prey also observe
the nearest visible grass patch (4 values), and every agent observes its own
energy as a fraction of its cap (1 value). Prey inputs are 33 values and
predator inputs are 29 values, before observation stacking. Checkpoints from
earlier modules cannot be used here, because their rewards and populations
differ.

The viewer draws grass under the animals and an energy bar above each animal.

Evaluation settings live in `config/config_eval.py`: what to run
(`source`: a training `checkpoint`, or `random` actions as a baseline), the
number of episodes and their seed, sampled or argmax actions, the display
(`render`: `window`, `video` or `none`) and its speed, and an optional
population CSV. Edit it, then run without command-line arguments:

```bash
.conda/bin/python add_reproduction/eval.py
```

With `source = "random"`, the population counts and episode length come from
`config_env.py` unless `predator_count`, `prey_count` or `max_time_steps` is
set in `config_eval.py`; a checkpoint always keeps the settings it was trained
with. Evaluation prints returns, prey caught, grass eaten, starvations,
births and peak and final populations per species (as predator/prey).
`population_csv` writes the population and cumulative births after every
decision.

Tests:

```bash
.conda/bin/python -m pytest add_reproduction -q
```

Run each module's tests separately: these independent experiment copies use
the same flat module names.
