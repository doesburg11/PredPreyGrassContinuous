# Add walls

A contained copy of `add_reproduction` that adds **walls**: obstacles that
block movement and sight, so prey can hide from predators and predators can
ambush prey, as in PredPreyGrass's `walls_occlusion` experiments (see
[Walls](#walls) below). Walls are set with `walls_layout` in
`config/config_env.py` (default `"blocks"`); with `walls_layout = None` the
module behaves like `add_reproduction`.

This module also computes observations once per decision instead of after
every physics step to reduce environment-stepping overhead. Physics, food,
energy, deaths and births still update every step; the policy reads the
observation at the end of the decision. The speedup depends on the configuration
(see `SafeParallelPettingZooEnv.step` in `env_wrapper.py`).

Everything else (reproduction, energy, moving grass, rewards, populations)
works as in `add_reproduction`, as described below.

## Additions made in add_walls

This module started as a contained copy of `add_reproduction`. Its additions
are listed below in order; inherited features are described in the later
reference sections.

| Addition | Behavior |
|---|---|
| Wall layouts | Built-in `blocks` and `chambers`, plus custom `(x, y, width, height)` rectangles; walls wrap across arena edges |
| Wall collisions | Animals are pushed out of obstacles and lose velocity into a wall while retaining movement along it; newborns are also moved clear of walls |
| Sight occlusion | Walls hide animals and grass using a precomputed visibility table |
| Wall sensing | `walls_rays` distance inputs are appended to each agent's observations; eight rays increase the default observation size from 33 to 41 |
| Observation computation | Observations are computed once per repeated decision, including the updated state after wall collision resolution |
| Wall geometry validation | Nonfinite coordinates and sizes, undersized walls, and rectangles larger than the arena are rejected; origins are normalized so collisions, sight, sensing and rendering use the same geometry |
| Wall-aware grass placement | Cluster centres skip blocked cells; patch placement and dispersal use open-space fallbacks; collapsed clusters retain their seed bank when destination cells are blocked |
| Direction and target speed | New configurations enable 16 directions at full, half or stopped speed; acceleration and braking remain controlled by physics |
| Movement energy costs | Actual speed squared and change in velocity squared add to resting decay each physics step; speed is measured after wall resolution |
| Occluded view cones | The viewer clips each view cone at the walls, so the drawn cone shows what an animal can see; one predator and one prey are followed, switching when it dies |
| Regression coverage | Tests cover wall geometry, wrapping, visibility, open-space placement, target speeds, braking, movement costs, newborn action spaces and PPO's expanded action outputs |

Walls default to the `"blocks"` layout (`walls_layout = None` switches them off). Target-speed controls
and movement costs are enabled in the current configuration; they were also
added to `add_reproduction`. Older checkpoint configurations keep their original
16-action controls and fixed metabolic decay.

Aquarium's `observable_walls` parameter does not construct these obstacles:
its border-observation calls are commented out in the installed package. Our
walls are controlled by the `walls_*` settings.

Edit `config/config_env.py` and `config/config_ppo.py`, then run from the
repository root:

```bash
.conda/bin/python add_walls/train.py
```

Training defaults to 1,000 iterations. Outputs are saved in
`add_walls/runs/add_walls_<Amsterdam timestamp>/`. Training must
use `mode = "ps"` (one policy per species). `il` mode is rejected because a
policy per individual needs every agent ID in advance, and newborns get IDs
that no policy was built for.

Each run saves what it was started with in
`add_walls/runs/<run>/source_code/`, before training starts: the
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
PredPreyGrass's 100 grass cells become `grass_count = 100` patches. By
default the grass grows in 5 clusters that move as the prey graze them (see
Grass below).

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
own-energy input (energy / max). With resting decay alone and no food, a
predator starves after 267 steps and a prey after 480 steps. Movement costs
shorten those lifetimes when enabled.

Each step, catching predators gain catch_efficiency_predator times the prey's energy,
prey gain energy per grass patch eaten, and every animal loses its resting
decay plus the configured speed and acceleration costs (see
[Direction and target speed](#direction-and-target-speed)).
Animals at or below zero energy starve: they are removed and terminated, with
`infos[agent]["starved"] = True`. Reproduction follows. The episode ends once
no predators or no prey are left. If the prey are gone, every agent is
terminated. If the predators are gone, the surviving prey are truncated.

## Grass

Grass is the prey's only food. It consists of `grass_count` patches. A patch
is either available (visible, edible) or regrowing (hidden). The closest
living prey within `grass_consume_radius` of an available patch eats it and
gains `energy_grass_gain` energy. The patch is then hidden for
`grass_respawn_delay` physics steps before it regrows. Prey observe the
nearest *available* patch in their view cone (4 inputs). With
`grass_predators_observe`, predators observe it in their own view cone as
well, as in PredPreyGrass, where both species see grass. This lets them
learn, for example, to wait where prey come to eat.

Grass can be static, as in earlier modules, or it can move: settings choose
where patches start and where an eaten patch regrows. The default
configuration (`config/config_env.py`) uses **moving clusters**: clustered
placement, seed dispersal and overgrazing, all three switched on.

### 1. Starting layout: scattered or clustered

- `grass_clustered = False`: patches are scattered uniformly over the arena.
- `grass_clustered = True`: patches are grouped in `grass_cluster_count`
  clusters. The arena is divided into a grid with at least that many cells
  (3 x 2 for 5 clusters, 4 x 3 for 10 on a square arena). Each cluster
  centre is placed in its own randomly chosen cell, anywhere in the middle
  half of it, so clusters start spread out instead of bunching together.
  Patches are split evenly over the clusters (100 patches in 5 clusters gives
  20 each). They are scattered around their centre with normal offsets of
  standard deviation `grass_cluster_spread`, wrapped across the arena's
  edges.

Each episode draws a new layout.

### 2. Where an eaten patch regrows

Exactly one of these applies:

- **In place** (default when the switches below are off): the patch regrows
  in exactly the same spot.
- **Random respawn** (`grass_random_respawn = True`): the patch regrows at a
  new random spot, within its own cluster when clustered (so clusters
  persist), and anywhere in the arena otherwise.
- **Seed dispersal** (`grass_dispersal = True`, requires `grass_clustered`,
  excludes `grass_random_respawn`): when the patch regrows, it appears next to
  a randomly chosen living patch of its own cluster, scattered by normal
  offsets of standard deviation `grass_dispersal_distance`. The new position
  is chosen at the moment of regrowth, so it follows wherever the cluster
  currently has grass. If the cluster has no living patch left, the patch
  regrows around the cluster centre instead, like seed from the soil's seed
  bank.

Seed dispersal makes clusters move gradually. Where prey graze heavily, a
cluster loses patches and nothing regrows there. Regrowth happens next to the
patches that survive, so a cluster creeps away from the side where it is
being eaten, without any imposed direction.

### 3. Overgrazing: clusters that are grazed bare move away

With `grass_overgrazing = True` (requires `grass_clustered`), a cluster that
is grazed down to `grass_overgrazing_threshold` or less of its patches
(0.0: grazed completely bare) collapses:

1. Its remaining patches, if any, die back too.
2. Its centre moves to a grid cell that no other cluster uses (if every cell
   is taken, to any cell other than its own).
3. After `grass_overgrazing_delay` physics steps, all its patches regrow
   together, scattered around the new centre.
4. It cannot collapse again until it has recovered above the threshold.

Overgrazing produces sudden relocations, where dispersal produces gradual
drift.

### Moving clusters together

With clustering, dispersal and overgrazing combined, the prey shape where
their food goes. Grazing pushes clusters away gradually through dispersal,
and stripping a cluster bare sends it to another part of the arena. A
cluster that is grazed lightly stays where it is. Prey that overgraze lose
their food for a while and have to find where it reappears. Predators can
use this: prey gather where grass is. Ecologically, the two mechanisms
correspond to vegetation spreading by seed dispersal and to the shifting
mosaic of grazed and recovering patches in rangelands.

In a test with policies trained on static grass, each cluster was grazed
bare and relocated roughly every 500 steps. Moving grass is a harder world.
Policies trained on it kept both species alive to the time limit in 0 of 20
evaluated episodes after 1,000 iterations and in 4 of 20 after 2,000, still
improving, against 18 of 20 for static grass.

### Settings

Values as set in `config/config_env.py`. If a switch is left out of the
configuration, it is off.

| Setting | Value | Meaning |
|---|---:|---|
| `grass_count` | 100 | Number of patches; 0 disables grass |
| `grass_consume_radius` | 12 | Distance within which a prey eats a patch |
| `grass_food_reward` | 0 | Reward per patch, before global reward scaling |
| `grass_respawn_delay` | 400 | Physics steps before an eaten patch regrows |
| `grass_clustered` | True | Clusters instead of uniform placement |
| `grass_cluster_count` | 5 | Number of clusters |
| `grass_cluster_spread` | 48.0 | Scatter of patches around a cluster centre (sd, units) |
| `grass_random_respawn` | False | Regrow at a random spot (in its cluster when clustered) |
| `grass_dispersal` | True | Regrow next to a living patch of the cluster |
| `grass_dispersal_distance` | 24.0 | Scatter around that patch (sd, units) |
| `grass_overgrazing` | True | Clusters grazed down to the threshold relocate |
| `grass_overgrazing_threshold` | 0.0 | Share of a cluster's patches at or below which it collapses |
| `grass_overgrazing_delay` | 400 | Physics steps before a collapsed cluster regrows elsewhere |
| `grass_predators_observe` | True | Predators also observe the nearest visible patch (4 inputs) |

For scale: one PredPreyGrass grid cell is about 32 units, so the spread of 48
is 1.5 cells and the dispersal distance of 24 is 0.75 cells. The regrowth
delay of 400 physics steps equals 50 grid steps, the time PredPreyGrass's
grass needs to regrow fully.

The grass layer uses its own random generator, seeded from the episode seed,
for layouts, relocations and regrowth positions. The same seed and the same
animal behavior therefore always give the same grass history, and grass
randomness does not change Aquarium's own random stream.

To test a policy trained on one kind of grass in another, use
`env_overrides` in `config/config_eval.py`, for example
`{"grass_dispersal": False, "grass_overgrazing": False}`. A checkpoint
otherwise always runs with the grass settings it was trained with.

## Walls

Walls are obstacles that block movement and sight, so prey can hide from
predators and predators can ambush prey. They are the continuous counterpart
of PredPreyGrass's `walls_occlusion` experiments. The default layout is
`"blocks"`; set `walls_layout = None` in `config/config_env.py` to switch
walls off, or choose another layout:

- `"blocks"`: a "forest" of 4 x 4 square blocks, each two wall thicknesses
  wide (64 units), evenly spread over the arena.
- `"chambers"`: walls along the arena's edges and a cross through the middle,
  making four rooms. Each room side has a doorway of a quarter of its length,
  in the middle. The arena wraps around, so the wall along the top edge is
  also the boundary at the bottom, and the left wall is also the boundary at
  the right.
- `"custom"`: your own rectangles in `walls_rectangles`, as
  `(x, y, width, height)` in arena units. All four values must be finite;
  sizes cannot exceed the arena dimensions. Origins wrap into the arena,
  and rectangles crossing an edge continue on the opposite side.

Grass cluster centres are placed in open space, skipping fully blocked grid
cells. If too few open cells remain for distinct clusters, placement raises
a clear error. Failed patch or dispersal sampling falls back to an open
seed-bank location or direct sampling of open area. A collapsed cluster
keeps its existing seed bank if all eligible destination cells are blocked.

Walls do three things:

1. **They block movement.** After every physics step, an animal that overlaps
   a wall is pushed back out, and its speed into the wall is removed, so it
   slides along the wall. Where walls overlap (corners, crossings), they act
   as one shape: an animal is always moved to a side that is clear of every
   wall. Every wall, custom ones included, must be at least 16 units wide and
   high. The environment also refuses settings where an animal could move
   past the middle of the thinnest wall in one step (its speed would have to
   reach its body radius plus half that wall), so nothing passes through.
   Grass never grows inside walls, and an animal born next to a wall is moved
   out of it at once.
2. **They block sight (occlusion).** An animal or grass patch behind a wall is
   not seen. Visibility is decided between cells of a grid over the arena
   (`walls_visibility_cell = 32` units, one PredPreyGrass cell, so 25 x 25
   cells), from a table of which cells can see which, computed once when the
   environment is created (2 to 3 seconds). During an episode, a visibility
   check is then a table lookup. Each cell is judged from a point inside it
   that is clear of walls, so a patch next to a wall is seen from its own side
   only. This is the continuous equivalent of PredPreyGrass's grid-based line
   of sight. Within a single cell everything counts as visible, so a wall
   thinner than a cell can leave a thin strip on its far side visible.
3. **They are sensed.** Every agent gets `walls_rays` extra inputs (8 by
   default): the distance to the nearest wall in each of the action
   directions (up, then clockwise), divided by its view distance, so 1.0
   means no wall within view. Without these inputs, agents would only notice
   walls by bumping into them.

The viewer draws walls in grey-brown. With `draw_view_cones` (on in
`config_eval.py`), it draws the view cone of one predator (blue) and one prey
(red), clipped at the exact wall outlines, so the shaded area is what that
animal can see. The selected animal is kept until it dies; then the next
living animal of its species is followed. The drawing is exact; the agents'
own visibility uses the cell table above, so the two can differ by up to a
cell near walls. Walls add roughly 15% to the time per
environment step in a typical episode (pushing animals out, the ray inputs
and the visibility lookups). Because walls add inputs, runs with walls must
be trained from scratch.

| Setting | Default | Meaning |
|---|---:|---|
| `walls_layout` | `"blocks"` | `None` (no walls), `"blocks"`, `"chambers"` or `"custom"` |
| `walls_thickness` | 32 | Wall thickness for the preset layouts (at least 16) |
| `walls_rectangles` | [] | Rectangles for `"custom"`: `(x, y, width, height)` |
| `walls_occlusion` | True | Walls block sight of animals and grass |
| `walls_rays` | 8 | Wall-distance inputs per agent (0: none) |
| `walls_visibility_cell` | 32 | Cell size of the line-of-sight table, in units |

## Observations and display

Agents observe the egocentric layout from `dying_prey` (28 values). Prey
also observe the nearest visible grass patch (4 values), and so do predators
with `grass_predators_observe` (on in `config_env.py`). Every agent observes
its own energy as a fraction of its cap (1 value). Both species therefore have
33 inputs, before observation stacking, plus `walls_rays` wall-distance
inputs (8) when walls are on. Runs trained before
`grass_predators_observe` existed have 29 predator inputs, and their
checkpoints keep that layout, because a missing setting means off.
Checkpoints from earlier modules cannot be used here, because their rewards
and populations differ.

The viewer draws grass under the animals and an energy bar above each animal.

Evaluation settings live in `config/config_eval.py`: what to run
(`source`: a training `checkpoint`, or `random` actions as a baseline), the
number of episodes and their seed, sampled or argmax actions, the display
(`render`: `window`, `video` or `none`) and its speed, and an optional
population CSV. Edit it, then run without command-line arguments:

```bash
.conda/bin/python add_walls/eval.py
```

To watch random agents in the current wall layout in a live window,
whatever `source` and `render` are set to:

```bash
.conda/bin/python add_walls/random_eval.py
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
.conda/bin/python -m pytest add_walls -q
```

Run each module's tests separately: these independent experiment copies use
the same flat module names.


## Direction and target speed

New training configurations enable `target_speed_actions = True`. Each
agent has 48 categorical actions: 16 arena directions at three target speeds.
Encode an action as `direction + 16 * speed_index`, with speed index 0 for
full speed, 1 for half speed, and 2 for stopped. For example, action 4 moves
right at full speed, 20 moves right at half speed, and 36 brakes to a stop.
Stopping ignores the direction. Acceleration and braking are limited by the
species' configured maximum acceleration; agents do not control acceleration
separately. Existing scripted direction policies continue to request full speed.
Random evaluation samples all 48 actions, and PPO learns a categorical policy
over all direction/speed combinations.

Energy loss is charged after movement on every physics step:

```text
resting decay + speed coefficient * actual speed squared
              + acceleration coefficient * actual change in velocity squared
```

The coefficients are `energy_predator_speed_cost`, `energy_prey_speed_cost`,
`energy_predator_acceleration_cost`, and `energy_prey_acceleration_cost`.
Braking, turns and collision-induced changes in velocity also count. In
`add_walls`, speed is measured after wall collision resolution, so a stationary
animal pushing against a wall pays no travelling cost. Repeated decisions
pay separately for each physics step. Defaults make steady full-speed travel
add half the existing resting decay, plus any acceleration cost; these are
initial tuning values, not experimentally calibrated rates.

Checkpoint evaluation uses the checkpoint's saved environment settings.
Older configurations omit the toggle and movement coefficients, retaining
16 direction-only actions and fixed metabolic decay. New 48-action training
requires new policies rather than loading the old 16-action network weights.
To run legacy movement explicitly, set `target_speed_actions = False` and
all four movement-energy coefficients to zero.
