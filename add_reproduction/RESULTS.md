# add_reproduction: experiment log

A record of the training runs in this module, what each one changed, what it
showed, and what was concluded. All runs live in `runs/` (not in git). The
numbers come from the files saved there: the training metrics in each run's
`tensorboard/`, and the evaluation results in `final_check.csv`,
`fixed_prey*.csv`, `tournament*.csv` and `scripted_predators.csv`. Where a
number comes from an evaluation whose output was not saved, it is marked
*(not saved)*.

## Terms

- **Decision**: one action choice by every agent. With `action_repeat = 8`
  (from run A on), a decision lasts up to 8 physics steps, and an episode
  lasts at most 8,000 physics steps, about 1,000 decisions.
- **Coexistence**: an episode that reaches the 8,000-step limit with both
  species alive. Otherwise the prey died out, the predators died out, or
  both did.
- **Catch risk**: prey caught per prey alive per 1,000 physics steps. It
  measures how catchable prey are, independent of how many there are.
- **Final check**: 20 evaluation episodes of a run's final model, seeds 0 to
  19, actions sampled as in training (`tournament.py --iterations 990
  --episodes 20`).
- **Births per episode** in training tables are returns divided by 10 (the
  only reward is +10 per birth), except in run G, which also had a catch
  reward.

## Run index

| Run | Folder (`runs/add_reproduction_…`) | Started from | Iterations | What changed | Status |
|---|---|---|---:|---|---|
| A | `2026-10-04_23-48-12_573698` | scratch | 4,607 | baseline: Aquarium's speeds and sizes | stopped |
| B | `2026-10-05_12-25-58_654906` | scratch | 311 | prey as fast as predators, prey see farther | stopped |
| C | `2026-10-05_13-03-36_352152` | scratch | 409 | smaller bodies (catch distance 32) | stopped |
| D | `2026-10-05_14-05-03_102196` | scratch | 385 | predators gain half a prey's energy | stopped |
| E | `2026-10-05_18-18-53_020370` | scratch | 1,000 | same as D, full length | completed |
| F | `2026-10-05_23-47-59_932848` | E final | 300 | prey frozen, only predators train | completed |
| G | `2026-10-06_09-55-47_528702` | E final | 150 | as F, plus +3 reward per catch | completed |
| — | `2026-10-06_19-50-02_166744` | scratch | 0 | failed at start (RLlib env check, fixed) | failed |
| H | `2026-10-06_19-52-31_442849` | scratch | 1,000 | moving grass | completed |
| I | `2026-10-06_22-25-55_664977` | H final | 1,000 | continuation of H | completed |
| J | `2026-10-07_09-38-10_103867` | I final | 1,000 | continuation of I | completed |
| K | `2026-10-07_15-34-38_396259` | J final | 517 | prey faster than predators (5.5 vs 5) | stopped |
| L | `2026-10-07_18-42-19_138184` | scratch | 1,000 | moving grass, predators see grass | completed |
| M | `2026-10-07_21-26-01_880217` | L final | 3,000 (planned) | continuation of L | running |

Common to all runs: 6 predators and 8 prey at the start, 100 grass patches,
`max_time_steps = 8000`, `action_repeat = 8`, sparse rewards (+10 per birth,
everything else 0), `reward_scale = 1.0`, the PredPreyGrass energy settings
(see the README), PPO with one policy per species.

## Settings that changed between runs

| Setting | A | B | C | D, E, F, H, I, J | G | K | L |
|---|---:|---:|---:|---:|---:|---:|---:|
| `prey_max_velocity` | 4 (default) | 5 | 5 | 5 | 5 | **5.5** | 5 |
| `prey_view_distance` / predator | 200 / 200 | 267 / 200 | 267 / 200 | 267 / 200 | 267 / 200 | 267 / 200 | 267 / 200 |
| `predator_radius` / `prey_radius` | 30 / 20 (default) | 30 / 20 | 16 / 16 | 16 / 16 | 16 / 16 | 16 / 16 | 16 / 16 |
| catch efficiency (predator) | 1.0 | 1.0 | 1.0 | 0.5 | 0.5 | 0.5 | 0.5 |
| `predator_reward` (per catch) | 0 | 0 | 0 | 0 | **3** | 0 | 0 |
| grass | static, scattered | static | static | E, F: static; H, I, J: **moving** | static | moving | moving |
| `grass_predators_observe` | no | no | no | no | no | no | **yes** |

"Moving" grass means `grass_clustered` (5 clusters of 20 patches),
`grass_dispersal` and `grass_overgrazing` (threshold 0, delay 400), as
described in the README's Grass section.

## 1. Balancing predators and prey (runs A to E)

In the first runs the prey always died out. Each change below made the prey
harder to catch, and episode length at the same training stage is the
clearest summary:

| Run | Change | Episode length (decisions), iterations 201–300 | Episode length, last 100 iterations |
|---|---|---:|---:|
| A | baseline | 36 | 50 (iteration ~4,600) |
| B | prey speed 5, prey view 267 | 91 | 91 (iteration ~311) |
| C | radii 16 / 16 | 106 | 96 (iteration ~409) |
| D | catch efficiency 0.5 | 885 | 983 (iteration ~385) |
| E | as D, 1,000 iterations | — | 1,149 |

- **A:** predators are faster (5 vs 4) and catch from 50 units away (radii
  30 + 20). Trained predators wiped out the prey in about 50 decisions.
  Evaluation of the final model: prey died out in 20 of 20 episodes *(not
  saved)*.
- **B:** equal speed and longer prey vision, as in PredPreyGrass. Episodes
  lasted 2.5 times longer, but evaluation at iteration 250 still had the prey
  dying out in 20 of 20 episodes *(not saved)*.
- **C:** a catch distance of 32 units (one grid cell) instead of 50.
  Episodes started much longer (about 300 decisions at first), then
  shortened as predators learned, levelling off at about 100. Evaluation at
  iteration 330: prey died out in 20 of 20 *(not saved)*.
- **D:** predators gain only half a caught prey's energy, so they breed more
  slowly after a successful hunt. Episodes lasted about ten times longer.
  Evaluations *(not saved)*: 1 of 10 coexisting at iteration 130, 0 of 5 at
  iteration 300, 3 of 3 at iteration 380.
- **E** repeats D for the full 1,000 iterations.

**Final check, run E (static grass): 18 of 20 coexisting**, 2 of 20 with the
prey dying out. Averages per episode: length 7,756 physics steps, 22.9 prey
and 7.4 predators alive, catch risk 2.60, 476 prey births against 460 prey
caught, 40 predator births against 38 predator starvations, 7.3 prey
starvations.

## 2. Co-adaptation on static grass (run E tournament)

Every predator checkpoint of run E (100, 200, …, 900, 990) played every prey
checkpoint, 20 episodes per pair (`tournament.csv`):

- Newer prey are harder to catch for a fixed predator in **96%** of pairs.
- Newer predators catch more of a fixed prey in **55%** of pairs, close to no
  trend.
- Catch risk along the diagonal (each checkpoint against its training
  partner) falls from 5.09 to 2.60.
- Each predator catches about 8 prey per 1,000 steps, whatever its
  checkpoint: the rate at which its energy just breaks even.

**Conclusion:** the prey adapted to the predators, the predators hardly
improved after the first ~100 iterations. No evidence of co-adaptation.

## 3. Why the predators stalled (runs F and G)

**Run F: predators train against frozen prey** (run E's final prey; both
policies start from run E's final weights; `frozen_policies =
["prey_policy"]`). Evaluated against that same frozen prey, 30 episodes per
predator checkpoint (`fixed_prey.csv`):

| Predator checkpoint | Catch risk | Catches per predator per 1,000 steps | Coexisting |
|---:|---:|---:|---:|
| 10 | 2.95 | 8.01 | 29 of 30 |
| 150 | 2.94 | 8.00 | 27 of 30 |
| 290 | 2.39 | 8.02 | 26 of 30 |

No improvement, even with the prey standing still. (An earlier tournament on
this run, in `tournament.csv`, reused the same seeds for identical prey
columns, so it had only 10 distinct episodes per predator; the table above
replaces it.)

**Run G: as F, plus +3 reward per catch for the catching predator**
(`predator_reward = 3`, `catch_radius = 33`). Fixed-prey evaluation, 30
episodes each: catch risk 2.62, 2.72, 2.69, 2.71, 2.79 and 2.76 at
checkpoints 10 to 140, and 7.9 to 8.1 catches per predator in every row.
No improvement with direct catch credit either.

**Exploration:** policy entropy (maximum 2.77 for 16 actions) fell from 2.75
to 0.55 for predators and from 2.76 to 0.53 for prey over run E, and stayed
at about 0.5 in run F. Both species became equally decisive, yet the prey
kept improving, so low exploration alone does not explain the stall.

**Conclusion:** the predators did not stall because the prey kept changing
(F) or because the reward was too sparse (G).

## 4. Moving grass (runs H, I and J)

With grass in 5 clusters that move as the prey graze them (seed dispersal
and overgrazing), training from scratch (H) and continued twice (I, J),
3,000 iterations in total:

| Iterations | Run | Predator births | Prey births | Episode length (decisions) |
|---|---|---:|---:|---:|
| 1–200 | H | 8.2 | 59 | 178 |
| 201–400 | H | 8.2 | 50 | 138 |
| 401–600 | H | 10.1 | 67 | 190 |
| 601–800 | H | 11.9 | 84 | 239 |
| 801–1000 | H | 13.3 | 100 | 288 |
| 1001–1200 | I | 16.9 | 136 | 395 |
| 1201–1400 | I | 19.9 | 168 | 489 |
| 1401–1600 | I | 22.9 | 200 | 580 |
| 1601–1800 | I | 24.0 | 220 | 634 |
| 1801–2000 | I | 26.3 | 251 | 722 |
| 2001–2200 | J | 26.6 | 252 | 724 |
| 2201–2400 | J | 27.6 | 270 | 775 |
| 2401–2600 | J | 27.7 | 278 | 792 |
| 2601–2800 | J | 29.9 | 307 | 873 |
| 2801–3000 | J | 30.1 | 317 | 902 |

Final checks (20 episodes each, `final_check.csv`):

| | H (1,000 iterations) | I (2,000) | J (3,000) | E (static grass) |
|---|---:|---:|---:|---:|
| Coexisting | 0 of 20 | 4 of 20 | **8 of 20** | 18 of 20 |
| Prey / predators died out | 19 / 1 | 14 / 2 | 9 / 3 | 2 / 0 |
| Length (physics steps) | 1,802 | 4,686 | 5,757 | 7,756 |
| Prey / predators alive (average) | 10.3 / 7.4 | 13.7 / 6.7 | 17.1 / 6.2 | 22.9 / 7.4 |
| Catch risk | 5.35 | 3.61 | 2.86 | 2.60 |
| Prey starved per episode | 0.8 | 5.0 | 12.9 | 7.3 |

Moving grass is a much harder world: after 1,000 iterations no episode
coexisted. The prey kept improving through all 3,000 iterations, at a
slowing rate, and their catch risk approached the static-grass level. More
prey starve as they get better at escaping.

**Tournament over H, I and J** (checkpoints every 300 iterations, numbered
as one training; `tournament_HIJ.csv` in run J):

- Newer prey are harder to catch for a fixed predator in **98%** of pairs:
  catch risk falls from about 8–10 (prey at iteration 300) to about 3 (prey
  at 2,990) in every row.
- Newer predators catch more in **32%** of pairs: newer predators cause
  slightly *lower* catch risk against a fixed prey.
- The diagonal falls from 9.75 to 2.86. Coexistence depends mostly on the
  prey checkpoint (0% with prey 300, 36% with prey 2,990, averaged over
  predators) and only a little on the predator checkpoint (15% to 22%).

**Conclusion:** the same pattern as on static grass. Moving grass made the
prey's task harder, not the predators'.

## 5. Faster prey (run K)

Continued from run J with prey speed 5.5 (predators 5), to make plain
chasing fail. Stopped at iteration 517: training had levelled off since
about iteration 100 (episode length about 970–1,000 decisions, predator
births about 29).

Predator checkpoints against prey checkpoint 270, 30 episodes each
(`fixed_prey_270.csv`):

| Predator checkpoint | Catch risk | Coexisting | Prey died out | Predators died out |
|---:|---:|---:|---:|---:|
| 10 | 2.32 | 18 of 30 | 5 | 7 |
| 100 | 2.15 | 21 of 30 | 4 | 5 |
| 270 | 2.02 | 15 of 30 | 8 | 7 |

The ecosystem coexisted more often than in run J (50–70% against 40%), but
the predators did not find new ways to hunt, and they died out more often.

## 6. How much room do predators have? (scripted predators)

Hand-coded predators against run K's prey checkpoint 270 (prey speed 5.5),
20 episodes each (`scripted_predators.csv`, from `scripted_predators.py`):

| Predator | Catch risk | Catches per predator per 1,000 steps | Coexisting | Predators died out |
|---|---:|---:|---:|---:|
| `chase` (nearest visible prey) | 1.91 | 8.17 | 11 of 20 | 8 |
| `intercept` (lead pursuit, own view) | **2.27** | 8.29 | **16 of 20** | **2** |
| `chase_all` (sees all prey) | 1.66 | 8.34 | 9 of 20 | 8 |
| `intercept_all` (sees all prey) | 2.05 | 8.36 | 10 of 20 | 6 |
| learned, checkpoint 270 | 1.98 | 8.41 | 11 of 20 | 5 |

- The learned predators perform like simple chasing.
- Aiming at where a prey will be (lead pursuit) catches about 15% more and
  keeps predators alive much better: a skill the learned predators could
  learn (they observe prey velocities) but have not.
- Seeing all prey does not help: steering at distant prey wastes effort.
- Every strategy settles at about 8.2–8.4 catches per predator: better
  hunting shows up as fewer predator deaths, not more catches per predator.

**Conclusion:** in this setup simple pursuit is close to the ceiling, and
the learned predators have reached it. The prey's task (evade and find food)
takes thousands of iterations; the predators' task (steer at visible prey)
is learned within a few hundred, after which there is little left to gain.

## 7. Spatial behaviour (analyses)

These were run with analysis scripts outside the repository, on run E's and
run I's final models.

**Prey clustering** (Clark–Evans index R: 1 random, below 1 clustered;
10 episodes each):

| | Static grass (E) | Moving grass (I) |
|---|---:|---:|
| Prey R | 0.89 | 0.84 |
| Available grass R | 0.84 | 0.40 |
| Prey distance to nearest available grass | 116 | 130 |
| Prey distance to nearest predator | 149 | 156 |

Prey are mildly clustered in both worlds, slightly more with moving grass,
but they do not gather tightly at the strongly clustered grass.

**Static grass** (run E, 8 episodes): all patch spots, including regrowing
ones, are randomly placed (R = 1.00); prey are 12% closer to patch spots than
random points (34.9 against 39.9 units), which fits prey just having eaten
there.

**Do prey stay near regrowth spots?** (6 episodes per world): no. Per 400
physics steps a prey travels about 1,900 units and ends about 550 units from
where it started, in both worlds. Only 4% (static) and 1% (moving) of
regrown patches are eaten within the first decision after regrowing. The
prey's strategy is to keep roaming fast and head for visible grass. Their
policies have no memory and cannot see regrowing patches, so they cannot
learn specific spots.

## 8. Predators see grass (run L)

`grass_predators_observe` gives predators the same nearest-grass input as
prey, as in PredPreyGrass, where both species see grass. Otherwise run L
equals run H. At the same training stage, run L's episodes are about half as
long as run H's:

| Iterations | H: predator / prey births, length | L: predator / prey births, length |
|---|---|---|
| 1–100 | 9.3 / 74, 237 | 7.8 / 64, 210 |
| 101–200 | 7.1 / 44, 120 | 5.5 / 32, 81 |
| 201–300 | 7.7 / 46, 124 | 5.8 / 32, 79 |
| 301–400 | 8.7 / 55, 151 | 5.9 / 33, 83 |
| 401–500 | 9.5 / 62, 174 | 6.2 / 36, 92 |
| 501–566 | 10.6 / 71, 200 | 6.4 / 38, 100 |

The gap persisted to the end of training (about 100 against 200 decisions
at iterations 501–600).

**Final check, run L:** the prey died out in 20 of 20 episodes, after 1,178
physics steps on average (run H: 1,802), with a catch risk of 7.78 (run H:
5.35), 7.7 prey and 8.2 predators alive on average, and 5.4 predators
starving per episode (run H: 10.2).

Both runs also trained their prey, so the final checks alone cannot tell
whether L's predators improved or L's prey learned less. Prey observations
are the same in both runs, so both runs' final predators were played against
**the same prey** (run H's final prey), each in its own environment, 20
episodes each (analysis script outside the repository):

| Against run H's final prey | Run H's predators (blind to grass) | Run L's predators (see grass) |
|---|---:|---:|
| Prey died out | 19 of 20 | 20 of 20 |
| Length (physics steps) | 1,802 | 1,149 |
| Catch risk | 5.35 | **7.18** (+34%) |
| Predators starved per episode | 10.2 | 7.0 |

The run H row reproduces run H's final check exactly. With the prey held
fixed, predators that see grass catch about a third more effectively and
starve less: **the first clear predator improvement in these experiments.**
Unlike freezing the prey (F), a catch reward (G) or faster prey (K), grass
vision gave the predators information worth learning from, most likely where
prey come to eat. The ecosystem became less balanced as a result, as prey
need many more iterations to adapt.

## Findings so far

1. **A stable ecosystem needs the prey to be hard to catch early on.** Equal
   speed, longer prey vision, a catch distance of one cell and half energy
   per catch turned guaranteed prey extinction into 90% coexistence on
   static grass (run E).
2. **Prey learn for thousands of iterations; predators stop after a few
   hundred.** In every tournament, newer prey beat older ones (96–98% of
   pairs) while newer predators did not (32–55%). No co-adaptation so far.
3. **Predators sit at energy break-even.** About 8 catches per predator per
   1,000 steps in every run, checkpoint and strategy. Predator numbers
   adjust to prey availability; individual skill barely changes the rate.
4. **The predators' ceiling is set by what they can perceive.** Scripted
   pursuit does about as well as the learned predators; only lead pursuit
   is clearly better (~15%). Freezing the prey (F), adding a catch reward
   (G) and faster prey (K) did not make predators improve, but letting them
   see grass (L) did: 34% higher catch risk against the same prey.
5. **Moving grass is a harder, still learnable world for prey:** 0, 4 and 8
   of 20 coexisting after 1,000, 2,000 and 3,000 iterations.

## Open questions and next steps

- **Continue run L** (as H was continued by I and J): do the prey catch up
  with predators that see grass, and do both species now keep adapting to
  each other (tournament)?
- **Walls** (`add_walls`): hiding and ambush through blocked movement and
  sight.
- **Group hunting:** catches that need, or are easier with, several
  predators, so that coordination becomes a skill.
- **Memory** (`obs_stack` > 1): tracking prey out of view, remembering grass.
- **A "slow down" or "stay" action:** Aquarium's 16 actions always ask for
  full speed, which rules out waiting or ambushing in place.
