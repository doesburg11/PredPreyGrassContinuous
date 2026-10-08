# add_walls: experiment log

A record of the training runs in this module, what each one changed, what it
showed, and what was concluded. It continues the
[add_reproduction log](../add_reproduction/RESULTS.md), whose run letters
(A to M) it carries on. All runs live in `runs/` (not in git). The numbers
come from the files saved there: the training metrics in each run's
`tensorboard/`, and the evaluation results in `final_check.csv` and
`tournament*.csv`.

Terms (decision, coexistence, catch risk, final check, births per episode)
are as defined in the add_reproduction log.

## Run index

| Run | Folder (`runs/add_walls_…`) | Started from | Iterations | What changed | Status |
|---|---|---|---:|---|---|
| N | `2026-10-08_11-27-42_148248` | scratch | 1,000 | walls, speed actions, linear movement cost | completed |
| O | `2026-10-08_16-02-39_634931` | N final | 2,000 | continuation of N | completed |
| P | `2026-10-08_20-13-41_273170` | scratch | 1,000 | as N, without walls (control) | completed |

Launchers: `runs/continue_walls.py` (O) and `runs/no_walls_control.py` (P).

## Settings compared with run L

Run N uses run L's settings (moving grass, predators see grass, prey speed
5, radii 16 / 16, predator catch efficiency 0.5; see the add_reproduction
log) with three changes, read from each run's `source_code/settings.json`:

| Setting | Run L | Runs N, O | Run P |
|---|---|---|---|
| `walls_layout` | none | `"blocks"` (8 wall rays, occlusion) | none |
| Actions | 16 directions, always full speed | 48: 16 directions × full, half, stop | as N |
| Energy cost per physics step | fixed: 0.01875 predator, 0.00625 prey | resting half of that + cost linear in speed; full speed = the old fixed cost | as N |

At full speed the energy cost equals run L's, so an animal that never slows
down pays the same as before. Stopping halves it. Acceleration costs nothing
(see the README's "Direction and target speed" section).

## 10. Walls, speed actions and movement costs (runs N and O)

Run N trained from scratch for 1,000 iterations (about 2 hours) and run O
continued it for 2,000 (about 4 hours). Per episode, in blocks of 250
iterations, numbered as one training:

| Iterations | Run | Predator births | Prey births | Episode length (decisions) |
|---|---|---:|---:|---:|
| 1–250 | N | 7.6 | 46 | 142 |
| 251–500 | N | 6.7 | 38 | 103 |
| 501–750 | N | 6.4 | 37 | 98 |
| 751–1000 | N | 6.6 | 39 | 107 |
| 1001–1250 | O | 6.4 | 39 | 105 |
| 1251–1500 | O | 6.4 | 39 | 104 |
| 1501–1750 | O | 6.4 | 39 | 106 |
| 1751–2000 | O | 6.6 | 41 | 112 |
| 2001–2250 | O | 7.0 | 45 | 127 |
| 2251–2500 | O | 7.2 | 48 | 138 |
| 2501–2750 | O | 6.9 | 47 | 134 |
| 2751–3000 | O | 7.2 | 50 | 146 |

Runs L and M at the same stage: 109–122 decisions over iterations 1–1000,
227 over 1001–1500 and 590 over 2501–3000. Run N was close to run L for the first
1,000 iterations; from there runs L and M took off and run O barely moved.

**Final checks** (`final_check.csv` in N and in O):

| | Run N (iteration 990) | Run O (iteration 2,990) | Run M (iteration 3,990), for comparison |
|---|---:|---:|---:|
| Coexisting | 0 of 20 | 0 of 20 | 10 of 20 |
| Prey extinct | 20 of 20 | 20 of 20 | 10 of 20 |
| Episode length (physics steps) | 718 | 752 | 5,724 |
| Catch risk | 9.57 | 8.13 | 2.93 |
| Prey caught per episode | 50 | 52 | — |
| Prey born per episode | 42 | 44 | — |
| Prey starved per episode | 0 | 0.1 | 14.7 |

Prey die only by being caught, and they are caught faster than they are born.

**Tournament over run N** (checkpoints every 100 iterations, 10 episodes per
matchup; `tournament.csv` in run N): prey died out in 990 of 1,000 episodes.
Newer prey are harder to catch in **74%** of pairs, and newer predators catch
more in **74%**. Predators gained until about iteration 500 (catch risk
averaged over all prey: 7.2 at iteration 100, 10.2 at 500, 10.0 at 990);
prey improved throughout (catch risk averaged over all predators: 11.2 at
100, 8.5 at 990).

**Tournament over runs N and O** (checkpoints every 300 iterations, 10
episodes per matchup; `tournament_combined.csv` in run O): prey died out in
all 1,000 episodes.

| Checkpoint | 300 | 600 | 900 | 1200 | 1500 | 1800 | 2100 | 2400 | 2700 | 2990 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Predator: catch risk averaged over all prey | 7.9 | 9.1 | 8.5 | 8.8 | 8.9 | 9.0 | 8.9 | 8.9 | 9.0 | 9.5 |
| Prey: catch risk averaged over all predators | 10.0 | 9.4 | 9.4 | 9.1 | 9.0 | 8.9 | 8.7 | 8.1 | 8.0 | 8.0 |

- Newer prey are harder to catch in **81%** of pairs, newer predators catch
  more in **67%**. Both above 50% for the first time (run L/M: 96% and 43%),
  but the predators' gain after iteration 600 is small.
- Diagonal catch risk (each checkpoint against its training partner) stays
  between 8 and 10.2 throughout.
- Even the earliest predators (iteration 300) wipe out the latest prey.

**Conclusion:** together, walls, speed actions and movement costs shift the
balance strongly towards predators. Prey improve steadily, by about 20% in
catch risk over 3,000 iterations, but stay far from surviving, while in run M
prey brought their catch risk from about 10 to about 3 over the same span.
These runs cannot tell which of the three changes is responsible; run P
removes the walls to find out.

## 11. Control without walls (run P)

Same settings as run N with `walls_layout = None` (so also no wall rays in
the observations), 1,000 iterations from scratch (about 1.7 hours). Per
episode, in blocks of 250 iterations:

| Iterations | Predator births | Prey births | Episode length (decisions) | Run N, same iterations |
|---|---:|---:|---:|---:|
| 1–250 | 8.3 | 47 | 136 | 142 |
| 251–500 | 7.7 | 43 | 115 | 103 |
| 501–750 | 7.3 | 41 | 109 | 98 |
| 751–1000 | 7.1 | 41 | 110 | 107 |

**Final check** (`final_check.csv`, iteration 990): **0 of 20 coexisting**,
prey extinct in all 20. Averages per episode: length 455 physics steps
(run N: 718), catch risk 10.71 (run N: 9.57), 40 prey caught and 32 born,
no prey starving.

**Tournament** (checkpoints every 100 iterations, 10 episodes per matchup;
`tournament.csv`): prey died out in all 1,000 episodes. Newer prey are
harder to catch in **79%** of pairs, newer predators catch more in **69%**.

| Checkpoint | 100 | 200 | 300 | 400 | 500 | 600 | 700 | 800 | 900 | 990 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Predator: catch risk averaged over all prey | 9.1 | 9.8 | 10.5 | 10.1 | 10.8 | 11.0 | 10.4 | 11.0 | 10.7 | 10.8 |
| Prey: catch risk averaged over all predators | 12.9 | 11.4 | 10.6 | 10.7 | 10.4 | 9.8 | 9.8 | 9.6 | 9.4 | 9.7 |

Same stage, three worlds (final checks after 1,000 iterations, all with
prey extinct in 20 of 20 episodes):

| Run | Walls | Speed actions and speed-dependent cost | Episode length (physics steps) | Catch risk |
|---|---|---|---:|---:|
| L (add_reproduction) | no | no | 1,178 | 7.78 |
| N | yes | yes | 718 | 9.57 |
| P | no | yes | 455 | 10.71 |

**Conclusion:** the walls are not what favours the predators. Without them
the prey do slightly worse (catch risk 10.7 against 9.6, episodes about a
third shorter): blocks give prey some cover. What separates runs N and P
from run L is the action space (speed actions) and the energy cost that
depends on speed; with them, prey are caught about 25–40% more often at the
same stage. After 1,000 iterations run L's prey did not survive either; the
difference that matters is that run L's prey then caught up (run M: 10 of 20
coexisting at 4,000 iterations), while run O's prey barely gained in 2,000
more iterations. Run P was not continued, so it does not show whether its
prey would catch up.

With 16 full-speed actions, the linear cost equals run L's fixed cost, so
run L is in effect the control for "no speed actions" (an animal
slowed by turning pays slightly less). Separating the speed
actions from the speed-dependent cost needs one more run: speed actions with
run L's fixed per-step cost.

## Findings so far

1. **Walls, speed actions and movement costs together make prey much easier
   to catch:** 0 of 20 coexisting after 3,000 iterations, against 10 of 20
   without them (run M), with catch risk 8.1 against 2.9.
2. **The walls are not the cause** (run P): without walls the prey do
   slightly worse. The speed actions or the speed-dependent energy cost
   make prey 25–40% easier to catch than in run L at the same stage.
3. **Both species keep improving in this world** (81% and 67% of tournament
   pairs; run P: 79% and 69%), the closest to co-adaptation so far, though
   predators improve much more slowly after the first few hundred
   iterations.
