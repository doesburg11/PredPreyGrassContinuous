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

Run IDs: the letters A to Z ran out, so every run also has an ID of module
prefix and number (AR = add_reproduction, AW = add_walls, GH =
add_group_hunting), numbered in the order the runs started. Runs before the
switch keep their letters; new runs get only an ID.

| ID | Run | Folder (`runs/add_walls_…`) | Started from | Iterations | What changed | Status |
|---|---|---|---|---:|---|---|
| AW-01 | N | `2026-10-08_11-27-42_148248` | scratch | 1,000 | walls, speed actions, linear movement cost | completed |
| AW-02 | O | `2026-10-08_16-02-39_634931` | N final | 2,000 | continuation of N | completed |
| AW-03 | P | `2026-10-08_20-13-41_273170` | scratch | 1,000 | as N, without walls (control) | completed |
| AW-04 | Q | `2026-10-08_22-52-21_564755` | scratch | 1,000 | as P, with run L's fixed energy cost (control) | completed |
| AW-05 | R | `2026-10-09_05-19-18_892579` | scratch | 1,000 | walls, speed actions, fixed energy cost (new defaults) | completed |
| AW-06 | S | `2026-10-09_09-28-39_319209` | R final | 2,000 | continuation of R | completed |
| AW-07 | T | `2026-10-09_14-00-28_001411` | Q final | 2,000 | continuation of Q | completed |
| AW-08 | Y | `2026-10-11_02-04-58_294484` | T final | 1,000 | gentle movement cost: resting 90% of T's cost, linear speed cost adds 10% at full speed | completed |
| AW-09 | Y0 | `2026-10-11_05-09-56_037528` | T final | 1,000 | control for Y: T's fixed cost unchanged | completed |
| AW-10 | Z25 | `2026-10-11_09-45-20_409365` | T final | 1,000 | 25% of the cost from movement | completed |
| AW-11 | Z50 | `2026-10-11_12-04-23_210330` | T final | 1,000 | 50% of the cost from movement | completed |
| AW-12 | | `2026-10-11_15-33-38_217711` | T final | 1,000 | as Z25 (25% of the cost from movement), with prey top speed 5.5 | completed |
| AW-13 | | (deleted) | T final | 478 of 1,000 | as Z50 (50% of the cost from movement), with prey top speed 5.5 | stopped, see section 20 |
| AW-14 | | `2026-10-11_21-31-48_098348` | T final | 1,000 | 75% of the cost from movement, prey top speed 5.5 | running |

Launchers: `runs/continue_walls.py` (O), `runs/no_walls_control.py` (P) and
`runs/fixed_cost_control.py` (Q); run R is plain `train.py` with the
defaults of commit `441cae8`; `runs/continue_walls_fixed_cost.py` (S),
`runs/continue_fixed_cost_control.py` (T), `runs/continue_T_gentle_cost.py` (Y),
`runs/continue_T_control.py` (Y0), `runs/continue_T_move25.py` (Z25) and
`runs/continue_T_move50.py` (Z50). A 75% run (Z75) was stopped after about
120 iterations, once Z25 and Z50 showed the trend, and deleted.
`runs/continue_T_move25_fastprey.py` (AW-12), `runs/continue_T_move50_fastprey.py`
(AW-13), `runs/continue_T_move75_fastprey.py` (AW-14).

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
slowed by turning pays slightly less). Run Q separates the speed actions
from the speed-dependent cost.

## 12. Speed actions with a fixed energy cost (run Q)

Same settings as run P (speed actions, no walls), but with run L's fixed
energy cost: `energy_*_resting_metabolic_cost` 0.15 / 8 and 0.05 / 8, and
speed and acceleration costs 0. Standing still saves no energy. 1,000
iterations from scratch. Per episode, in blocks of 250 iterations:

| Iterations | Predator births | Prey births | Episode length (decisions) | Run P length | Run L length |
|---|---:|---:|---:|---:|---:|
| 1–250 | 19.4 | 146 | 412 | 136 | 132 |
| 251–500 | 12.8 | 101 | 279 | 115 | 86 |
| 501–750 | 13.3 | 109 | 302 | 109 | 105 |
| 751–1000 | 14.6 | 123 | 340 | 110 | 139 |

**Final checks after 1,000 iterations** (20 episodes each):

| Run | Walls | Speed actions | Energy cost | Prey extinct | Length (physics steps) | Catch risk | Predators starved |
|---|---|---|---|---:|---:|---:|---:|
| L | no | no | fixed | 20 | 1,178 | 7.78 | 5.4 |
| N | yes | yes | speed-dependent | 20 | 718 | 9.57 | 2.6 |
| P | no | yes | speed-dependent | 20 | 455 | 10.71 | 0.8 |
| **Q** | no | yes | **fixed** | **18** | **1,924** | **4.84** | **10.8** |

Run Q's other two episodes: one reached the time limit with both species
alive, in one the predators died out. Per episode, 112 prey were caught and
111 born, and 2.9 prey starved.

**Tournament** (checkpoints every 100 iterations, 10 episodes per matchup;
`tournament.csv`): prey died out in 881 of 1,000 episodes, both species
survived in 97 (66 of them against the iteration-100 predators), predators
died out in 22. Newer prey are harder to catch in **80%** of pairs, newer
predators catch more in **71%**; in the second half (iterations 600–990) 62%
and 36%. Coexistence rises with the prey checkpoint (0% with prey 100, 22%
with prey 990, averaged over predators).

| Checkpoint | 100 | 200 | 300 | 400 | 500 | 600 | 700 | 800 | 900 | 990 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Predator: catch risk averaged over all prey | 3.7 | 5.0 | 5.9 | 6.7 | 6.1 | 6.5 | 7.5 | 6.9 | 6.3 | 6.6 |
| Prey: catch risk averaged over all predators | 8.1 | 7.3 | 6.5 | 6.4 | 6.0 | 5.9 | 5.6 | 5.3 | 5.2 | 4.8 |

**Conclusion:** the speed-dependent energy cost, not the speed actions, is
what turned the balance against the prey. With a fixed cost, the same speed
actions give the best prey results of any run at this stage: catch risk
4.8, against 7.8 in run L and 10.7 in run P. The likely mechanism is
predator starvation. With the speed-dependent cost, a predator that stops
or walks pays only half its resting rate, so predators can wait cheaply and
rarely starve (0.8 per episode in run P); with the fixed cost, waiting saves
nothing and 10.8 predators starve per episode, which keeps their numbers
and the pressure on the prey down. Prey, which mainly need to keep moving
to escape, gain little from cheap slow movement.

## 13. Walls with the fixed energy cost (run R)

After run Q, the `add_walls` defaults were changed to the fixed cost (speed
costs 0, resting costs 0.15 / 8 and 0.05 / 8). Run R trains that world, walls
included, from scratch for 1,000 iterations. Per episode, in blocks of 250
iterations:

| Iterations | Predator births | Prey births | Episode length (decisions) | Run Q length | Run N length |
|---|---:|---:|---:|---:|---:|
| 1–250 | 8.3 | 59 | 177 | 412 | 142 |
| 251–500 | 7.5 | 52 | 146 | 279 | 103 |
| 501–750 | 7.2 | 50 | 137 | 302 | 98 |
| 751–1000 | 7.6 | 56 | 158 | 340 | 107 |

**Final checks after 1,000 iterations**, all four `add_walls` worlds and run
L (20 episodes each):

| Run | Walls | Speed actions | Energy cost | Prey extinct | Length (physics steps) | Catch risk | Predators starved |
|---|---|---|---|---:|---:|---:|---:|
| L | no | no | fixed | 20 | 1,178 | 7.78 | 5.4 |
| N | yes | yes | speed-dependent | 20 | 718 | 9.57 | 2.6 |
| P | no | yes | speed-dependent | 20 | 455 | 10.71 | 0.8 |
| Q | no | yes | fixed | 18 | 1,924 | 4.84 | 10.8 |
| **R** | yes | yes | fixed | **20** | **942** | **7.82** | **4.8** |

**Tournament** (checkpoints every 100 iterations, 10 episodes per matchup;
`tournament.csv`): prey died out in 960 of 1,000 episodes, predators in 24,
both survived in 16. Newer prey are harder to catch in **80%** of pairs,
newer predators catch more in **68%**; in the second half (iterations
600–990) 70% and 52%.

| Checkpoint | 100 | 200 | 300 | 400 | 500 | 600 | 700 | 800 | 900 | 990 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Predator: catch risk averaged over all prey | 4.7 | 7.6 | 8.8 | 8.7 | 9.5 | 8.4 | 9.2 | 9.3 | 8.8 | 9.0 |
| Prey: catch risk averaged over all predators | 11.9 | 10.1 | 7.9 | 8.7 | 8.5 | 7.8 | 7.9 | 7.8 | 6.3 | 7.1 |

**Conclusion:** with the fixed cost, the walls make the prey's life harder,
not easier: catch risk 7.8 with walls (R) against 4.8 without (Q), episodes
half as long. With the speed-dependent cost the walls had helped the prey
slightly (N against P). Run R lands at run L's level at the same stage
(catch risk 7.8 in both, all prey extinct). Run L's prey went on to survive
in half the episodes after 3,000 more iterations (run M), so run R is the
first walls world that is not clearly behind that path.

## 14. Walls with the fixed energy cost, continued (run S)

Run S continued run R for 2,000 iterations (about 4.5 hours), so 3,000 in
total. Per episode, in blocks of 250 iterations, numbered as one training:

| Iterations | Run | Predator births | Prey births | Episode length (decisions) | Run O | Run M |
|---|---|---:|---:|---:|---:|---:|
| 751–1000 | R | 7.6 | 56 | 158 | 107 | — |
| 1001–1250 | S | 8.3 | 64 | 183 | 105 | 227 (1001–1500) |
| 1251–1500 | S | 8.7 | 69 | 201 | 104 | |
| 1501–1750 | S | 9.5 | 78 | 235 | 106 | 410 (1501–2000) |
| 1751–2000 | S | 10.4 | 90 | 275 | 112 | |
| 2001–2250 | S | 10.8 | 97 | 302 | 127 | 503 (2001–2500) |
| 2251–2500 | S | 11.6 | 109 | 346 | 138 | |
| 2501–2750 | S | 10.4 | 95 | 293 | 134 | 590 (2501–3000) |
| 2751–3000 | S | 11.3 | 105 | 330 | 146 | |

Episodes doubled in length over iterations 1,000–2,500, then levelled off
at about 300–350 decisions; run M was still growing at that stage.

**Final check, run S** (`final_check.csv`, iteration 2,990): **1 of 20
coexisting**, prey extinct in 19. Averages per episode: length 1,363
physics steps (run R: 942, run O: 752), catch risk 5.62 (run R: 7.82, run
O: 8.13), 75 prey caught and 68 born, 6.6 predators starved.

**Tournament over runs R and S** (checkpoints every 300 iterations, 10
episodes per matchup; `tournament_combined.csv` in run S): prey died out in
953 of 1,000 episodes, predators in 26, both survived in 21. Newer prey are
harder to catch in **83%** of pairs, newer predators catch more in **60%**;
in the second half (iterations 1,800–2,990) 66% and 56%.

| Checkpoint | 300 | 600 | 900 | 1200 | 1500 | 1800 | 2100 | 2400 | 2700 | 2990 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Predator: catch risk averaged over all prey | 6.0 | 6.3 | 6.8 | 7.1 | 7.3 | 6.9 | 6.6 | 7.1 | 7.5 | 6.9 |
| Prey: catch risk averaged over all predators | 9.5 | 8.5 | 7.7 | 7.3 | 7.3 | 6.4 | 5.4 | 5.4 | 5.9 | 5.2 |

**Conclusion:** with the fixed cost, prey among walls do improve: catch
risk falls from 9.5 to about 5.2 over 3,000 iterations (run O: 10.0 to
8.0), and the first walls episodes with both species alive appear. They
improve more slowly than run M's prey without walls (catch risk about 3
after 4,000 iterations, 10 of 20 coexisting), and predators gain little
after iteration 1,500, so there is no sustained arms race either.

## 15. Speed actions with the fixed cost, no walls, continued (run T)

Run T continued run Q for 2,000 iterations (about 5 hours), so 3,000 in
total: the same world as runs R and S, without walls. Per episode, in blocks
of 250 iterations, numbered as one training:

| Iterations | Run | Predator births | Prey births | Episode length (decisions) | Run S (walls) |
|---|---|---:|---:|---:|---:|
| 751–1000 | Q | 14.6 | 123 | 340 | 158 |
| 1001–1250 | T | 19.2 | 175 | 480 | 183 |
| 1251–1500 | T | 20.9 | 193 | 529 | 201 |
| 1501–1750 | T | 22.1 | 209 | 571 | 235 |
| 1751–2000 | T | 23.6 | 223 | 606 | 275 |
| 2001–2250 | T | 26.5 | 260 | 706 | 302 |
| 2251–2500 | T | 26.9 | 268 | 729 | 346 |
| 2501–2750 | T | 29.0 | 295 | 802 | 293 |
| 2751–3000 | T | 31.8 | 342 | 921 | 330 |

Run M, for comparison: 227, 410, 503 and 590 decisions over iterations
1001–1500, 1501–2000, 2001–2500 and 2501–3000.

**Final checks after 3,000 iterations** (20 episodes each):

| Run | Walls | Coexisting | Prey extinct | Predators extinct | Length (physics steps) | Catch risk | Predators starved |
|---|---|---:|---:|---:|---:|---:|---:|
| O | yes (speed-dependent cost) | 0 | 20 | 0 | 752 | 8.13 | — |
| S | yes | 1 | 19 | 0 | 1,363 | 5.62 | 6.6 |
| **T** | no | **11** | 7 | 2 | **5,885** | **3.02** | 32.0 |
| M (4,000 iterations) | no (16 actions) | 10 | 10 | 0 | 5,724 | 2.93 | 6.3 |

Run T per episode: 313 prey caught, 341 born, 22.6 starved; 6.5 predators
and 17.0 prey alive on average.

**Tournament over runs Q and T** (checkpoints every 300 iterations, 10
episodes per matchup; `tournament_combined.csv` in run T): both species
survived in 185 of 1,000 episodes, prey died out in 757, predators in 58.
Newer prey are harder to catch in **89%** of pairs, newer predators catch
more in **55%**; in the second half (iterations 1,800–2,990) 76% and 34%.
Coexistence rises with the prey checkpoint (0% with prey 300, 39% with prey
2,990) and hardly depends on the predator checkpoint after iteration 600
(10–25%).

| Checkpoint | 300 | 600 | 900 | 1200 | 1500 | 1800 | 2100 | 2400 | 2700 | 2990 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Predator: catch risk averaged over all prey | 4.2 | 4.7 | 4.7 | 4.7 | 4.6 | 5.1 | 4.8 | 4.9 | 4.6 | 4.3 |
| Prey: catch risk averaged over all predators | 6.7 | 5.9 | 5.8 | 4.9 | 4.4 | 4.8 | 3.9 | 3.6 | 3.4 | 3.0 |

**Conclusion:** without walls, the speed actions with the fixed cost give
the healthiest ecosystem so far: 11 of 20 coexisting after 3,000
iterations, where run M needed 4,000 for 10 of 20, and episodes still
growing at the end of training. The walls are what holds run S back: with
them, the same world reaches 1 of 20. The pattern of runs E to M is back,
though: prey keep improving and predators stop after a few hundred
iterations (55% of pairs overall, 34% late), so there is still no arms race.

## 16. A gentle movement cost (runs Y and Y0)

Run P showed that a speed-dependent cost which halves the cost of standing
still turns the balance against the prey. Runs Y and Y0 test a gentle
version on the working ecosystem of run T. Both continue run T's final
checkpoint for 1,000 iterations, without walls, and differ only in the
energy cost:

| Cost per physics step | Run T, Y0 | Run Y | Run P |
|---|---:|---:|---:|
| Stopped | 100% | 90% | 50% |
| Half speed | 100% | 95% | 75% |
| Full speed | 100% | 100% | 100% |

(Percentages of the fixed cost, 0.15/8 for predators and 0.05/8 for prey.
Run Y: resting cost 90%, plus a cost linear in speed adding the other 10% at
full speed.) Because both start from the same policies, run Y0 separates
the effect of the cost from the effect of training longer.

Per episode, in blocks of 250 iterations:

| Iterations | Run Y: predator / prey births, length | Run Y0: predator / prey births, length |
|---|---|---|
| 1–250 | 32.2 / 339, 919 | 32.8 / 348, 937 |
| 251–500 | 33.9 / 367, 992 | 34.7 / 379, 1,016 |
| 501–750 | 34.5 / 376, 1,012 | 35.2 / 391, 1,046 |
| 751–1000 | 35.2 / 397, 1,064 | 34.6 / 396, 1,055 |

**Final checks at iteration 990** (20 episodes each):

| | Run T (iteration 2,990) | Run Y (gentle cost) | Run Y0 (control) |
|---|---:|---:|---:|
| Coexisting | 11 | **14** | **17** |
| Prey extinct | 7 | 4 | 2 |
| Predators extinct | 2 | 2 | 1 |
| Length (physics steps) | 5,885 | 6,797 | 7,235 |
| Catch risk | 3.02 | 2.76 | 2.57 |
| Predators / prey alive on average | 6.5 / 17.0 | 6.6 / 18.4 | 6.3 / 20.4 |

**Tournaments** (each run's checkpoints every 100 iterations, 10 episodes
per matchup; `tournament.csv`): both species survived in 568 of 1,000
episodes in run Y and 578 in run Y0. Newer prey are harder to catch in 60%
(Y) and 53% (Y0) of pairs, newer predators catch more in 38% and 35%: over
these 1,000 iterations neither species clearly improved, and the predators
tended to get slightly worse.

**Conclusion:** the gentle movement cost does not change the ecosystem. Run
Y and its control are within the spread of 20-episode samples on every
measure (coexistence 14 against 17 of 20, catch risk 2.76 against 2.57,
tournament coexistence 57% against 58%). Making standing still 10% cheaper
does not give predators the advantage that run P's 50% did. Both runs also
show that run T's ecosystem kept improving with 1,000 more iterations of
training: coexistence rose from 11 to 14–17 of 20, the healthiest so far.

## 17. How much movement cost does the ecosystem tolerate? (runs Y0, Y, Z25, Z50)

A dose-response series on the protocol of section 16: every run continues
run T's final checkpoint for 1,000 iterations without walls, and only the
share of the energy cost that depends on movement differs. A resting cost of
(100 − s)% of run T's fixed cost is paid every physics step, and a cost
linear in speed adds the other s% at full speed, so full speed always costs
run T's rate and standing still saves s%.

Per episode, averaged over the last 250 training iterations:

| Run | Movement share s | Predator births | Prey births | Episode length (decisions) |
|---|---:|---:|---:|---:|
| Y0 | 0% | 34.6 | 396 | 1,055 |
| Y | 10% | 35.2 | 397 | 1,064 |
| Z25 | 25% | 25.9 | 257 | 715 |
| Z50 | 50% | 17.0 | 160 | 467 |

**Final checks with 50 episodes** (`final_check_50.csv` in each run,
iteration 990, seeds 0–49; Fisher's exact test of coexistence against Y0):

| Run | s | Coexisting | Prey extinct | Predators extinct | Length (physics steps) | Catch risk | Predators starved | Prey starved | Predators / prey alive |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Y0 | 0% | **40** | 7 | 3 | 7,248 | 2.50 | 37.1 | 40.1 | 6.2 / 20.6 |
| Y | 10% | 33 (p = 0.18) | 12 | 5 | 6,664 | 2.67 | 36.2 | 34.2 | 6.6 / 18.9 |
| Z25 | 25% | 11 (p < 0.001) | 37 | 2 | 4,856 | 4.50 | 24.4 | 6.7 | 7.6 / 12.3 |
| Z50 | 50% | 1 (p < 0.001) | 49 | 0 | 2,245 | 5.28 | 9.2 | 2.3 | 8.7 / 10.2 |

The 20-episode checks of section 16 (14 and 17 of 20 for Y and Y0) agree
with these within sampling spread.

**Tournaments** (checkpoints every 100 iterations, 10 episodes per matchup;
`tournament.csv`): both species survived in 568 of 1,000 episodes in run Y,
578 in Y0, 231 in Z25 and 106 in Z50. Newer prey are harder to catch in
54% (Z25) and 44% (Z50) of pairs, newer predators catch more in 65% and
74%; in the second half (iterations 600–990) 68% and 64% (Z25), 50% and
62% (Z50).

**Conclusion:** the more of the energy cost depends on movement, the worse
the prey fare, gradually rather than at a sharp threshold. The mechanism is
visible in starvation: with s = 0% about 37 predators starve per episode, at
25% about 24 and at 50% about 9, because waiting gets cheaper. More
predators survive, prey are caught more often, and coexistence falls from
80% (Y0) to 66% (10%), 22% (25%) and 2% (50%). The 10% step is not
statistically distinguishable from the control with 50 episodes (p = 0.18),
but the series suggests it carries a small cost too. Run P's collapse at 50%
from scratch is reproduced from a healthy, trained starting point.

## 18. Why movement costs favour the predators (runs Y0, Y, Z25)

Section 17 showed that predators starve less as the movement share grows.
The obvious explanation, predators learning to wait more, was tested by
recording every agent's chosen speed in 10 episodes of each run's final
checkpoint (`analysis/speed_choices.py`; output `analysis/speed_choices.json`
in each run). "Near grass" means within 64 units of an available patch.

| | Y0 (0%) | Y (10%) | Z25 (25%) |
|---|---|---|---|
| Predators: full / half / stop | 64% / 20% / 16% | 63% / 20% / 18% | 64% / 17% / 18% |
| Predators: mean speed (maximum 5) | 3.33 | 3.29 | 3.22 |
| Predators: stopping near grass / elsewhere | 25% / 12% | 26% / 13% | 28% / 12% |
| Predators: time near grass | 36% | 35% | 41% |
| Prey: full / half / stop | 92% / 5% / 3% | 94% / 4% / 2% | 91% / 5% / 3% |
| Prey: mean speed | 4.38 | 4.45 | 4.31 |

The predators do not wait more as the movement share grows: their speed
choices are nearly the same in all three runs. They already stopped twice as
often near grass as elsewhere in run Y0, without any movement cost, a
sit-and-wait habit inherited from run T. What separates the species is speed:
predators average about 3.3, prey about 4.4, because prey escape by running.

With a cost of (1 − s) + s × speed / 5 times the fixed rate, the same
behaviour therefore costs predators less than prey:

| Share of the fixed cost paid at the observed mean speeds | Predators | Prey |
|---|---:|---:|
| s = 10% | 96.6% | 98.9% |
| s = 25% | 91.1% | 96.6% |

**Conclusion:** a movement-based cost favours whichever species moves least.
Here that is the predators, whose existing sit-and-wait style near grass
becomes cheaper without any change in behaviour; prey, which have to run to
escape, save about a third as much. That lowers predator starvation, raises
predator numbers and turns the prey from food-limited to predation-limited
(section 17). Predators spent somewhat more time near grass at 25% (41%
against 35%), a possible small behavioural contribution.

## 19. Faster prey under a movement cost (run AW-12)

Section 18 found that a movement cost favours the slower species, here the
predators, whose sit-and-wait style near grass becomes cheaper. Run AW-12
tests whether faster prey restore the balance: run Z25's settings (25% of
the energy cost from movement), with the prey's top speed raised from 5 to
5.5 while predators keep 5. A prey can then outrun a predator, so waiting
near grass and chasing alone should catch fewer prey, and predators would
have to work harder, and pay for it under the movement cost. Like the other
runs of the series it continues run T's final checkpoint for 1,000
iterations without walls; it is compared with run Z25 (11 of 50 coexisting)
and run Y0 (40 of 50).

**Energy calibration.** The speed cost per unit of speed is left unchanged,
so a prey running at its new top speed pays slightly more than the fixed
rate: 0.75 + 0.25 × 5.5 / 5 = 102.5% at 25% movement share. Recalibrating so
that 5.5 costs exactly the fixed rate was rejected, for three reasons:

1. **Biology.** In Taylor, Heglund and Maloiy (1982), the extra power of
   running rises with speed at the same rate for animals of the same body
   mass. Predators and prey have the same body size here, so a faster prey
   should pay more for running faster, not get its extra speed for free.
2. **One change at a time.** Recalibrating would also lower the prey's cost
   per unit of speed: a hidden energy subsidy on top of the higher top
   speed. Run AW-12 would then differ from run Z25 in two ways, and a
   difference could come from either.
3. **The extra cost only applies when used.** Prey average about 4.4 (section
   18), below the old top speed, where they pay exactly what they paid in
   run Z25. Only sprints above 5 cost more.

Per episode, in blocks of 250 iterations:

| Iterations | Predator births | Prey births | Episode length (decisions) | Run Z25 length |
|---|---:|---:|---:|---:|
| 1–250 | 34.9 | 412 | 1,104 | 757 |
| 251–500 | 30.8 | 398 | 1,072 | 690 |
| 501–750 | 26.9 | 374 | 1,014 | 711 |
| 751–1000 | 21.3 | 341 | 930 | 715 |

**Final check with 50 episodes** (`final_check_50.csv`, iteration 990, seeds
0–49):

| | Run Y0 (0%, prey 5) | Run Z25 (25%, prey 5) | Run AW-12 (25%, prey 5.5) |
|---|---:|---:|---:|
| Coexisting | 40 | 11 | 15 |
| Prey extinct | 7 | 37 | **0** |
| Predators extinct | 3 | 2 | **35** |
| Length (physics steps) | 7,248 | 4,856 | 5,532 |
| Catch risk | 2.50 | 4.50 | **0.82** |
| Predators / prey alive on average | 6.2 / 20.6 | 7.6 / 12.3 | 4.7 / 38.6 |
| Predators starved per episode | 37.1 | 24.4 | 23.5 |
| Prey starved per episode | 40.1 | 6.7 | **96.4** |

**Tournament** (checkpoints every 100 iterations, 10 episodes per matchup;
`tournament.csv`): both species survived in 567 of 1,000 episodes,
predators died out in 405, prey in 28. Newer predators catch more in only
**8%** of pairs: they got worse almost throughout. Their catch risk
averaged over all prey fell from 2.75 (iteration 100) to 0.76 (iteration
990), and the share of episodes in which they died out rose from 9% to 69%
with the predator checkpoint. Newer prey are harder to catch in 52% of
pairs, so the prey did not need to improve further.

**Speed choices** (`analysis/speed_choices.json`, 10 episodes):

| | Run Z25 predators | Run AW-12 predators | Run Z25 prey | Run AW-12 prey |
|---|---:|---:|---:|---:|
| Full / half / stop | 64% / 17% / 18% | 37% / 30% / 33% | 91% / 5% / 3% | 89% / 7% / 3% |
| Mean speed | 3.22 | **2.27** | 4.31 | 4.67 |
| Time near grass | 41% | 48% | 24% | 11% |
| Stopping near grass | 28% | 36% | 1% | 1% |

**Conclusion:** a 10% speed advantage outweighs the 25% movement cost by far
and reverses the balance: prey become hard to catch (catch risk 0.82), and
the predators die out in 35 of 50 episodes. Instead of chasing harder, the
predators learned to chase less: they ran at full speed 37% of the time
instead of 64%, stopped twice as often, and waited near grass more. With
fast prey this cheap waiting fails, catches became rarer, and the predator
policy kept getting worse through training, as in add_group_hunting run W
when predators became too weak. The prey, now limited by food rather than
predation, avoid grass more (11% of their time near it, against 24%). The
balanced setting lies between: for example a smaller speed advantage, or
the same advantage with a larger movement share.


## 20. Towards a realistic movement share (runs AW-13 and AW-14)

**How large is the movement share in real animals?** Taylor, Heglund and
Maloiy (1982) fitted the energy use of running birds and mammals as
`0.533 × M^-0.316 × v + 0.300 × M^-0.303` (ml oxygen per second per kg,
body mass `M` in kg, speed `v` in m/s): a running cost rising linearly with
speed plus a cost at zero speed. For a 50 kg animal, the movement part is
about 60% of the total at 1 m/s, 83% at 3 m/s and 89% at 5 m/s. The real
share is higher still: the zero-speed term is about 1.7 times the
resting metabolism of a lying animal, sprints also cost anaerobic energy,
and smaller animals pay more per kg. Running at speed costs roughly 5 to 10
times as much as resting, a movement share of about 0.8 to 0.9 at full
speed in this model's terms (README, "How large is the movement share in
real animals?"). Averaged over a day, wild animals spend only 2 to 3 times
their resting rate, because they rest and walk much of the time; that is a
daily average, not the cost of moving.

The shares tested so far (10% to 50%) are therefore well below the
biological value, and with equal top speeds even 50% wiped out the prey
(section 17). Since prey in nature survive such costs through advantages
such as speed, the series moves towards realistic shares while keeping
run AW-12's speed advantage (prey 5.5, predators 5):

- **AW-13** (50% movement share) was stopped at iteration 478 to go directly
  to 75%. Up to then it looked balanced: predator births rose from 35.9 to
  38.2 per episode, prey births from 381 to 418, and episodes lasted about
  1,050 to 1,170 decisions. Its partial run was deleted.
- **AW-14** (75% movement share): the resting cost is 25% of run T's fixed
  rate, a speed cost adds 75% at full speed (5), and a prey sprinting at 5.5
  pays 0.25 + 0.75 × 5.5 / 5 = 107.5% (unchanged calibration, section 19).
  Running.

## Findings so far

1. **Walls, speed actions and movement costs together make prey much easier
   to catch:** 0 of 20 coexisting after 3,000 iterations, against 10 of 20
   without them (run M), with catch risk 8.1 against 2.9.
2. **The walls are not the cause** (run P): without walls the prey do
   slightly worse.
3. **The speed-dependent energy cost is the cause** (run Q): with the speed
   actions and a fixed cost, prey are caught about 40% less often than in
   run L and 55% less often than in run P at the same stage, and predators
   starve again. Cheap waiting mainly helps the predators.
4. **With the fixed cost, walls favour the predators** (run R against Q):
   catch risk 7.8 against 4.8, the same as run L without walls or speed
   actions. Continued to 3,000 iterations (run S), prey reach catch risk 5.2
   and 1 of 20 coexisting, more slowly than run M's prey without walls.
5. **Without walls, speed actions with the fixed cost give the healthiest
   ecosystem so far** (run T): 11 of 20 coexisting after 3,000 iterations.
   Predators again stop improving early, so there is no arms race.
6. **The ecosystem tolerates only a small movement cost** (runs Y0, Y,
   Z25, Z50): with 0%, 10%, 25% and 50% of the cost depending on movement,
   coexistence was 40, 33, 11 and 1 of 50, because cheaper waiting lets
   more predators survive. 10% is not significantly different from 0%
   (p = 0.18). Continuing run T to 4,000 iterations raised coexistence to
   80% (40 of 50).
7. **Movement costs favour the slower species** (section 18): predators
   already move slower than prey (mean speed 3.3 against 4.4, stopping
   near grass), so the same behaviour costs them less; they do not learn
   to wait more.
8. **Faster prey outweigh a movement cost by far** (run AW-12): prey at top
   speed 5.5 against 5, at 25% movement share, are hardly catchable
   (catch risk 0.82) and the predators die out in 35 of 50 episodes. The
   predators respond by chasing less, not more, and degrade through
   training.
9. **With walls and the speed-dependent cost, both species keep improving**
   (runs N and O: 81% and 67% of tournament pairs; run P, the same without
   walls: 79% and 69%), the highest predator share of the add_walls runs,
   though predators improve much more slowly after the first few hundred
   iterations.
