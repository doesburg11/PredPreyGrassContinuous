# add_group_hunting: experiment log

A record of the training runs in this module, what each one changed, what it
showed, and what was concluded. It continues the
[add_walls log](../add_walls/RESULTS.md) (runs N to T), whose run letters it
carries on. All runs live in `runs/` (not in git). The numbers come from the
files saved there: the training metrics in each run's `tensorboard/`, the
evaluation results in `final_check.csv` and `tournament*.csv`, and the
outputs of the analysis scripts in `analysis/` (scripts in this module's
`analysis/` folder).

Terms (decision, coexistence, catch risk, final check, births per episode)
are as defined in the [add_reproduction log](../add_reproduction/RESULTS.md).

## Run index

| Run | Folder (`runs/add_group_hunting_…`) | Started from | Iterations | What changed | Status |
|---|---|---|---:|---|---|
| U | `2026-10-09_19-40-02_971713` | scratch | 1,000 | run Q's world, every animal's energy visible (baseline) | completed |
| V | `2026-10-09_20-24-36_412858` | scratch | 1,000 | as U, plus group hunting | completed |
| W | `2026-10-09_23-38-48_765124` | scratch | 3,000 | as V, with run U's catch efficiency 0.5 | running |

Both runs: no walls, 48 speed actions, fixed energy cost, moving grass,
predators see grass, `energy_observe_others = True` (37 inputs). Run U uses
catch efficiency 0.5 and no group hunting, so it differs from `add_walls`
run Q only in the visible energy. Run V uses the module's defaults: group
hunting with γ = 2, w = 2, radius 64, cooldown 16, and catch efficiency 1.0.

## 1. Seeing other animals' energy (run U)

Per episode, in blocks of 250 iterations:

| Iterations | Predator births | Prey births | Episode length (decisions) | Run Q length |
|---|---:|---:|---:|---:|
| 1–250 | 21.6 | 162 | 452 | 412 |
| 251–500 | 11.8 | 86 | 237 | 279 |
| 501–750 | 10.6 | 78 | 214 | 302 |
| 751–1000 | 10.9 | 81 | 222 | 340 |

Run Q's episodes grew again after iteration 500; run U's stayed at about 220
decisions.

**Final check** (`final_check.csv`, iteration 990, 20 episodes):

| | Run Q (energy not visible) | Run U (energy visible) |
|---|---:|---:|
| Prey extinct | 18 (1 coexisting, 1 predators extinct) | 20 |
| Length (physics steps) | 1,924 | 1,225 |
| Catch risk | 4.84 | 7.51 |
| Predators starved per episode | 10.8 | 7.6 |

**Tournament** (checkpoints every 100 iterations, 10 episodes per matchup;
`tournament.csv`): prey died out in 909 of 1,000 episodes, both survived in
84, predators died out in 7. Newer prey are harder to catch in **76%** of
pairs, newer predators catch more in **71%**; in the second half
(iterations 600–990) 54% and 42%.

| Checkpoint | 100 | 200 | 300 | 400 | 500 | 600 | 700 | 800 | 900 | 990 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Predator: catch risk averaged over all prey | 4.4 | 6.3 | 7.3 | 7.5 | 7.7 | 8.4 | 7.6 | 7.8 | 8.0 | 7.2 |
| Prey: catch risk averaged over all predators | 9.9 | 7.8 | 7.6 | 7.2 | 6.7 | 6.5 | 7.0 | 6.3 | 6.4 | 6.8 |

Runs Q and U are one training run each, so part of the difference between
them may be chance. The two analyses below test whether run U's predators
use the energy inputs at all.

## 2. What do predators use the energy inputs for? (run U)

**Do predators choose prey by energy?** (`analysis/prey_selection.py`,
outputs `analysis/selection_*.json` in run U.) At every catch, the caught
prey's energy was compared with the energies of the other living prey in
the catcher's view cone at that moment; final checkpoints, 20 episodes each
(seeds 1000–1019):

| | Run Q (energy not visible) | Run U (energy visible) |
|---|---:|---:|
| Catches | 2,551 | 1,605 |
| …with other prey in view | 1,935 (76%) | 1,172 (73%) |
| Mean energy of caught prey | 4.45 | 4.54 |
| Visible alternatives with more energy than the caught prey (0.5: no preference) | 0.49 | 0.47 |
| Caught the weakest prey in view (chance: 33%) | 30% | 29% |
| Caught prey's energy minus the alternatives' mean | −0.03 ± 0.05 | +0.02 ± 0.07 |

Without group hunting every touch is a catch, and the catcher gains half the
prey's energy, so a strong prey is worth more at no extra difficulty. Still,
predators show no preference, neither for strong nor for weak prey. The gain
is small (prey energies are mostly 3–8, worth 1.5–4 to the catcher), the
nearest prey is usually the cheapest to catch, and the only reward, a birth,
comes many catches later.

**Which energy inputs do predators use?** (`analysis/energy_ablation.py`,
outputs `analysis/ablation_*.json` in run U.) Run U's final checkpoint,
played with some energy inputs set to 0, 20 episodes each (seeds 0–19):

| Hidden | Catch risk | Physics steps per episode | Prey extinct |
|---|---:|---:|---:|
| Nothing | 7.01 | 1,553 | 20 |
| From predators: all other animals' energy | 5.55 (−21%) | 3,897 | 16 |
| From predators: only the other predator's energy | 5.91 (−16%) | 1,409 | 20 |
| From predators: only the prey's energy | 4.98 (−29%) | 3,488 | 17 |
| From prey: all other animals' energy | 6.34 (−10%) | 1,635 | 20 |

Predators rely on the prey's energy most: without it they catch 29% less and
episodes last twice as long. Since they do not use it to choose between
visible prey, they use it for something else, perhaps whether or how hard to
chase; this was not tested. The prey row may be chance (catch risk varies by
about ±0.5 between samples of 20 episodes). A hidden value reads as 0, which
to the network looks like a starving animal rather than "unknown", so part
of each effect may come from that.

**Conclusion:** visible energy makes predators about 55% more effective than
in run Q at the same stage, and they demonstrably use the prey's energy,
though not to pick which prey to catch. Group hunting (run V) must therefore
be compared with run U, not with run Q.

## 3. Group hunting (run V)

Per episode, in blocks of 250 iterations:

| Iterations | Predator births | Prey births | Episode length (decisions) | Run U length |
|---|---:|---:|---:|---:|
| 1–250 | 23.3 | 69 | 173 | 452 |
| 251–500 | 20.0 | 52 | 127 | 237 |
| 501–750 | 19.6 | 50 | 120 | 214 |
| 751–1000 | 18.2 | 44 | 104 | 222 |

**Final check** (`final_check.csv`, iteration 990, 20 episodes): prey
extinct in all 20, after 530 physics steps on average (run U: 1,225).
Catch risk 8.21 (run U: 7.51). Per episode: 15.3 predators and 11.1 prey
alive on average (run U: 8.4 and 8.6), 18.2 predator births, 3.8 predators
starved.

**Tournament** (checkpoints every 100 iterations, 10 episodes per matchup;
`tournament.csv`): prey died out in 998 of 1,000 episodes. Newer prey are
harder to catch in only **53%** of pairs, newer predators catch more in
**58%**; catch risk stays between 8.3 and 9.8 for every checkpoint of
either species. Neither species improved much after the first few hundred
iterations.

Group hunting did not weaken the predators; it strengthened them. A catch
now succeeds less often, but with catch efficiency 1.0 (run U: 0.5) each
catch yields twice the energy, so predators breed about twice as much
(18–23 births per episode against 11–22), reach about 15 alive on average
and wipe out the prey within about 530 physics steps.

**Do predators hunt together?** (`analysis/group_catches.py`, outputs
`analysis/group_catches_*.json` in run V.) For every attack: how many
predators were within 64 units of the prey, whether it succeeded, and how
many predators were alive. Run V's final checkpoint, and as a control run
U's final checkpoint (never trained with group hunting) played in run V's
world (group hunting on, catch efficiency 1.0); 20 episodes each, seeds
0–19.

| | Run V | Control (run U's policies) |
|---|---:|---:|
| Attacks | 2,070 | 4,132 |
| Success rate | 50% | 46% |
| Living predators at an attack (mean) | 12.6 | 12.3 |
| Attacks with at least 2 predators near the prey | **50%** | 38% |
| …expected if predators were spread at random | 21% | 20% |
| Catches made with at least 2 hunters | 70% | 54% |

At the same predator numbers, run V's predators attack with company more
often than the control's:

| Living predators | Run V: attacks with ≥ 2 near | Control | Random |
|---|---:|---:|---:|
| 1–7 | 29% | 22% | 9–10% |
| 8–13 | 55% | 36% | 17–18% |
| 14–19 | 57% | 48% | 26–27% |
| 20+ | 65% | 52% | 34–35% |

Success rates by number of hunters match the contest formula (run V: alone
30%, predicted 30%; two 63%, predicted 61%; three or more 79%, predicted
79%). Per attack, a pair member's expected share (0.63 / 2 = 0.32 of the
prey) is about what a lone predator expects (0.30), and three or more
hunters get less each (0.26), so grouping roughly breaks even per attack;
it also ends chases sooner.

Both groups of predators attack with company far more often than random
placement would give, because predators converge on the same prey and on
the same grass. Run V's predators do so about 1.3 times as often as the
control's at equal numbers. That is a sign that grouping was learned, but
not proof: the control's policies were trained in a different world (no
group hunting, catch efficiency 0.5), so the difference may also come from
other learned behaviour, such as chasing more persistently.

**Conclusion:** with these settings, group hunting turned into a predator
advantage through the doubled catch efficiency, and the ecosystem collapsed
faster than in run U; there is no arms race (53% and 58%). Predators attack
together more often than the control, so some grouping may have emerged,
but per attack it barely pays. Group hunting has to be tested with a catch
efficiency that keeps predators from booming.

## Findings so far

1. **Seeing other animals' energy helps the predators** (run U against
   run Q): catch risk 7.5 against 4.8 after 1,000 iterations. Hiding the
   prey's energy from run U's predators lowers their catch risk by 29%.
2. **Predators do not choose prey by energy**, neither weak nor strong,
   although without group hunting strong prey are worth more at no extra
   cost.
3. **Group hunting with catch efficiency 1.0 makes predators boom** (run V):
   prey extinct within about 530 physics steps, neither species improving
   much. Predators attack with company in 50% of attacks, against 38% for
   untrained-for-grouping predators at the same numbers; the success rates
   follow the contest formula.
