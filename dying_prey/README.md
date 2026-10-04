# PredPreyGrassContinuous

A continuous-space counterpart to the grid-based
[PredPreyGrass](https://github.com/doesburg11/PredPreyGrass), built on Aquarium.
There is no grass layer yet: so far it adds agents that die for good
(`config_env["keep_prey_count_constant"] = False`) on top of the Aquarium port described below.

Runs [Aquarium](https://github.com/michaelkoelle/marl-aquarium) (Kölle, Erpelding,
Ritz, Phan, Illium & Linnhoff-Popien, 2024) — a PettingZoo-native, physics-based
predator-prey MARL environment — under RLlib's **new API stack** (RLModule +
Learner). Built as a side exercise while researching prior art for
[PredPreyGrass](https://github.com/doesburg11/PredPreyGrass)'s
`predator_sexual_reproduction` module; see that repo's
`predpreygrass/non_evolutionary/predator_sexual_reproduction/` for the actual
research line this supports.

Aquarium has **no sexual/mate reproduction** — prey reproduce solo, age-gated,
capped at a max count (off by default via `procreate=False`); predators don't
reproduce at all (confirmed by reading `marl_aquarium/env/{aquarium,predator,prey}.py`
directly — only `prey.py` has a `replicate()` method). This port is an
engineering exercise (RLlib new-stack fluency, comparing Aquarium's
steering-force/toroidal-distance/FOV machinery to PredPreyGrass's grid model),
not an extension of the reproduction research itself.

## What's here

- `env_wrapper.py` — registers Aquarium's `parallel_env()` with Ray Tune's env
  registry via RLlib's built-in `ParallelPettingZooEnv` wrapper. No custom
  `MultiAgentEnv` subclass was needed: Aquarium is already a
  `ParallelEnv[str, Box, Discrete]`, which `ParallelPettingZooEnv` handles directly.
- `train.py` — PPO training on RLlib's new API stack, configured entirely in
  `config/config_env.py` and `config/config_ppo.py`. No training CLI arguments.
  Set `config_ppo["mode"]` to `"il"` for one policy per individual, or `"ps"`
  for one shared policy per species. Set `checkpoint_dir` to save a final
  checkpoint and `checkpoint_every` to save periodic checkpoints too.
- `eval.py` — rolls out either a checkpoint from training with `checkpoint_dir` configured
  (greedily: argmax over each policy's action logits, no exploration) or, with
  `--random`, a uniform-random baseline with no RLlib/checkpoint involved at
  all -- visualized either way with Aquarium's own `pygame` renderer:
  - `--render window` (default) opens a live pygame window.
  - `--render video` does the same and also saves an mp4 per episode (via
    `imageio`/ffmpeg) to `--out-dir`.
  - `--render none` skips rendering for a fast, display-free numeric eval.
  - `--draw-view-cones`, `--draw-force-vectors`, `--draw-hit-boxes`,
    `--draw-death-circles` toggle Aquarium's debug overlays.
  - On a headless box, run with `SDL_VIDEODRIVER=dummy` to render off-screen
    (still works with `--render video`; pygame needs a real display surface
    even though nothing is shown).

## Install

Aquarium's `setup.py` pins exact old versions
(`gymnasium==0.28.1`, `numpy==1.22.4`, `pettingzoo==1.24.2`, `pygame==2.1.3`,
`moviepy==1.0.3`) that conflict with a modern `ray[rllib]` install (which pulls
`gymnasium>=1.0`). In practice the runtime code doesn't care — it was verified to
run correctly under `numpy==2.4.6`/`gymnasium==1.2.2`/`pygame==2.6.1`, just not
under those exact pinned versions. So: install ray first, then install
marl-aquarium **with `--no-deps`** so pip doesn't try (and fail) to downgrade
ray's dependencies to satisfy Aquarium's stale pins. The one real exception is
`moviepy`: Aquarium's video-export code imports
`moviepy.video.compositing.concatenate`, which moviepy 2.x removed/restructured —
that one genuinely needs the pinned `moviepy==1.0.3`.

```bash
# Run from dying_prey/ (the directory containing train.py).
conda create --prefix ../.conda python=3.11 pip -y
conda activate ../.conda
pip install "ray[rllib]==2.58.0" torch --extra-index-url https://download.pytorch.org/whl/cpu
pip install --no-deps marl-aquarium==0.1.10
pip install "pettingzoo==1.24.2" "pygame==2.6.1"
pip install moviepy==1.0.3
```

Use a CUDA-enabled PyTorch build for GPU training. Training defaults to one
GPU; set `config_ppo["num_gpus_per_learner"] = 0` for CPU-only runs.

## Run

Edit `config/config_env.py` for population, episode length, observations,
reward scaling, shaping, action repeat, and respawning. Edit
`config/config_ppo.py` for PPO hyperparameters, resources, iterations, and output
paths. The current run configuration uses 2,500 training iterations. Checkpoints and
TensorBoard logs are saved under the existing `dying_prey/runs/`
in a shared `dying_prey_<timestamp>` directory. The timestamp uses Amsterdam
time and includes microseconds. `{run_name}` in either output path is replaced
when training starts; set a path to `None` to disable that output.
Periodic checkpoints are saved every 10 iterations.

From the repository root:

```bash
.conda/bin/python dying_prey/train.py
```

For example, configure a longer permanent-death run by editing the dictionaries:

```python
# config/config_env.py
"keep_prey_count_constant": False,
"predator_shaping": 1.0,
"action_repeat": 4,

# config/config_ppo.py
"iterations": 100,
"tensorboard_dir": "runs/my_training/tensorboard",
"checkpoint_dir": "runs/my_training/checkpoint",
```

Then train with the same command and evaluate the saved checkpoint:

```bash
.conda/bin/python dying_prey/eval.py --checkpoint runs/my_training/checkpoint --episodes 3
.conda/bin/python dying_prey/eval.py --random --predator-count 1 --prey-count 4 --episodes 3
```

Evaluation retains its command-line options. Both training modes were previously
smoke-tested end to end. For inline CPU debugging, set both `num_env_runners`
and `num_gpus_per_learner` to 0 in `config/config_ppo.py`.

**Known upstream bug, worked around here**: Aquarium's own `env.close()` calls
`sys.exit()` unconditionally (`marl_aquarium/env/aquarium.py:320` — apparently
meant for closing a pygame render window). A Codex review flagged this as more
than cosmetic: RLlib calls `close()` from `algo.stop()`, not just at interpreter
shutdown, so the `SystemExit` could kill the calling thread/process mid-cleanup.
`SafeParallelPettingZooEnv.close()` in `env_wrapper.py` catches it. Also fixed
per that review: `procreate=True` is now explicitly rejected (see docstring in
`make_env()` — it would create prey agent IDs the wrapper/policies can't
enumerate ahead of time), configured population/step/iteration counts must be positive,
and the PS `policy_mapping_fn` raises on an unrecognized agent-id prefix instead
of silently treating it as prey.

**Known upstream limitation, not fixed**: Aquarium's `reset(seed=...)` accepts a
seed but never uses it internally — initialization draws from Python's
process-global `random` module. `SafeParallelPettingZooEnv.reset()` seeds that
module for single-environment reproducibility, but this does **not** isolate
multiple Aquarium environments sharing one process (e.g. more than one env per
`num_env_runners` worker) — they'd still share one global RNG stream. Fixing
that properly requires an upstream Aquarium change (per-instance RNG instead of
`random.*` module-level calls).

## Extending beyond the paper's own scope

The original paper only ever RL-trains prey (predator is a fixed `NaivChase`
heuristic that isn't part of the published package). Aquarium's
observation/action/reward interfaces are symmetric between predator and prey, so
this port trains both by default — there's no structural reason not to.

## Learning defaults

Training now defaults to egocentric observations, reward scaling of 0.01,
entropy coefficient 0.01, learning rate 3e-4, GAE lambda 0.95, 10 PPO epochs,
separate actor/critic encoders, and gradient norm clipping at 0.5. These settings
reduce the geometry the network must learn, keep death penalties manageable,
and maintain exploration. They are starting points, not measured improvements
in catch rate or survival; compare several seeds against the random baseline.
All settings can be edited in the two configuration files.

`vf_clip_param` defaults to 1000: RLlib's new PPO learner caps **squared value
error**, so its default of 10 can cut off critic gradients for a scaled death
penalty of -10. The cap of 1000 gives that target additional headroom. Larger reward scales may require a larger cap.
`gamma` (default 0.99) applies per policy decision and is also passed to
potential-based predator shaping. With action repeat, rewards are summed and
shaping is applied once per decision; increasing repeat therefore changes the
physical-time discount horizon. Predator shaping and action repeat remain
opt-in, and prey still respawn unless `keep_prey_count_constant` is set to `False`.

Evaluation reports unscaled, unshaped rewards. Inspect species-level returns
and critic explained variance in TensorBoard; the combined episode return can
hide improvements in one species and regressions in the other.

## Resource defaults

The local throughput benchmark selected 16 environment runners (one CPU each),
one local GPU learner, a 4,800-step training batch, and minibatches of 1,024.
The main process reserves one additional CPU. BLAS and PyTorch use one thread
per process to avoid competing thread pools; Ray CPU allocations do not pin
workers to physical cores. The script exposes its module directory to remote
workers through PYTHONPATH, preserving existing entries.

On the Ryzen 9 7950X / RTX 5070 Ti, this processed about 3,042 environment
steps/second; 24 runners improved throughput by only 2%. This was a short
permanent-death workload benchmark, not a universal optimum or a learning-quality
comparison. GPU execution requires device access (outside Codex's restricted
sandbox on this machine). Explicit CPU-only debugging remains available:

Set `num_env_runners = 0` and `num_gpus_per_learner = 0` in the PPO dictionary.

Thread limits are set before NumPy/PyTorch imports for command-line runs. The
worker import-path setup targets a fresh local Ray instance started by this
script; an existing or remote cluster needs its own runtime environment/package
setup. RLlib uses cyclic minibatches: a non-divisible training batch wraps into
the next pass rather than discarding the remainder or creating a short final
minibatch. Species have different agent-sample counts, so divisibility of the
environment-step batch alone would not make both species' batches divisible.

## Measured network and training comparisons

On the fixed-opponent, one-predator/four-prey, permanent-death experiments, the
256×256 MLP was the best tested baseline. Smaller MLPs, the tested LSTM, and
entity attention did not improve the overall result at the matched interaction
budget. [Architecture results](../runs/architecture_comparison/RESULTS.md) and
[attention results](../runs/architecture_comparison/ATTENTION_RESULTS.md) record
all three seeds and limitations.

A subsequent weights-only warm-start study gave each saved MLP 200 additional
PPO iterations (960,000 new environment steps). Longer fixed-opponent predator
training improved catches from 1.45 to 1.74 on common fresh starts and improved
transfer to held-out heading prey. Mixing opponents gave similar primary catches
but weaker transfer. Longer prey training reduced greedy survival from 86.1% to
83.0% (fixed) or 79.0% (mixed); retaining the earlier prey checkpoints is preferable
for that task. These results do not establish jointly evolving self-play behavior.
[Study protocol and results](../runs/training_generalization/RESULTS.md) and
[recommended module paths](../runs/training_generalization/recommended_models.json)
are local experiment artifacts under `runs/`.

The recommended predator/prey pairs were subsequently tested together on fresh
starts. Improved predator catches increased in all three seed pairs: 1.11 to
1.36 with greedy actions and 1.19 to 1.52 with stochastic actions. Retained prey
survived 87.3% against the scripted searcher but 65.9% against the improved
learned predator in greedy evaluation, exposing limited opponent transfer.
[Joint matchup results](../runs/learned_matchups/RESULTS.md) include all seed
pairs, source fingerprints and limitations. This is fixed-policy evaluation,
not joint self-play training.

A later prey-pool study trained against four frozen learned predators plus the
searcher, holding out another predator lineage. Pool training improved survival
against its held-out improved checkpoint from 55.7% to 72.0% greedy and 56.1% to
61.7% stochastic across three prey seeds. Searcher survival fell from 83.7% to
72.5% greedy, so original checkpoints remain preferable for searcher-focused use.
[Prey-pool results](../runs/prey_predator_pool/RESULTS.md) and
[scenario-specific module paths](../runs/prey_predator_pool/recommended_models.json)
record the tradeoffs. These results cover one held-out predator lineage at two
stages and do not establish ongoing self-play stability.

## Run the recommended learned agents together

From the repository root:

```bash
.conda/bin/python dying_prey/eval.py --recommended --episodes 3
```

This opens the simulation with the improved predator and pool-trained prey,
using separate frozen CPU policies without starting Ray training workers.
The environment matches the study: one predator, four prey, egocentric
observations, and caught prey die permanently. Model seed 0 is the default;
`--model-seed 1` or `--model-seed 2` selects the other trained pairs.
The checkpoints under `runs/` must be present; they are local training artifacts.

Use `--render none` for numeric output, `--render video` to save a video,
and `--stochastic` to sample actions instead of choosing their maximum.
`--seed 0` sets the first episode seed; subsequent episodes increment it.
These are frozen learned behaviors; this command does not continue training.
