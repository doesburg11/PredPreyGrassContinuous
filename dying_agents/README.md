# PredPreyGrassContinuous

A continuous-space counterpart to the grid-based
[PredPreyGrass](https://github.com/doesburg11/PredPreyGrass), built on Aquarium.
There is no grass layer yet: so far it adds agents that die for good
(`--no-respawn`) on top of the Aquarium port described below.

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
- `train.py` — PPO training script on the new API stack, with two modes that
  mirror the original paper's two conditions but extended to *both* species
  (the paper only ever trained prey, against a fixed heuristic predator):
  - `--mode il`: independent learning, one policy per individual agent
    (`predator_0`, `prey_0`, `prey_1`, ...).
  - `--mode ps`: parameter sharing, one shared policy per species
    (`predator_policy`, `prey_policy`).
  - `--checkpoint-dir <dir>` saves an RLlib checkpoint after training, for use
    with `eval.py`. Add `--checkpoint-every N` to also save one every N
    iterations, to `<dir>/iter_<NNNNNN>`.
- `eval.py` — rolls out either a checkpoint from `train.py --checkpoint-dir`
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
# Run from dying_agents/ (the directory containing train.py).
conda create --prefix ../.conda python=3.11 pip -y
conda activate ../.conda
pip install "ray[rllib]==2.58.0" torch --extra-index-url https://download.pytorch.org/whl/cpu
pip install --no-deps marl-aquarium==0.1.10
pip install "pettingzoo==1.24.2" "pygame==2.6.1"
pip install moviepy==1.0.3
```

(Swap the `torch` line for a CUDA wheel if you want GPU training; CPU is enough
for a smoke test.)

## Run

```bash
python train.py --mode ps --predator-count 1 --prey-count 4 --iterations 5
python train.py --mode il --predator-count 1 --prey-count 4 --iterations 5

# Train with a checkpoint, then watch/record the trained agents:
python train.py --mode ps --predator-count 1 --prey-count 4 --iterations 20 \
    --checkpoint-dir /tmp/aquarium-ckpt
python eval.py --checkpoint /tmp/aquarium-ckpt --episodes 3
python eval.py --checkpoint /tmp/aquarium-ckpt --episodes 1 --render video

# Or watch a uniform-random baseline, no checkpoint needed:
python eval.py --random --predator-count 1 --prey-count 4 --episodes 3
```

Both modes were smoke-tested end to end (real `PPOConfig().build_algo()` +
`.train()` calls, not just env reset/step) and produce a moving
`episode_return_mean` across iterations — e.g. a 2-iteration PS run went from
-104.8 to -53.5. `--num-env-runners 0` (the default) runs everything inline for
easy debugging; raise it for real training throughput.

**Known upstream bug, worked around here**: Aquarium's own `env.close()` calls
`sys.exit()` unconditionally (`marl_aquarium/env/aquarium.py:320` — apparently
meant for closing a pygame render window). A Codex review flagged this as more
than cosmetic: RLlib calls `close()` from `algo.stop()`, not just at interpreter
shutdown, so the `SystemExit` could kill the calling thread/process mid-cleanup.
`SafeParallelPettingZooEnv.close()` in `env_wrapper.py` catches it. Also fixed
per that review: `procreate=True` is now explicitly rejected (see docstring in
`make_env()` — it would create prey agent IDs the wrapper/policies can't
enumerate ahead of time), CLI population/step/iteration counts must be positive,
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
