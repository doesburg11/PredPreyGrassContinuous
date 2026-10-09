"""Count attacks and catches by the number of hunters near the prey, to test
whether predators hunt together (RESULTS.md, section 3).

For every attack (a predator touching a prey it may attack), records how
many predators were within the group-hunting radius of the prey (the
attacker included), the prey's and the hunters' summed energy, whether the
attack succeeded, and the number of living predators (to compare with how
many would be near if predators were spread at random). Run a group-hunting
checkpoint, and as a control a checkpoint trained without group hunting
played in the same group-hunting world (env_overrides).

Usage, from the repository root:

    .conda/bin/python add_group_hunting/analysis/group_catches.py \\
        <absolute checkpoint dir> <episodes> <out.json> [env_overrides JSON]

Episodes use seeds 0, 1, ...; actions are sampled as in training.
"""

import json
import math
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT = sys.argv[1]
EPISODES = int(sys.argv[2])
OUT = sys.argv[3]
OVERRIDES = json.loads(sys.argv[4]) if len(sys.argv) > 4 else {}
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

import torch
from ray.rllib.algorithms.algorithm import Algorithm

from env_wrapper import make_env, register
from eval import select_actions

register()
torch.set_num_threads(1)
torch.manual_seed(0)
algo = Algorithm.from_checkpoint(CHECKPOINT)
env_config = dict(algo.config.env_config, render_mode=None)
env_config.update(OVERRIDES)
env = make_env(env_config)
raw = env.par_env.aec_env.unwrapped
hunt = raw.group_hunting
energy = raw.energy
torus = raw.torus
attacks = []  # one record per attack
exposure = {"prey_steps": 0, "physics_steps": 0}


def distance(a, b):
    dx = b.position.x - a.position.x
    dy = b.position.y - a.position.y
    dx = (dx + torus.width / 2) % torus.width - torus.width / 2
    dy = (dy + torus.height / 2) % torus.height - torus.height / 2
    return math.hypot(dx, dy)


inner_update = raw.update_prey


def update_prey(prey, predators, desired_velocity):
    before = hunt.stats["attacks"]
    nearby = [p for p in predators if distance(p, prey) <= hunt.radius]
    record = {
        "nearby": len(nearby),
        "predators": len(predators),
        "prey_energy": energy.energy(prey),
        "hunters_energy": sum(energy.energy(p) for p in nearby),
    }
    result = inner_update(prey, predators, desired_velocity)
    made = hunt.stats["attacks"] - before
    for index in range(made):
        # Several touching predators attack in turn; only the last can succeed.
        caught = not prey.alive and index == made - 1
        attacks.append({**record, "caught": caught})
    return result


inner_step = raw.step


def step(actions):
    exposure["prey_steps"] += len(raw.prey)
    exposure["physics_steps"] += 1
    return inner_step(actions)


raw.update_prey = update_prey
raw.step = step
for episode in range(EPISODES):
    obs, _ = env.reset(seed=episode)
    done = False
    while not done:
        actions = select_actions(
            algo, obs, algo.config.policy_mapping_fn, stochastic=True
        )
        obs, _, terms, truncs, _ = env.step(actions)
        done = terms["__all__"] or truncs["__all__"]
    print(f"episode {episode + 1}: {len(attacks)} attacks so far", flush=True)
env.close()
Path(OUT).write_text(json.dumps({"attacks": attacks, **exposure}))
