"""How fast are predators moving when they catch? (RESULTS.md, section 21)

Plays a checkpoint in its own environment (actions sampled as in training)
and records, at every catch, the catcher's speed, the prey's speed, the
closing speed (how fast the gap between them was shrinking, positive when
approaching), the prey's speed towards the predator, and the catcher's
distance to the nearest available grass patch. A catch at (almost) zero
predator speed means a prey ran into a waiting predator.

Usage, from the repository root:

    .conda/bin/python add_walls/analysis/catch_speeds.py \\
        <absolute checkpoint dir> <episodes> <out.json>

Episodes use seeds 0, 1, ...
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
env = make_env(dict(algo.config.env_config, render_mode=None))
raw = env.par_env.aec_env.unwrapped
torus = raw.torus
grass = raw.grass


def offset(a, b):
    dx = b.x - a.x
    dy = b.y - a.y
    dx = (dx + torus.width / 2) % torus.width - torus.width / 2
    dy = (dy + torus.height / 2) % torus.height - torus.height / 2
    return dx, dy


catches = []
inner_update = raw.update_prey


def update_prey(prey, predators, desired_velocity):
    catcher = torus.get_colliding_animal(prey, predators)
    record = None
    if catcher is not None:
        dx, dy = offset(catcher.position, prey.position)
        gap = math.hypot(dx, dy) or 1.0
        ux, uy = dx / gap, dy / gap  # unit vector predator -> prey
        pv, qv = catcher.velocity, prey.velocity
        patches = [p.position for p in grass.patches if grass.available(p)]
        record = {
            "predator_speed": pv.mag(),
            "prey_speed": qv.mag(),
            # Rate at which the gap shrinks: predator towards prey minus
            # prey away from predator.
            "closing_speed": (pv.x * ux + pv.y * uy) - (qv.x * ux + qv.y * uy),
            "prey_towards_predator": -(qv.x * ux + qv.y * uy),
            "grass_distance": min(
                (math.hypot(*offset(catcher.position, p)) for p in patches),
                default=-1.0,
            ),
        }
    result = inner_update(prey, predators, desired_velocity)
    if record is not None and not prey.alive:
        catches.append(record)
    return result


raw.update_prey = update_prey
for episode in range(EPISODES):
    obs, _ = env.reset(seed=episode)
    done = False
    while not done:
        actions = select_actions(
            algo, obs, algo.config.policy_mapping_fn, stochastic=True
        )
        obs, _, terms, truncs, _ = env.step(actions)
        done = terms["__all__"] or truncs["__all__"]
    print(f"episode {episode + 1}: {len(catches)} catches", flush=True)
env.close()
Path(OUT).write_text(json.dumps(catches))
