"""How often do predators and prey choose full speed, half speed or stop,
and where? (RESULTS.md, section 18)

Plays a checkpoint trained with target_speed_actions in its own environment
(actions sampled as in training) and records, for every agent and decision,
its species, the chosen speed (action // 16: 0 full, 1 half, 2 stop), its
actual speed, and its distance to the nearest available grass patch and to
the nearest animal of the other species.

Usage, from the repository root:

    .conda/bin/python add_walls/analysis/speed_choices.py \\
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


def distance(a, b):
    dx = b.x - a.x
    dy = b.y - a.y
    dx = (dx + torus.width / 2) % torus.width - torus.width / 2
    dy = (dy + torus.height / 2) % torus.height - torus.height / 2
    return math.hypot(dx, dy)


records = []  # [species, speed choice, actual speed, grass distance, other distance]
for episode in range(EPISODES):
    obs, _ = env.reset(seed=episode)
    done = False
    while not done:
        actions = select_actions(
            algo, obs, algo.config.policy_mapping_fn, stochastic=True
        )
        patches = [p.position for p in grass.patches if grass.available(p)]
        for animal in raw.predators + raw.prey:
            agent = animal.id()
            if agent not in actions:
                continue
            predator = animal in raw.predators
            others = raw.prey if predator else raw.predators
            records.append(
                [
                    "predator" if predator else "prey",
                    int(actions[agent]) // 16,
                    round(animal.velocity.mag(), 2),
                    round(
                        min(
                            (distance(animal.position, p) for p in patches), default=-1
                        ),
                        1,
                    ),
                    round(
                        min(
                            (distance(animal.position, o.position) for o in others),
                            default=-1,
                        ),
                        1,
                    ),
                ]
            )
        obs, _, terms, truncs, _ = env.step(actions)
        done = terms["__all__"] or truncs["__all__"]
    print(f"episode {episode + 1}: {len(records)} records", flush=True)
env.close()
Path(OUT).write_text(json.dumps(records))
