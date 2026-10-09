"""At every catch, record the caught prey's energy and the energies of the
other living prey in the catcher's view cone at that moment, to test whether
predators choose prey by energy (RESULTS.md, section 2).

Usage, from the repository root (module: the module whose code the
checkpoint was trained with, e.g. add_walls or add_group_hunting):

    .conda/bin/python add_group_hunting/analysis/prey_selection.py \
        <module> <absolute checkpoint dir> <episodes> <output.json>

Episodes use seeds 1000, 1001, ...; actions are sampled as in training.
The output is a list of {"caught": energy, "others": [energies]}.
"""

import json
import os
import sys
from pathlib import Path

ROOT = Path(sys.argv[1]).resolve()
CHECKPOINT = sys.argv[2]
EPISODES = int(sys.argv[3])
OUT = sys.argv[4]
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
env_config = dict(algo.config.env_config)
env_config.update(render_mode=None)
env = make_env(env_config)
raw = env.par_env.aec_env.unwrapped
layer = raw.energy
torus = raw.torus
records = []
inner = raw.update_prey


def update_prey(prey, predators, desired_velocity):
    catcher = torus.get_colliding_animal(prey, predators)
    before = None
    if catcher is not None:
        seen = [
            p
            for p in raw.prey
            if p.alive
            and torus.check_if_entity_is_in_view_in_torus(
                catcher, p, raw.predator_view_distance, raw.predator_fov
            )
        ]
        if prey not in seen:
            seen.append(prey)
        before = (layer.energy(prey), [layer.energy(p) for p in seen if p is not prey])
    result = inner(prey, predators, desired_velocity)
    if before is not None and not prey.alive:
        records.append({"caught": before[0], "others": before[1]})
    return result


raw.update_prey = update_prey
for episode in range(EPISODES):
    obs, _ = env.reset(seed=1000 + episode)
    done = False
    while not done:
        actions = select_actions(
            algo, obs, algo.config.policy_mapping_fn, stochastic=True
        )
        obs, _, terms, truncs, _ = env.step(actions)
        done = terms["__all__"] or truncs["__all__"]
    print(f"episode {episode + 1}: {len(records)} catches so far", flush=True)
env.close()
Path(OUT).write_text(json.dumps(records))
