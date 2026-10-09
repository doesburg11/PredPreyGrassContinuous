"""Play a checkpoint with other animals' energy inputs set to 0, and measure
catch risk (prey caught per prey per 1000 physics steps), to test which
energy inputs a policy uses (RESULTS.md, section 2). Needs a checkpoint
trained with energy_observe_others.

Usage, from the repository root:

    .conda/bin/python add_group_hunting/analysis/energy_ablation.py \
        add_group_hunting <absolute checkpoint dir> <blind> <episodes> <out.json>

blind: "none"; "predator" or "prey" (that species sees no other animal's
energy); "predator_slot" or "prey_slots" (predators lose only the other
predator's energy, or only the prey's). Episodes use seeds 0, 1, ...
"""

import json
import os
import sys
from pathlib import Path

ROOT = Path(sys.argv[1]).resolve()
CHECKPOINT = sys.argv[2]
BLIND = sys.argv[3]  # "none", "predator" or "prey"
EPISODES = int(sys.argv[4])
OUT = sys.argv[5]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

import numpy as np
import torch
from ray.rllib.algorithms.algorithm import Algorithm

from env_wrapper import make_env, register
from eval import select_actions

# Egocentric layout: own 4 values, then 4 slots of 7 (seen, dx, dy, distance,
# vx, vy, energy): the other animals' energies sit at these indices.
SLOT_ENERGY = [4 + 7 * slot + 6 for slot in range(4)]

register()
torch.set_num_threads(1)
torch.manual_seed(0)
algo = Algorithm.from_checkpoint(CHECKPOINT)
env = make_env(dict(algo.config.env_config, render_mode=None))
raw = env.par_env.aec_env.unwrapped
assert raw._ego_slot_energy is not None
stats = {"caught": 0, "exposure": 0.0, "steps": 0, "episodes": []}
inner_step = raw.step


def step(actions):
    stats["exposure"] += len(raw.prey)
    stats["steps"] += 1
    out = inner_step(actions)
    return out


inner_update = raw.update_prey


def update_prey(prey, predators, desired_velocity):
    result = inner_update(prey, predators, desired_velocity)
    if not prey.alive:
        stats["caught"] += 1
    return result


raw.step = step
raw.update_prey = update_prey
for episode in range(EPISODES):
    obs, _ = env.reset(seed=episode)
    done = False
    while not done:
        if BLIND != "none":
            # "predator"/"prey": that species sees no other energy;
            # "predator_slot"/"prey_slots": predators lose only the other
            # predator's energy (slot 0) or only the prey's (slots 1-3).
            species, indices = {
                "predator": ("predator", SLOT_ENERGY),
                "prey": ("prey", SLOT_ENERGY),
                "predator_slot": ("predator", SLOT_ENERGY[:1]),
                "prey_slots": ("predator", SLOT_ENERGY[1:]),
            }[BLIND]
            for agent in obs:
                if agent.startswith(species):
                    obs[agent] = np.array(obs[agent], copy=True)
                    obs[agent][indices] = 0.0
        actions = select_actions(
            algo, obs, algo.config.policy_mapping_fn, stochastic=True
        )
        obs, _, terms, truncs, _ = env.step(actions)
        done = terms["__all__"] or truncs["__all__"]
    stats["episodes"].append(
        {"prey_left": len(raw.prey), "predators_left": len(raw.predators)}
    )
env.close()
stats["catch_risk"] = 1000 * stats["caught"] / stats["exposure"]
Path(OUT).write_text(json.dumps(stats))
print(BLIND, round(stats["catch_risk"], 2), stats["steps"])
