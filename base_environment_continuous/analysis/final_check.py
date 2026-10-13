"""Final check of a trained checkpoint: outcomes, catch risk, populations,
and behaviour (speed choices, time near grass, catches near grass).

Plays a checkpoint's policies (actions sampled as in training, each
species' agents in sorted order, one torch generator per episode seeded with
the episode's seed) for a number of seeded episodes in this environment.
Checkpoints of add_walls runs in the milestone world work too: this
environment reproduces add_walls exactly (SPEC.md, section 10), so all runs
are measured with the same code and seeds.

"Near grass" means within 64 units (two body diameters) of an available
patch, as in add_walls's RESULTS.md. Catch risk is prey caught per prey per
1,000 physics steps.

Usage, from the repository root:

    .conda/bin/python base_environment_continuous/analysis/final_check.py \\
        <checkpoint dir> <episodes> [out.json]

<checkpoint dir> is a run's checkpoint/ or checkpoint/iter_<n> directory.
Episodes use seeds 0, 1, ...
"""

import json
import math
import pickle
import sys
from collections import Counter
from pathlib import Path

MODULE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MODULE))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from ray.rllib.core.rl_module.rl_module import RLModule  # noqa: E402

from environment import DEFAULT_CONFIG, PredPreyGrassContinuous  # noqa: E402
from tournament import SPECIES, classify, sample_actions  # noqa: E402

NEAR_GRASS = 64.0

# add_walls settings without a counterpart here, with the only values the
# milestone world (and so this environment) allows.
ADD_WALLS_ONLY = {
    "reward_scale": 1.0,
    "prey_reward": 0,
    "prey_punishment": 0,
    "predator_reward": 0,
    "predator_shaping": 0.0,
    "obs_stack": 1,
    "target_speed_actions": True,
    "grass_food_reward": 0.0,
    "grass_predators_observe": True,
    "walls_layout": None,
    "energy_predator_acceleration_cost": 0.0,
    "energy_prey_acceleration_cost": 0.0,
}
IGNORED = {
    "render_mode", "render_window_size", "shaping_gamma", "predator_max_age",
    "walls_thickness", "walls_rectangles", "walls_occlusion", "walls_rays",
    "walls_visibility_cell",
}


def env_config(checkpoint):
    """The checkpoint's env_config, translated from add_walls if needed."""
    with open(Path(checkpoint) / "class_and_ctor_args.pkl", "rb") as f:
        saved = pickle.load(f)
    config = dict(saved["ctor_args_and_kwargs"][0][0]["env_config"])
    for key, value in ADD_WALLS_ONLY.items():
        if key in config and config.pop(key) != value:
            raise ValueError(f"{key} must be {value!r} for this environment")
    for key in IGNORED:
        config.pop(key, None)
    unknown = set(config) - set(DEFAULT_CONFIG)
    if unknown:
        raise ValueError(f"settings without a counterpart: {sorted(unknown)}")
    return config


def load_modules(checkpoint):
    modules = {}
    for species in SPECIES:
        path = (Path(checkpoint) / "learner_group" / "learner" / "rl_module"
                / f"{species}_policy")
        module = RLModule.from_checkpoint(str(path.resolve()))
        module.to("cpu")
        module.eval()
        modules[species] = module
    return modules


def near_grass_distance(env, position):
    patches = [p.position for p in env.grass.patches if env.grass.available(p)]
    return min((env.torus.distance(position, p) for p in patches), default=math.inf)


def play(env, modules, seed, stats):
    generator = torch.Generator().manual_seed(seed)
    catches = []
    move_prey = env._move_prey

    def recording_move_prey(prey, desired, caught):
        before = len(caught)
        move_prey(prey, desired, caught)
        if len(caught) > before:
            catcher = caught[-1][0]
            catches.append((catcher.velocity.mag(),
                            near_grass_distance(env, catcher.position)))

    env._move_prey = recording_move_prey
    try:
        obs, _ = env.reset(seed=seed)
        exposure = {"predator": 0.0, "prey": 0.0}
        caught = 0
        while True:
            ordered = {agent: obs[agent] for agent in sorted(obs)}
            actions = sample_actions(modules, ordered, generator)
            for animal in env.predators + env.prey:
                if animal.id not in env.active:
                    continue
                key = animal.species
                stats[f"{key}_decisions"] += 1
                stats[f"{key}_stopped"] += actions[animal.id] // 16 == 2
                stats[f"{key}_near_grass"] += (
                    near_grass_distance(env, animal.position) <= NEAR_GRASS
                )
            before = (env.time_step, len(env.predators), len(env.prey))
            obs, _, terms, truncs, infos = env.step(actions)
            steps = env.time_step - before[0]
            exposure["predator"] += steps * (before[1] + len(env.predators)) / 2
            exposure["prey"] += steps * (before[2] + len(env.prey)) / 2
            for agent, info in infos.items():
                if agent.startswith("prey") and terms.get(agent) and not info.get("starved"):
                    caught += 1
            if terms["__all__"] or truncs["__all__"]:
                break
    finally:
        env._move_prey = move_prey
    stats["catches"] += len(catches)
    stats["catches_near_grass"] += sum(d <= NEAR_GRASS for _, d in catches)
    stats["catches_standing"] += sum(speed < 0.5 for speed, _ in catches)
    return {
        "seed": seed,
        "outcome": classify(len(env.predators), len(env.prey)),
        "physics_steps": env.time_step,
        "prey_caught": caught,
        "prey_exposure": exposure["prey"],
        "mean_predators": exposure["predator"] / max(env.time_step, 1),
        "mean_prey": exposure["prey"] / max(env.time_step, 1),
    }


def main():
    checkpoint, episodes = sys.argv[1], int(sys.argv[2])
    torch.set_num_threads(1)
    env = PredPreyGrassContinuous(env_config(checkpoint))
    modules = load_modules(checkpoint)
    stats = Counter()
    rows = [play(env, modules, seed, stats) for seed in range(episodes)]
    outcomes = Counter(row["outcome"] for row in rows)
    summary = {
        "checkpoint": str(checkpoint),
        "episodes": episodes,
        "coexisting": outcomes["time_limit"],
        "prey_extinct": outcomes["prey_extinct"],
        "predators_extinct": outcomes["predators_extinct"],
        "both_extinct": outcomes["both_extinct"],
        "catch_risk": 1000 * sum(r["prey_caught"] for r in rows)
        / sum(r["prey_exposure"] for r in rows),
        "mean_predators": float(np.mean([r["mean_predators"] for r in rows])),
        "mean_prey": float(np.mean([r["mean_prey"] for r in rows])),
        "predators_stopped": stats["predator_stopped"] / stats["predator_decisions"],
        "prey_stopped": stats["prey_stopped"] / stats["prey_decisions"],
        "predators_near_grass": stats["predator_near_grass"] / stats["predator_decisions"],
        "prey_near_grass": stats["prey_near_grass"] / stats["prey_decisions"],
        "catches_near_grass": stats["catches_near_grass"] / max(stats["catches"], 1),
        "catches_by_standing_predator": stats["catches_standing"] / max(stats["catches"], 1),
    }
    for key, value in summary.items():
        print(f"{key}: {round(value, 3) if isinstance(value, float) else value}")
    if len(sys.argv) > 3:
        Path(sys.argv[3]).write_text(json.dumps({"summary": summary, "episodes": rows}, indent=1))


if __name__ == "__main__":
    main()
