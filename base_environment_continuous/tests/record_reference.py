"""Record trajectories of add_walls's environment (Aquarium plus patches) in
the milestone configuration, as the reference for the equivalence tests.

Runs in its own process because add_walls and this module share file names.
Usage, from the repository root:

    .conda/bin/python base_environment_continuous/tests/record_reference.py \\
        <out.pkl> <episodes> <max_decisions> [first_seed] [overrides.json]

overrides.json holds settings that replace the milestone's, with keys that
both environments share (e.g. {"action_repeat": 1}).
"""

import json
import pickle
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "add_walls"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from compare import random_actions  # noqa: E402
from env_wrapper import make_env  # noqa: E402

# The milestone world (runs AW-17 to AW-22), in add_walls's config keys.
MILESTONE = {
    "predator_count": 6,
    "prey_count": 8,
    "max_time_steps": 8000,
    "render_mode": None,
    "reward_scale": 1.0,
    "prey_reward": 0,
    "prey_punishment": 0,
    "predator_reward": 0,
    "predator_shaping": 0.0,
    "action_repeat": 8,
    "obs_stack": 1,
    "target_speed_actions": True,
    "predator_max_velocity": 5,
    "prey_max_velocity": 6.0,
    "predator_max_acceleration": 0.6,
    "prey_max_acceleration": 1,
    "predator_max_steer_force": 0.6,
    "prey_max_steer_force": 0.6,
    "predator_radius": 16,
    "prey_radius": 16,
    "prey_fov": 240,
    "predator_fov": 150,
    "prey_view_distance": 267,
    "predator_view_distance": 200,
    "walls_layout": None,
    "grass_count": 100,
    "grass_consume_radius": 12.0,
    "grass_food_reward": 0.0,
    "grass_respawn_delay": 400,
    "grass_clustered": True,
    "grass_cluster_count": 20,
    "grass_cluster_spread": 24.0,
    "grass_random_respawn": False,
    "grass_dispersal": True,
    "grass_dispersal_distance": 24.0,
    "grass_overgrazing": True,
    "grass_overgrazing_threshold": 0.0,
    "grass_overgrazing_delay": 400,
    "grass_predators_observe": True,
    "energy_predator_initial": 5.0,
    "energy_prey_initial": 3.0,
    "energy_predator_max": 24.0,
    "energy_prey_max": 16.0,
    "energy_predator_resting_metabolic_cost": 0.25 * 0.15 / 8,
    "energy_prey_resting_metabolic_cost": 0.25 * 0.05 / 8,
    "energy_predator_speed_cost": 0.75 * 0.15 / 8 / 5,
    "energy_prey_speed_cost": 0.75 * 0.05 / 8 / 5,
    "energy_predator_acceleration_cost": 0.0,
    "energy_prey_acceleration_cost": 0.0,
    "energy_grass_gain": 2.0,
    "energy_catch_efficiency_predator": 0.5,
    "predator_max_age": 10**9,
    "reproduction_predator_threshold": 12.0,
    "reproduction_prey_threshold": 8.0,
    "reproduction_predator_reward": 10.0,
    "reproduction_prey_reward": 10.0,
    "reproduction_max_predators": 50,
    "reproduction_max_prey": 100,
    "reproduction_predator_pool": 2000,
    "reproduction_prey_pool": 2000,
}


def snapshot(env, raw):
    """The full simulation state, including what only future steps use."""
    animals = {
        a.id(): (a.position.x, a.position.y, a.velocity.x, a.velocity.y,
                 a.acceleration.x, a.acceleration.y, a.orientation_angle, a.energy)
        for a in raw.predators + raw.prey
    }
    grass = raw.grass
    return {
        "animals": animals,
        "patches": [(p.position.x, p.position.y, p.ready_at, p.regrow, p.cluster)
                    for p in grass.patches],
        "centres": [(c.x, c.y) for c in grass.centres],
        "collapsed": list(grass.collapsed),
        "grass_counters": (grass.tick, grass.total_consumed, grass.total_collapses),
        "grass_rng": grass.rng.getstate(),
        "birth_rng": raw.reproduction.rng.getstate(),
        "next_id": dict(raw.reproduction.next_id),
        "time_step": raw.time_step,
        "order": [a.id() for a in raw.all_entities],
        "active": sorted(raw.agents),
        "agents": sorted(env.agents),
        "births": dict(raw.reproduction.births),
    }


def as_array(value):
    return np.asarray(value, dtype=np.float32)


def record(episodes, max_decisions, first_seed, overrides=None):
    env = make_env(dict(MILESTONE, **(overrides or {})))
    raw = env.par_env.aec_env.unwrapped
    out = []
    try:
        for e in range(episodes):
            seed = first_seed + e
            rng = np.random.default_rng(seed)
            obs, infos = env.reset(seed=seed)
            episode = {"seed": seed, "reset": ({k: as_array(v) for k, v in obs.items()}, infos, snapshot(env, raw)), "steps": []}
            actions = random_actions(rng, obs)
            for _ in range(max_decisions):
                obs, rewards, terms, truncs, infos = env.step(actions)
                episode["steps"].append((actions, {k: as_array(v) for k, v in obs.items()},
                                         rewards, terms, truncs, infos, snapshot(env, raw)))
                if terms["__all__"] or truncs["__all__"]:
                    break
                actions = random_actions(rng, obs, terms, truncs)
            out.append(episode)
    finally:
        env.close()
    return out


if __name__ == "__main__":
    overrides = json.loads(Path(sys.argv[5]).read_text()) if len(sys.argv) > 5 else None
    episodes = record(int(sys.argv[2]), int(sys.argv[3]),
                      int(sys.argv[4]) if len(sys.argv) > 4 else 0, overrides)
    Path(sys.argv[1]).write_bytes(pickle.dumps(episodes))
    print(f"recorded {len(episodes)} episodes, "
          f"{sum(len(e['steps']) for e in episodes)} decisions")
