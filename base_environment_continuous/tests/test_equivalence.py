"""Exact equivalence with add_walls (SPEC.md, section 10).

Records trajectories of add_walls's environment in the milestone
configuration in a separate process (record_reference.py), replays the same
actions in this environment and compares every decision bit for bit:
observations (their bytes), rewards, flags and infos, and the full state
(animal positions, velocities, accelerations, orientations and energies,
grass patches, cluster centres and collapse flags, the grass and birth
random-number generators, ID counters, the step counter, the movement order
and the acting agents). Floats compare by their exact bits (so -0.0 differs
from 0.0) and by type (an int differs from an equal float).

Run from the repository root:
    .conda/bin/python -m pytest base_environment_continuous/tests
"""

import json
import os
import pickle
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
MODULE = HERE.parent
sys.path.insert(0, str(MODULE))

from environment import PredPreyGrassContinuous  # noqa: E402

# name: (episodes, decisions per episode, first seed, setting overrides,
#        events that must occur). Random actions.
CASES = {
    # Many short episodes: resets, early births, catches and starvation.
    "short": (40, 120, 0, {}, ("births", "caught", "starved")),
    # Long episodes: grass collapse, extinction and the time limit.
    "long": (3, 1100, 1000, {}, ("collapses", "extinct", "time_limit")),
    # One physics step per decision.
    "repeat1": (8, 600, 2000, {"action_repeat": 1}, ("births", "caught")),
    # Small caps and ID pools: births blocked by the cap, pools used up.
    "caps": (
        10,
        400,
        3000,
        {
            "reproduction_max_predators": 8,
            "reproduction_max_prey": 12,
            "reproduction_predator_pool": 9,
            "reproduction_prey_pool": 20,
        },
        ("births", "pool_exhausted", "capped"),
    ),
}


def snapshot(env):
    animals = {
        a.id: (a.position.x, a.position.y, a.velocity.x, a.velocity.y,
               a.acceleration.x, a.acceleration.y, a.orientation, a.energy)
        for a in env.predators + env.prey
    }
    grass = env.grass
    return {
        "animals": animals,
        "patches": [(p.position.x, p.position.y, p.ready_at, p.regrow, p.cluster)
                    for p in grass.patches],
        "centres": [(c.x, c.y) for c in grass.centres],
        "collapsed": list(grass.collapsed),
        "grass_counters": (grass.tick, grass.total_consumed, grass.total_collapses),
        "grass_rng": grass.rng.getstate(),
        "birth_rng": env.birth_rng.getstate(),
        "next_id": dict(env.next_id),
        "time_step": env.time_step,
        "order": [a.id for a in env.entities],
        "active": sorted(env.active),
    }


def exact(value):
    """A comparable form that tells apart float bits (-0.0 from 0.0) and
    types (1 from 1.0, Python from NumPy scalars)."""
    if isinstance(value, dict):
        return {key: exact(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return (type(value).__name__, [exact(item) for item in value])
    if isinstance(value, np.ndarray):
        return ("array", str(value.dtype), value.shape, value.tobytes())
    if isinstance(value, (float, np.floating)):
        return (type(value).__name__, float(value).hex())
    return (type(value).__name__, value)


@pytest.fixture(scope="module", params=sorted(CASES))
def case(request, tmp_path_factory):
    episodes, decisions, first_seed, overrides, required = CASES[request.param]
    folder = tmp_path_factory.mktemp(request.param)
    settings = folder / "overrides.json"
    settings.write_text(json.dumps(overrides))
    out = folder / "reference.pkl"
    subprocess.run(
        [sys.executable, str(HERE / "record_reference.py"), str(out),
         str(episodes), str(decisions), str(first_seed), str(settings)],
        check=True,
        cwd=MODULE.parent,
        env=dict(os.environ, PYTHONHASHSEED="0"),
    )
    return overrides, required, pickle.loads(out.read_bytes())


def test_identical_to_add_walls(case):
    overrides, required, reference = case
    env = PredPreyGrassContinuous(overrides)
    events = dict.fromkeys(
        ("births", "caught", "starved", "collapses", "extinct", "time_limit",
         "pool_exhausted", "capped"),
        0,
    )
    for episode in reference:
        seed = episode["seed"]
        ref_obs, ref_infos, ref_state = episode["reset"]
        obs, infos = env.reset(seed=seed)
        assert exact(obs) == exact(ref_obs), (seed, "reset")
        assert exact(infos) == exact(ref_infos), (seed, "reset")
        assert exact(snapshot(env)) == exact(ref_state), (seed, "reset")
        for i, (actions, r_obs, r_rew, r_terms, r_truncs, r_infos, r_state) in enumerate(
            episode["steps"]
        ):
            where = (seed, i)
            population = {"predator": len(env.predators), "prey": len(env.prey)}
            obs, rewards, terms, truncs, infos = env.step(actions)
            assert exact(snapshot(env)) == exact(r_state), where
            assert exact(obs) == exact(r_obs), where
            assert exact(rewards) == exact(r_rew), where
            assert exact(terms) == exact(r_terms), where
            assert exact(truncs) == exact(r_truncs), where
            assert exact(infos) == exact(r_infos), where
            for agent, info in infos.items():
                events["births"] += info.get("born", False)
                events["starved"] += info.get("starved", False)
                if agent.startswith("prey") and terms.get(agent) and not info.get("starved"):
                    events["caught"] += 1
            if truncs["__all__"] and env.prey and env.predators:
                events["time_limit"] += 1
            if not env.prey or not env.predators:
                events["extinct"] += 1
            if any(env.next_id[s] >= env.pool[s] for s in env.pool):
                events["pool_exhausted"] += 1
            if any(
                population[s] >= env.max_population[s]
                and any(
                    a.energy >= env.threshold[s]
                    for a in (env.predators if s == "predator" else env.prey)
                )
                for s in population
            ):
                events["capped"] += 1
        events["collapses"] += env.grass.total_collapses
    print(events)
    # The comparison must have exercised the paths this case is for.
    for event in required:
        assert events[event], (event, events)
