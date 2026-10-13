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

import importlib.util
import json
import os
import pickle
import subprocess
import sys
from collections import Counter
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
MODULE = HERE.parent
sys.path.insert(0, str(MODULE))
sys.path.insert(0, str(HERE))

from compare import CASES, count_events, exact, snapshot  # noqa: E402
from environment import PredPreyGrassContinuous  # noqa: E402

if (
    not (MODULE.parent / "add_walls" / "env_wrapper.py").is_file()
    or importlib.util.find_spec("marl_aquarium") is None
):
    pytest.skip(
        "needs add_walls and Aquarium; test_regression.py covers the same cases",
        allow_module_level=True,
    )


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
    events = Counter()
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
            count_events(env, population, terms, truncs, infos, events)
        events["collapses"] += env.grass.total_collapses
    print(events)
    # The comparison must have exercised the paths this case is for.
    for event in required:
        assert events[event], (event, events)
