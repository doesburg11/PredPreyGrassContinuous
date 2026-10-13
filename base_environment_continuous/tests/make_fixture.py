"""Write tests/data/reference_digests.json: a SHA-256 digest of every reset
and decision of add_walls's reference trajectories (the cases in
compare.py), so test_regression.py can check this environment against
add_walls without add_walls or Aquarium installed.

Needs add_walls and its requirements; run from the repository root:

    .conda/bin/python base_environment_continuous/tests/make_fixture.py
"""

import json
import os
import pickle
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from compare import CASES, digest  # noqa: E402

OUT = HERE / "data" / "reference_digests.json"


def main():
    fixture = {}
    with tempfile.TemporaryDirectory() as folder:
        for name, (episodes, decisions, first_seed, overrides, _) in CASES.items():
            settings = Path(folder) / f"{name}.json"
            settings.write_text(json.dumps(overrides))
            out = Path(folder) / f"{name}.pkl"
            subprocess.run(
                [sys.executable, str(HERE / "record_reference.py"), str(out),
                 str(episodes), str(decisions), str(first_seed), str(settings)],
                check=True,
                cwd=HERE.parents[1],
                env=dict(os.environ, PYTHONHASHSEED="0"),
            )
            fixture[name] = {
                "settings": overrides,
                "episodes": [
                    {
                        "seed": episode["seed"],
                        "reset": digest(episode["reset"]),
                        "steps": [digest(step) for step in episode["steps"]],
                    }
                    for episode in pickle.loads(out.read_bytes())
                ],
            }
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(fixture, separators=(",", ":")) + "\n")
    total = sum(len(e["steps"]) for case in fixture.values() for e in case["episodes"])
    print(f"wrote {OUT}: {total} decisions")


if __name__ == "__main__":
    main()
