"""Do predators team up specifically against strong prey? (RESULTS.md, section 7)

Group hunting only pays against well-fed prey (with steepness 2 and prey
strength 2), so purposeful cooperation would show as more company at
attacks on high-energy prey. Predators also crowd near grass, where prey are
well fed, so the same link can arise without any cooperation: the test is
whether it is stronger for trained group hunters than for a control
(predators never trained with group hunting, played in the same world).

Reads attack records written by group_catches.py and, per file, reports
the share of attacks with at least two predators near the prey by prey
energy quartile (pooled over all files' quartile limits), and a logistic
regression of "company" on the prey's energy and the log number of living
predators, with a bootstrap 95% interval for the prey-energy coefficient.

Usage, from the repository root:

    .conda/bin/python add_group_hunting/analysis/conditional_grouping.py \\
        <label>=<attacks.json> [<label>=<attacks.json> ...]
"""

import json
import sys

import numpy as np


def logistic(x, y, iterations=50):
    """Logistic regression by iteratively reweighted least squares."""
    x = np.column_stack([np.ones(len(x)), x])
    beta = np.zeros(x.shape[1])
    for _ in range(iterations):
        p = 1 / (1 + np.exp(-x @ beta))
        w = p * (1 - p) + 1e-9
        step = np.linalg.solve(x.T @ (x * w[:, None]), x.T @ (y - p))
        beta += step
        if np.max(np.abs(step)) < 1e-8:
            break
    return beta


def load(path):
    with open(path) as file:
        attacks = json.load(file)["attacks"]
    energy = np.array([a["prey_energy"] for a in attacks])
    company = np.array([a["nearby"] >= 2 for a in attacks], dtype=float)
    predators = np.array([a["predators"] for a in attacks], dtype=float)
    return energy, company, predators


def main(pairs):
    data = {label: load(path) for label, path in (p.split("=", 1) for p in pairs)}
    edges = np.quantile(
        np.concatenate([d[0] for d in data.values()]), [0.25, 0.5, 0.75]
    )
    print("prey energy quartile limits:", np.round(edges, 2))
    rng = np.random.default_rng(0)
    for label, (energy, company, predators) in data.items():
        quartile = np.searchsorted(edges, energy)
        shares = [company[quartile == q].mean() for q in range(4)]
        counts = [int((quartile == q).sum()) for q in range(4)]
        x = np.column_stack([(energy - 5.0), np.log(predators)])
        beta = logistic(x, company)
        boot = []
        for _ in range(500):
            i = rng.integers(0, len(energy), len(energy))
            boot.append(logistic(x[i], company[i])[1])
        low, high = np.percentile(boot, [2.5, 97.5])
        print(
            f"{label}: {len(energy)} attacks; company by quartile "
            + ", ".join(f"{s:.0%} (n={n})" for s, n in zip(shares, counts))
        )
        print(
            f"  logistic: prey energy {beta[1]:+.3f} per energy unit "
            f"(95% {low:+.3f} to {high:+.3f}), log predators {beta[2]:+.3f}"
        )


if __name__ == "__main__":
    main(sys.argv[1:])
