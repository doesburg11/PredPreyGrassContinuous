"""Hand-coded predators against a learned prey: is there room to hunt better?

Each strategy replaces the learned predators and plays against one prey
checkpoint of a run, in that run's environment. The learned predators of a
checkpoint play the same prey for comparison. If a simple scripted strategy
catches clearly more than the learned predators, they are underperforming
(a learning problem); if none does, predators are near what this setup
allows.

Strategies (one of the 16 directions every decision, as learned predators):

- chase: steer at the nearest prey in the predator's own view cone;
- intercept: steer at where that prey will be on arrival (lead pursuit:
  its position plus its velocity times the time to reach it);
- chase_all / intercept_all: the same, but seeing every prey in the arena,
  an upper bound on information.

With no prey in sight, a scripted predator keeps its heading.

Usage (from add_group_hunting/):

    python scripted_predators.py runs/<run> --prey-iteration 270
    python scripted_predators.py runs/<run> --prey-iteration 270 --episodes 30 \\
        --learned-iteration 270 --strategies chase intercept

Writes one CSV row per episode (default: <run>/scripted_predators.csv) and
prints one row per strategy, as tournament.py --prey-iteration does.
"""

import argparse
import math
import os
import time
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from pathlib import Path

import tournament

ACTION_COUNT = 16
STRATEGIES = ("chase", "intercept", "chase_all", "intercept_all")


def wrapped_offset(raw_env, source, target):
    """Shortest (dx, dy) from source to target across the arena's edges."""
    dx = target.position.x - source.position.x
    dy = target.position.y - source.position.y
    dx = (dx + raw_env.width / 2) % raw_env.width - raw_env.width / 2
    dy = (dy + raw_env.height / 2) % raw_env.height - raw_env.height / 2
    return dx, dy


def action_towards(dx, dy):
    """The action whose direction is closest to (dx, dy). Action a moves
    along (sin t, -cos t) with t = a * 360 / 16 degrees (Aquarium's
    get_vector_from_action)."""
    angle = math.degrees(math.atan2(dx, -dy)) % 360
    return round(angle / (360 / ACTION_COUNT)) % ACTION_COUNT


class ScriptedPredators:
    """A tournament-compatible predator policy (see tournament.sample_actions)."""

    def __init__(self, strategy):
        if strategy not in STRATEGIES:
            raise ValueError(f"strategy must be one of {STRATEGIES}")
        self.strategy = strategy
        self.lead = strategy.startswith("intercept")
        self.omniscient = strategy.endswith("_all")

    def target(self, raw_env, predator):
        """The (dx, dy) to steer along, or None with no prey to go for."""
        candidates = []
        for prey in raw_env.prey:
            if (
                not self.omniscient
                and not raw_env.torus.check_if_entity_is_in_view_in_torus(
                    predator, prey, raw_env.predator_view_distance, raw_env.predator_fov
                )
            ):
                continue
            dx, dy = wrapped_offset(raw_env, predator, prey)
            candidates.append((math.hypot(dx, dy), dx, dy, prey))
        if not candidates:
            return None
        distance, dx, dy, prey = min(candidates, key=lambda c: c[0])
        if self.lead:
            # Time to close the distance at full speed; aim where the prey
            # will be by then if it keeps its velocity.
            time_to_reach = distance / max(raw_env.predator_max_velocity, 1e-6)
            dx += prey.velocity.x * time_to_reach
            dy += prey.velocity.y * time_to_reach
        return dx, dy

    def act(self, raw_env, agents):
        predators = {p.id(): p for p in raw_env.predators}
        actions = {}
        for agent in agents:
            predator = predators.get(agent)
            if predator is None:  # died this step; any action is ignored
                actions[agent] = 0
                continue
            direction = self.target(raw_env, predator)
            if direction is None:
                direction = (predator.velocity.x, predator.velocity.y)
            actions[agent] = action_towards(*direction)
        return actions


def _play(task):
    predator, prey_iter, episode, seed = task
    modules = {
        "predator": (
            ScriptedPredators(predator)
            if predator in STRATEGIES
            else tournament._module("predator", int(predator))
        ),
        "prey": tournament._module("prey", prey_iter),
    }
    result = tournament.play_episode(tournament._env(), modules, seed)
    return dict(
        predator=predator, prey_iter=prey_iter, episode=episode, seed=seed, **result
    )


def summary(rows):
    """Per predator (strategy or learned checkpoint), pooled over episodes."""
    table = []
    for predator in dict.fromkeys(row["predator"] for row in rows):
        episodes = [row for row in rows if row["predator"] == predator]
        caught = sum(row["prey_caught"] for row in episodes)
        predator_exposure = sum(
            row["mean_predators"] * row["physics_steps"] for row in episodes
        )
        outcomes = [row["outcome"] for row in episodes]
        table.append(
            {
                "predator": predator,
                "episodes": len(episodes),
                "catch_risk": 1000
                * caught
                / max(sum(r["prey_exposure"] for r in episodes), 1),
                "catches_per_predator": 1000 * caught / max(predator_exposure, 1),
                "coexist": outcomes.count("time_limit") / len(episodes),
                "prey_extinct": outcomes.count("prey_extinct") / len(episodes),
                "predators_extinct": outcomes.count("predators_extinct")
                / len(episodes),
                "length": sum(r["physics_steps"] for r in episodes) / len(episodes),
            }
        )
    return table


def main():
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    parser.add_argument("run", help="Training run directory (contains checkpoint/)")
    parser.add_argument("--prey-iteration", type=int, required=True)
    parser.add_argument(
        "--learned-iteration",
        type=int,
        default=None,
        help="Learned predator checkpoint to compare with (default: the prey's)",
    )
    parser.add_argument(
        "--strategies", nargs="+", choices=STRATEGIES, default=list(STRATEGIES)
    )
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2)
    )
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    if args.episodes < 1 or args.workers < 1:
        raise SystemExit("--episodes and --workers must be positive")

    run_dir = Path(args.run).resolve()
    available = tournament.checkpoint_iterations(run_dir)
    learned = (
        args.prey_iteration
        if args.learned_iteration is None
        else args.learned_iteration
    )
    for iteration in (args.prey_iteration, learned):
        if iteration not in available:
            raise SystemExit(f"No checkpoint for iteration {iteration}")
    locations = {i: (str(run_dir), i) for i in available}
    env_config = tournament.run_env_config(run_dir, args.prey_iteration)
    predators = [*args.strategies, str(learned)]
    tasks = [
        (predator, args.prey_iteration, episode, args.seed + episode)
        for predator in predators
        for episode in range(args.episodes)
    ]
    out = Path(args.out) if args.out else run_dir / "scripted_predators.csv"
    print(
        f"predators {predators} against prey {args.prey_iteration}: "
        f"{len(tasks)} episodes on {args.workers} workers",
        flush=True,
    )
    fields = ("predator", "prey_iter", "episode", "seed", *tournament.FIELDS[4:])
    rows = []
    start = time.time()
    with (
        ProcessPoolExecutor(
            max_workers=args.workers,
            mp_context=get_context("spawn"),
            initializer=tournament._init_worker,
            initargs=(locations, env_config),
        ) as pool,
        open(out, "w", encoding="utf-8") as csv,
    ):
        csv.write(",".join(fields) + "\n")
        for done, row in enumerate(pool.map(_play, tasks, chunksize=1), 1):
            rows.append(row)
            csv.write(",".join(str(row[field]) for field in fields) + "\n")
            csv.flush()
            if done % max(1, len(tasks) // 10) == 0 or done == len(tasks):
                minutes = (time.time() - start) / 60
                print(f"{done}/{len(tasks)} episodes, {minutes:.1f} min", flush=True)

    print(f"\nWrote {out}")
    print(f"Against prey {args.prey_iteration} ({args.episodes} episodes each):\n")
    print(
        " predator        catch risk  catches/predator  coexist  prey extinct"
        "  predators extinct  length"
    )
    for row in summary(rows):
        name = (
            row["predator"]
            if row["predator"] in STRATEGIES
            else f"learned {row['predator']}"
        )
        print(
            f" {name:<14}  {row['catch_risk']:>10.2f}"
            f"  {row['catches_per_predator']:>16.2f}"
            f"  {row['coexist']:>7.0%}  {row['prey_extinct']:>12.0%}"
            f"  {row['predators_extinct']:>17.0%}  {row['length']:>6.0f}"
        )


if __name__ == "__main__":
    main()
