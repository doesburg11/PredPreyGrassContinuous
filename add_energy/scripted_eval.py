"""Scripted baselines: how many prey can a hand-written predator catch?

Gives a ceiling for what training should reach under the same env settings
as train.py's defaults (1 predator, 4 prey, 200 steps, egocentric
observations). Predator policies:

- sighted:   steer at the nearest prey in the predator's view cone (read from
             the egocentric observation), else keep going straight;
- omniscient: steer at the nearest prey anywhere (torus distance), an upper
             bound set by speeds, turning and episode length.

Prey policies:

- random: uniform random actions (trained prey behave close to this);
- flee:   run directly away from the predator when it's in view, else random.

Usage (from add_energy/):

    python scripted_eval.py --episodes 100
    python scripted_eval.py --episodes 100 --no-respawn
"""

import argparse
import math
import random

import numpy as np

from env_wrapper import make_env

ACTION_COUNT = 16


def action_towards(dx: float, dy: float) -> int:
    """Aquarium action whose direction is closest to (dx, dy).

    Action a moves along (sin t, -cos t) with t = a * 360 / ACTION_COUNT
    degrees (marl_aquarium/env/utils.py get_vector_from_action)."""
    angle = math.degrees(math.atan2(dx, -dy)) % 360
    return round(angle / (360 / ACTION_COUNT)) % ACTION_COUNT


def keep_heading(entity) -> int:
    return action_towards(entity.velocity.x, entity.velocity.y)


def wrapped_offset(raw_env, source, target):
    dx = target.position.x - source.position.x
    dy = target.position.y - source.position.y
    dx = (dx + raw_env.width / 2) % raw_env.width - raw_env.width / 2
    dy = (dy + raw_env.height / 2) % raw_env.height - raw_env.height / 2
    return dx, dy


def predator_action(kind, raw_env, predator, obs):
    if kind == "sighted":
        # Egocentric layout: 4 self values, 1 predator slot, then prey slots
        # [seen, dx, dy, distance, vx, vy], nearest first.
        seen, dx, dy = np.asarray(obs)[10:13]
        return action_towards(dx, dy) if seen else keep_heading(predator)
    offsets = [wrapped_offset(raw_env, predator, prey) for prey in raw_env.prey]
    if not offsets:
        return keep_heading(predator)
    dx, dy = min(offsets, key=lambda o: math.hypot(*o))
    return action_towards(dx, dy)


def prey_action(kind, obs, rng):
    if kind == "flee":
        seen, dx, dy = np.asarray(obs)[4:7]  # the predator slot
        if seen:
            return action_towards(-dx, -dy)
    return int(rng.integers(ACTION_COUNT))


def run(predator_kind, prey_kind, args):
    env = make_env(
        {
            "predator_count": 1,
            "prey_count": args.prey_count,
            "max_time_steps": args.max_time_steps,
            "obs_mode": "egocentric",
            "keep_prey_count_constant": not args.no_respawn,
            "action_repeat": args.action_repeat,
        }
    )
    raw_env = env.par_env.aec_env.unwrapped
    rng = np.random.default_rng(args.seed)
    catches = []
    for episode in range(args.episodes):
        random.seed(args.seed + episode)
        obs, _ = env.reset()
        eaten = 0
        while True:
            actions = {}
            for agent, agent_obs in obs.items():
                if agent.startswith("predator"):
                    predator = raw_env.predators[0]
                    actions[agent] = predator_action(
                        predator_kind, raw_env, predator, agent_obs
                    )
                else:
                    actions[agent] = prey_action(prey_kind, agent_obs, rng)
            obs, rewards, terminateds, truncateds, _ = env.step(actions)
            # Unshaped predator reward: predator_reward (10) per catch (summed
            # over the repeated sub-steps).
            eaten += round(rewards.get("predator_0", 0.0) / 10)
            if terminateds.get("__all__") or truncateds.get("__all__"):
                break
        catches.append(eaten)
    env.close()
    return np.mean(catches), np.std(catches) / math.sqrt(len(catches))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--prey-count", type=int, default=4)
    parser.add_argument("--max-time-steps", type=int, default=200)
    parser.add_argument("--no-respawn", action="store_true")
    parser.add_argument("--action-repeat", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    for predator_kind in ("sighted", "omniscient"):
        for prey_kind in ("random", "flee"):
            mean, sem = run(predator_kind, prey_kind, args)
            print(
                f"predator={predator_kind:10s} prey={prey_kind:6s} "
                f"prey_eaten={mean:.2f} +/- {sem:.2f}",
                flush=True,
            )


if __name__ == "__main__":
    main()
