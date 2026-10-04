"""Roll out a trained checkpoint (or a uniform-random baseline) and
visualize it with Aquarium's own pygame renderer.

train.py runs with render_mode=None (see env_wrapper.py) -- this script
is the first place render_mode="rgb_array" actually gets used. With
--checkpoint, it restores a full Algorithm from an RLlib checkpoint (see
train.py --checkpoint-dir) and steps it greedily (argmax over each
RLModule's action-dist logits -- no exploration), or with --stochastic
samples from those logits as training does. With --random, no
checkpoint/RLlib Algorithm is involved at all -- every agent just samples
its action space uniformly at random each step, as a baseline to compare
trained behavior against.

Usage:
    python eval.py --recommended --episodes 3
    python train.py --checkpoint-dir /tmp/aquarium-ckpt --iterations 20
    python eval.py --checkpoint /tmp/aquarium-ckpt --episodes 3
    python eval.py --checkpoint /tmp/aquarium-ckpt --episodes 1 --render video
    python eval.py --checkpoint /tmp/aquarium-ckpt --episodes 10 --render none
    python eval.py --random --predator-count 1 --prey-count 4 --episodes 3

--render window (default) opens Aquarium's live pygame window. --render
video does the same but also saves an mp4 per episode to --out-dir
(Aquarium always opens a real display window when rendering; on a
headless box, run with SDL_VIDEODRIVER=dummy to render off-screen and
still get the mp4s). --render none skips rendering entirely, for a fast,
display-free numeric eval.
"""

import argparse
from pathlib import Path

import numpy as np
import torch
from ray.rllib.algorithms.algorithm import Algorithm
from ray.rllib.core.columns import Columns

from config.config_env import config_env
from env_wrapper import make_env, register
from learned_policies import SpeciesPolicies, recommended_paths, species_mapping


@torch.inference_mode()
def select_actions(
    algo: Algorithm, obs: dict, mapping_fn, stochastic: bool = False
) -> dict:
    actions = {}
    for agent_id, agent_obs in obs.items():
        module = algo.get_module(mapping_fn(agent_id))
        if module is None:
            raise ValueError(f"No RLModule found for agent {agent_id!r}")
        obs_batch = torch.from_numpy(np.asarray(agent_obs, dtype=np.float32)).unsqueeze(
            0
        )
        fwd_out = module.forward_inference({Columns.OBS: obs_batch})
        logits = fwd_out[Columns.ACTION_DIST_INPUTS]
        if stochastic:
            # Sample like training does (a categorical over the logits).
            dist = torch.distributions.Categorical(logits=logits[0])
            actions[agent_id] = int(dist.sample())
        else:
            actions[agent_id] = int(torch.argmax(logits[0]))
    return actions


def select_random_actions(env, obs: dict) -> dict:
    return {agent_id: env.action_space[agent_id].sample() for agent_id in obs}


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError(f"must be a positive integer, got {parsed}")
    return parsed


def _fov_degrees(value: str) -> int:
    parsed = int(value)
    if not 0 < parsed <= 360:
        raise argparse.ArgumentTypeError(
            f"must be a field of view in (0, 360] degrees, got {parsed}"
        )
    return parsed


def save_video(frames: list, out_dir: str, episode: int, fps: int = 60) -> None:
    # imageio (an existing moviepy/Aquarium transitive dep), not moviepy
    # directly: moviepy==1.0.3's write_videofile is broken under the
    # `decorator>=5` this repo's other deps pull in (its co_varnames-based
    # fps-resolution decorator silently drops the fps kwarg -- see
    # https://github.com/Zulko/moviepy/issues/1319).
    import imageio.v2 as imageio

    Path(out_dir).mkdir(parents=True, exist_ok=True)
    out_path = Path(out_dir) / f"episode_{episode}.mp4"
    imageio.mimwrite(str(out_path), frames, fps=fps, codec="libx264")
    print(f"saved {out_path}")


def main():
    parser = argparse.ArgumentParser()
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--checkpoint",
        help="RLlib checkpoint dir (from train.py --checkpoint-dir)",
    )
    source.add_argument(
        "--recommended",
        action="store_true",
        help="Run improved learned predator and pool-trained prey together",
    )
    parser.add_argument("--model-seed", type=int, choices=[0, 1, 2], default=0)
    parser.add_argument("--seed", type=int, default=0, help="First episode seed")
    source.add_argument(
        "--random",
        action="store_true",
        help="No checkpoint -- every agent samples its action space "
        "uniformly at random",
    )
    parser.add_argument(
        "--predator-count",
        type=_positive_int,
        default=1,
        help="Only used with --random (a checkpoint's own env_config is "
        "reused otherwise)",
    )
    parser.add_argument(
        "--prey-count",
        type=_positive_int,
        default=4,
        help="Only used with --random",
    )
    parser.add_argument(
        "--max-time-steps",
        type=_positive_int,
        default=200,
        help="Only used with --random",
    )
    parser.add_argument(
        "--prey-fov",
        type=_fov_degrees,
        default=None,
        help="Override the prey field of view in degrees (Aquarium's default "
        "is 120; a --checkpoint policy was trained at whatever value it was "
        "trained with, so widening it here feeds that policy information it "
        "never learned to use -- a real behavior change, not just a wider "
        "drawn cone). Omit to keep the checkpoint's/default value.",
    )
    parser.add_argument(
        "--predator-fov",
        type=_fov_degrees,
        default=None,
        help="Same as --prey-fov but for predators (Aquarium's default is 150).",
    )
    parser.add_argument(
        "--no-respawn",
        action="store_true",
        help="Caught prey die for good instead of respawning. A --checkpoint "
        "trained with train.py --no-respawn already has this set.",
    )
    parser.add_argument("--episodes", type=_positive_int, default=1)
    parser.add_argument(
        "--stochastic",
        action="store_true",
        help="With --checkpoint or --recommended, sample each action from the policy's "
        "distribution (as during training) instead of taking the argmax.",
    )
    parser.add_argument(
        "--render", choices=["window", "video", "none"], default="window"
    )
    parser.add_argument(
        "--out-dir", default="videos", help="Where --render video saves mp4s"
    )
    parser.add_argument(
        "--fps",
        type=_positive_int,
        default=60,
        help="Frame rate of --render video mp4s. Frames are captured once per "
        "decision, so with action repeat lower it (e.g. 10) to keep the "
        "video watchable.",
    )
    parser.add_argument("--draw-view-cones", action="store_true")
    parser.add_argument("--draw-force-vectors", action="store_true")
    parser.add_argument("--draw-hit-boxes", action="store_true")
    parser.add_argument("--draw-death-circles", action="store_true")
    args = parser.parse_args()

    register()
    policies = None
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    if args.recommended:
        algo = None
        policies = SpeciesPolicies(recommended_paths(args.model_seed))
        mapping_fn = species_mapping
        env_config = {
            "predator_count": 1,
            "prey_count": 4,
            "max_time_steps": 200,
            "obs_mode": "egocentric",
            "keep_prey_count_constant": False,
            "action_repeat": 1,
        }
    elif args.random:
        algo = None
        mapping_fn = None
        env_config = dict(config_env)
        env_config.update(
            predator_count=args.predator_count,
            prey_count=args.prey_count,
            max_time_steps=args.max_time_steps,
        )
    else:
        algo = Algorithm.from_checkpoint(str(Path(args.checkpoint).resolve()))
        assert algo.config is not None, "restored Algorithm has no config"
        mapping_fn = algo.config.policy_mapping_fn
        env_config = dict(algo.config.env_config)

    env_config.update(
        # report raw Aquarium rewards even for a scaled-reward checkpoint
        reward_scale=1.0,
        predator_shaping=0.0,
        render_mode=None if args.render == "none" else "rgb_array",
        draw_view_cones=args.draw_view_cones,
        draw_force_vectors=args.draw_force_vectors,
        draw_hit_boxes=args.draw_hit_boxes,
        draw_death_circles=args.draw_death_circles,
    )
    if args.prey_fov is not None:
        env_config["prey_fov"] = args.prey_fov
    if args.predator_fov is not None:
        env_config["predator_fov"] = args.predator_fov
    if args.no_respawn:
        env_config["keep_prey_count_constant"] = False
    env = make_env(env_config)

    food_totals = []
    summary = []  # (predator_return, prey_return, prey_eaten) per episode
    try:
        for episode in range(1, args.episodes + 1):
            obs, _ = env.reset(seed=args.seed + episode - 1)
            frames = []
            episode_return = 0.0
            species_return = {"predator": 0.0, "prey": 0.0}
            prey_eaten = 0
            starved = {"predator": 0, "prey": 0}
            grass_eaten = 0
            steps = 0
            done = False
            while not done:
                if policies is not None:
                    actions = select_actions(policies, obs, mapping_fn, args.stochastic)
                elif algo is None:
                    actions = select_random_actions(env, obs)
                else:
                    actions = select_actions(
                        algo, obs, mapping_fn, stochastic=args.stochastic
                    )
                obs, rewards, terminateds, truncateds, infos = env.step(actions)
                grass_eaten += sum(
                    info.get("grass_eaten", 0) for info in infos.values()
                )
                episode_return += sum(rewards.values())
                for agent_id, reward in rewards.items():
                    is_predator = agent_id.startswith("predator")
                    species = "predator" if is_predator else "prey"
                    species_return[species] += reward
                    if infos.get(agent_id, {}).get("starved"):
                        starved[species] += 1
                        continue
                    # A prey that dies for good (--no-respawn) is terminated.
                    # A respawned prey isn't, but gets Aquarium's default
                    # prey_punishment (1000), which dwarfs every other
                    # per-step reward, so the threshold only fires on a catch.
                    if species == "prey" and (
                        terminateds.get(agent_id) or reward <= -500
                    ):
                        prey_eaten += 1
                steps += 1
                if args.render != "none":
                    # par_env is PettingZoo's aec_to_parallel_wrapper, whose
                    # render() takes no mode arg -- it uses the render_mode
                    # set via env_config at construction time. Aquarium's
                    # own render() draws to the pygame window (unless
                    # SDL_VIDEODRIVER=dummy is set) and, since render_mode
                    # is "rgb_array", also returns the frame as a
                    # (width, height, 3) array -- transpose to the
                    # (height, width, 3) video tooling expects.
                    frame = env.par_env.render()
                    if args.render == "video" and frame is not None:
                        frames.append(frame.transpose(1, 0, 2))
                done = terminateds["__all__"] or truncateds["__all__"]
            print(
                f"episode {episode}/{args.episodes}  "
                f"return={episode_return:.1f}  "
                f"predator_return={species_return['predator']:.1f}  "
                f"prey_return={species_return['prey']:.1f}  "
                f"prey_eaten={prey_eaten}  grass_eaten={grass_eaten}  "
                f"starved_predators={starved['predator']}  "
                f"starved_prey={starved['prey']}  steps={steps}"
            )
            food_totals.append(grass_eaten)
            summary.append(
                (
                    species_return["predator"],
                    species_return["prey"],
                    prey_eaten,
                )
            )
            if args.render == "video" and frames:
                save_video(frames, args.out_dir, episode, fps=args.fps)
        n = len(summary)
        if n:
            pred, prey, eaten = (sum(col) / n for col in zip(*summary))
            print(
                f"mean over {n} episodes: predator_return={pred:.1f}  "
                f"prey_return={prey:.1f}  prey_eaten={eaten:.2f}  "
                f"grass_eaten={sum(food_totals) / n:.2f}"
            )
    finally:
        env.close()
        if algo is not None:
            algo.stop()


if __name__ == "__main__":
    main()
