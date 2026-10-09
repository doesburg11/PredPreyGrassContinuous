"""Roll out a trained checkpoint (or a uniform-random baseline) and
visualize it with Aquarium's own pygame renderer.

Settings come from config/config_eval.py; edit it, then run from the
repository root (no command-line arguments):

    .conda/bin/python add_group_hunting/eval.py

source "checkpoint" restores a full Algorithm from an RLlib checkpoint and
samples actions from its policies as training does (stochastic=True) or
takes the argmax. source "random" involves no checkpoint at all: every agent
samples its action space uniformly at random, as a baseline.

render "window" opens Aquarium's live pygame window; "video" also saves an
mp4 per episode to out_dir (on a headless machine, set SDL_VIDEODRIVER=dummy
to render off-screen and still get the mp4s); "none" skips rendering, for a
fast numeric evaluation.
"""

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from ray.rllib.algorithms.algorithm import Algorithm
from ray.rllib.core.columns import Columns

from config.config_env import config_env
from config.config_eval import config_eval
from env_wrapper import make_env, register
from learned_policies import SpeciesPolicies, recommended_paths, species_mapping


@torch.inference_mode()
def select_actions(
    algo: Algorithm | SpeciesPolicies, obs: dict, mapping_fn, stochastic: bool = False
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


def limit_view_cones(raw_env):
    """Draw one persistent survivor per species; replace it after death/reset."""
    original_draw = raw_env.draw_view_cone_in_torus
    selected = {"predator": None, "prey": None}

    def draw(animal, view_distance, fov):
        for species, population in (
            ("predator", raw_env.predators),
            ("prey", raw_env.prey),
        ):
            living = [entity for entity in population if entity.alive]
            if not any(entity is selected[species] for entity in living):
                selected[species] = living[0] if living else None
        if any(animal is entity for entity in selected.values()):
            original_draw(animal, view_distance, fov)

    raw_env.draw_view_cone_in_torus = draw


SOURCES = ("checkpoint", "random", "recommended")
RENDERS = ("window", "video", "none")


def load_settings():
    """Copy and validate config_eval without modifying the source dictionary."""
    settings = dict(config_eval)
    if settings["source"] not in SOURCES:
        raise ValueError(f"source must be one of {SOURCES}")
    if settings["render"] not in RENDERS:
        raise ValueError(f"render must be one of {RENDERS}")
    if settings["source"] == "checkpoint" and not settings["checkpoint"]:
        raise ValueError('source "checkpoint" needs a checkpoint path')
    if settings["model_seed"] not in (0, 1, 2):
        raise ValueError("model_seed must be 0, 1 or 2")
    positive = ["episodes", "fps"]
    optional_positive = ["predator_count", "prey_count", "max_time_steps", "window_fps"]
    for name in positive + optional_positive:
        value = settings[name]
        if value is None and name in optional_positive:
            continue
        if type(value) is not int or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    overrides = settings["env_overrides"]
    if not isinstance(overrides, dict) or not all(
        isinstance(key, str) for key in overrides
    ):
        raise ValueError("env_overrides must be a dict of config_env settings")
    settings["env_overrides"] = dict(overrides)
    for name in ("prey_fov", "predator_fov"):
        value = settings[name]
        if value is not None and (type(value) is not int or not 0 < value <= 360):
            raise ValueError(f"{name} must be an integer in (0, 360] or None")
    for name in (
        "stochastic",
        "no_respawn",
        "draw_view_cones",
        "draw_force_vectors",
        "draw_hit_boxes",
        "draw_death_circles",
    ):
        if type(settings[name]) is not bool:
            raise ValueError(f"{name} must be True or False")
    if type(settings["seed"]) is not int:
        raise ValueError("seed must be an integer")
    return SimpleNamespace(**settings)


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
    args = load_settings()
    register()
    policies = None
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    if args.source == "recommended":
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
    elif args.source == "random":
        algo = None
        mapping_fn = None
        env_config = dict(config_env)
        for name in ("predator_count", "prey_count", "max_time_steps"):
            if getattr(args, name) is not None:
                env_config[name] = getattr(args, name)
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
    if args.window_fps is not None:
        env_config["fps"] = args.window_fps
    if args.prey_fov is not None:
        env_config["prey_fov"] = args.prey_fov
    if args.predator_fov is not None:
        env_config["predator_fov"] = args.predator_fov
    if args.no_respawn:
        env_config["keep_prey_count_constant"] = False
    env_config.update(args.env_overrides)
    if args.env_overrides:
        print(f"environment overrides: {args.env_overrides}")
    env = make_env(env_config)

    raw_env = env.par_env.aec_env.unwrapped
    if args.draw_view_cones:
        limit_view_cones(raw_env)
    csv_file = None
    if args.population_csv:
        # Closed in the finally block below.
        csv_file = open(args.population_csv, "w", encoding="utf-8")  # noqa: SIM115
        csv_file.write("episode,step,predators,prey,predator_births,prey_births\n")
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
            births = {"predator": 0, "prey": 0}
            peak = {"predator": len(raw_env.predators), "prey": len(raw_env.prey)}
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
                    births[species] += infos.get(agent_id, {}).get("births", 0)
                    if infos.get(agent_id, {}).get("starved"):
                        starved[species] += 1
                        continue
                    # A prey that dies for good (no respawn) is terminated.
                    # A respawned prey isn't, but gets Aquarium's default
                    # prey_punishment (1000), which dwarfs every other
                    # per-step reward, so the threshold only fires on a catch.
                    if species == "prey" and (
                        terminateds.get(agent_id) or reward <= -500
                    ):
                        prey_eaten += 1
                steps += 1
                peak["predator"] = max(peak["predator"], len(raw_env.predators))
                peak["prey"] = max(peak["prey"], len(raw_env.prey))
                if csv_file is not None:
                    csv_file.write(
                        f"{episode},{steps},{len(raw_env.predators)},"
                        f"{len(raw_env.prey)},{births['predator']},{births['prey']}\n"
                    )
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
                f"starved_prey={starved['prey']}  "
                f"births={births['predator']}/{births['prey']}  "
                f"peak={peak['predator']}/{peak['prey']}  "
                f"final={len(raw_env.predators)}/{len(raw_env.prey)}  "
                f"steps={steps}"
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
        if csv_file is not None:
            csv_file.close()
        env.close()
        if algo is not None:
            algo.stop()


if __name__ == "__main__":
    if len(sys.argv) > 1:
        raise SystemExit(
            "eval.py takes no command-line arguments. Edit config/config_eval.py."
        )
    main()
