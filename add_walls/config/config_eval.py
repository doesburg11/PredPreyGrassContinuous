"""Evaluation settings. Edit this dictionary before running eval.py."""

from pathlib import Path

RUNS_DIR = Path(__file__).resolve().parents[1] / "runs"

config_eval = {
    # What to run: "checkpoint" (a trained run, below), "random" (every agent
    # acts uniformly at random, as a baseline), or "recommended" (the older
    # 28-input dying_prey models; they do not fit this module's observations).
    "source": "random",  # add_walls has no trained runs yet.
    # A run's checkpoint/ directory (its final model) or one of its
    # checkpoint/iter_<n> directories. Only for "checkpoint".
    # Set to a run's checkpoint/ (or checkpoint/iter_<n>) once one exists.
    "checkpoint": str(RUNS_DIR / "<run>" / "checkpoint"),
    "model_seed": 0,  # Which recommended model (0, 1 or 2). Only for "recommended".
    "episodes": 1,
    "seed": 0,  # Seed of the first episode; episode k uses seed + k - 1.
    "stochastic": True,  # Sample actions as in training; False takes the argmax.
    # Only for "random" (a checkpoint keeps the settings it was trained with);
    # None takes the value from config_env.py.
    "predator_count": None,
    "prey_count": None,
    "max_time_steps": None,
    # Overrides for any source; None keeps the checkpoint's or config_env's
    # value. A wider view feeds a checkpoint's policy inputs it never learned
    # to use, so it changes behavior, not just the drawn cone.
    "prey_fov": None,  # Degrees, in (0, 360].
    "predator_fov": None,
    "no_respawn": False,  # True: caught prey die for good (already so here).
    # Environment settings that replace the checkpoint's (or config_env's)
    # values, e.g. {"grass_clustered": True} to test a policy trained on
    # scattered grass in clustered grass. Keys as in config_env.py.
    "env_overrides": {},
    # Display: "window" (live pygame window), "video" (also saves an mp4 per
    # episode in out_dir), or "none" (no rendering, fastest).
    "render": "window",
    # Window frame rate, in decisions per second; None keeps Aquarium's 60.
    # With action_repeat 8, 8 shows the simulation at about real time.
    "window_fps": 8,
    "out_dir": "videos",  # Where "video" saves its mp4s.
    "fps": 60,  # Frame rate of saved mp4s (one frame per decision).
    # Write episode, step, predators, prey and cumulative births per decision
    # to this CSV; None writes nothing.
    "population_csv": None,
    # One predator and one prey; selection switches when an animal dies.
    "draw_view_cones": True,
    "draw_force_vectors": False,
    "draw_hit_boxes": False,
    "draw_death_circles": False,
}
