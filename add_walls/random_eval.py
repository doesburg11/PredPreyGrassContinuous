"""View random agents using the current wall/environment configuration.

Run from the repository root:
    .conda/bin/python add_walls/random_eval.py

Environment settings come from config/config_env.py. Episode count, seed,
window speed and display options come from config/config_eval.py. This
launcher always selects random actions and a live pygame window.
"""

import eval as evaluation


def main():
    evaluation.config_eval.update(source="random", render="window")
    evaluation.main()


if __name__ == "__main__":
    main()
