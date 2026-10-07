"""Optional direction and target-speed actions, preserving Aquarium physics."""

import numpy as np
from gymnasium.spaces import Discrete


def patch_target_speed(raw_env):
    """Encode full/half/zero speed as direction + action_count * speed_index.

    Existing scripted direction actions remain full speed. Limit the steering
    acceleration to the remaining velocity error so braking and low-speed
    targets do not overshoot. Aquarium still handles collisions and deaths.
    """
    directions = raw_env.action_count
    speeds = (1.0, 0.5, 0.0)
    original_desired = raw_env.get_desired_velocity_from_action
    raw_env.action_space = lambda agent: Discrete(directions * len(speeds))

    def desired(action, animal):
        if (
            not isinstance(action, (int, np.integer))
            or isinstance(action, bool)
            or not 0 <= action < directions * len(speeds)
        ):
            raise ValueError(
                "target-speed action must be an integer in the action space"
            )
        velocity = original_desired(int(action) % directions, animal)
        velocity.mult(speeds[int(action) // directions])
        return velocity

    def wrap_update(original, species):
        def update(animal, *args):
            target = args[-1]
            error = target.copy()
            error.sub(animal.velocity)
            name = f"{species}_max_acceleration"
            limit = getattr(raw_env, name)
            # Aquarium accumulates acceleration; target-speed steering must
            # follow this step's velocity error rather than an earlier force.
            animal.acceleration.mult(0)
            setattr(raw_env, name, min(limit, error.mag()))
            try:
                return original(animal, *args)
            finally:
                setattr(raw_env, name, limit)

        return update

    raw_env.get_desired_velocity_from_action = desired
    raw_env.update_predator = wrap_update(raw_env.update_predator, "predator")
    raw_env.update_prey = wrap_update(raw_env.update_prey, "prey")
