"""Group hunting: a catch is a contest between the hunters and the prey.

When a predator touches a prey it attacks, and the attack succeeds with

    P(catch) = S^steepness / (S^steepness + (prey_strength * E)^steepness)

where S is the summed energy of every predator within `radius` of the prey
(the attacker included) and E is the prey's energy: a contest success
function, as used in economics and conflict models. Helpers count by being
close, not by touching. After a failed attack, that predator cannot attack
that prey again for `cooldown` physics steps, so the prey has a chance to
flee (contact lasts several steps, and repeated rolls would otherwise make
any chance nearly certain). A successful catch's energy is split equally
among the hunters (see energy.py).

Nothing forces predators to hunt together: a lone predator can still catch,
just less often. Whether teaming up pays depends on the settings. With
steepness 1, a pair's per-predator share of the chance is never higher
than a lone predator's chance; with steepness 2 and strong prey
(prey_strength around 2), it is, against well-fed prey. prey_strength 0
makes every attack succeed, as without group hunting.
"""

import math
import random


class GroupHunting:
    """Settings, random numbers and statistics of one environment."""

    def __init__(self, radius=64.0, prey_strength=2.0, steepness=2.0, cooldown=16):
        for name, value, positive in (
            ("radius", radius, True),
            ("prey_strength", prey_strength, False),
            ("steepness", steepness, True),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value < 0
                or (positive and value == 0)
            ):
                raise ValueError(
                    f"group_hunting_{name} must be finite and "
                    f"{'positive' if positive else 'nonnegative'}"
                )
        if type(cooldown) is not int or cooldown < 0:
            raise ValueError("group_hunting_cooldown must be an integer >= 0")
        self.radius = radius
        self.prey_strength = prey_strength
        self.steepness = steepness
        self.cooldown = cooldown
        self.rng = random.Random()
        # (predator id, prey id) -> first physics step it may attack again
        self.blocked = {}
        self.stats = self.empty_stats()

    @staticmethod
    def empty_stats():
        # catches_by_hunters[n]: catches made with n hunters near the prey
        return {"attacks": 0, "failed": 0, "catches_by_hunters": {}}

    def reset(self, seed=None):
        self.rng = random.Random(None if seed is None else seed + 2)
        self.blocked.clear()
        self.stats = self.empty_stats()

    def chance(self, hunters_energy, prey_energy):
        """Probability that one attack succeeds."""
        if self.prey_strength == 0 or prey_energy <= 0:
            return 1.0
        if hunters_energy <= 0:
            return 0.0
        # Raise only a ratio <= 1 to the power, so it cannot overflow.
        ratio = self.prey_strength * prey_energy / hunters_energy
        if ratio <= 1:
            return 1.0 / (1.0 + ratio**self.steepness)
        inverse = (1 / ratio) ** self.steepness
        return inverse / (inverse + 1.0)


def patch_group_hunting(raw_env, **settings):
    """Attach group hunting to one Aquarium instance. Needs the energy layer
    (patch_energy) and permanent prey deaths."""
    layer_energy = getattr(raw_env, "energy", None)
    if layer_energy is None:
        raise ValueError("Group hunting requires energy settings")
    layer = GroupHunting(**settings)
    raw_env.group_hunting = layer
    # prey id -> hunters of a successful catch; energy.py splits the prey's
    # energy among them.
    hunters_of = raw_env._catch_hunters = {}
    torus = raw_env.torus
    original_reset = raw_env.reset
    original_update_prey = raw_env.update_prey

    def distance(a, b):
        dx = b.position.x - a.position.x
        dy = b.position.y - a.position.y
        dx = (dx + torus.width / 2) % torus.width - torus.width / 2
        dy = (dy + torus.height / 2) % torus.height - torus.height / 2
        return math.hypot(dx, dy)

    def reset(seed=None, options=None):
        layer.reset(seed)
        hunters_of.clear()
        return original_reset(seed=seed, options=options)

    def update_prey(prey, predators, desired_velocity):
        now = raw_env.time_step
        touching = [p for p in predators if torus.collision(prey, p)]
        attackers = []
        for predator in touching:
            key = (predator.id(), prey.id())
            if layer.blocked.get(key, now) <= now:
                layer.blocked.pop(key, None)
                attackers.append(predator)
        catcher = None
        if attackers:
            nearby = [p for p in predators if distance(p, prey) <= layer.radius]
            layer.rng.shuffle(attackers)
            for attacker in attackers:
                # The attacker always counts, even with a radius below the
                # contact distance.
                hunters = nearby if attacker in nearby else [attacker] + nearby
                strength = sum(layer_energy.energy(p) for p in hunters)
                p_catch = layer.chance(strength, layer_energy.energy(prey))
                layer.stats["attacks"] += 1
                if layer.rng.random() < p_catch:
                    catcher = attacker
                    break
                layer.stats["failed"] += 1
                layer.blocked[(attacker.id(), prey.id())] = now + layer.cooldown + 1
        if catcher is None:
            # No attack succeeded: Aquarium must not see a touching predator.
            return original_update_prey(
                prey, [p for p in predators if p not in touching], desired_velocity
            )
        hunters_of[prey.id()] = [catcher] + [p for p in hunters if p is not catcher]
        by_size = layer.stats["catches_by_hunters"]
        by_size[len(hunters)] = by_size.get(len(hunters), 0) + 1
        return original_update_prey(prey, [catcher], desired_velocity)

    raw_env.reset = reset
    raw_env.update_prey = update_prey
