"""Chart of the arms race in runs AW-17 + AW-18 (RESULTS.md, section 23).

From a tournament CSV of runs that continue one another, plots for each
checkpoint: the catch risk of its predators averaged over all prey
checkpoints (rising: better hunters), of its prey averaged over all
predator checkpoints (falling: better at escaping), and against its own
training partner (flat: still evenly matched).

Usage, from the repository root:

    .conda/bin/python add_walls/analysis/plot_arms_race.py <tournament.csv> <out.png>
"""

import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SURFACE, INK, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1"
PREY, PREDATOR, PARTNER = "#2a78d6", "#eb6834", "#1baf7a"

games = pd.read_csv(sys.argv[1])
sums = games.groupby(["predator_iter", "prey_iter"]).agg(
    caught=("prey_caught", "sum"), exposure=("prey_exposure", "sum")
)
risk = (1000 * sums.caught / sums.exposure).unstack()
iterations = np.array(risk.index)
lines = [
    (risk.mean(axis=1).values, PREDATOR, "Predators against all prey"),
    (risk.mean(axis=0).values, PREY, "Prey against all predators"),
    (np.diag(risk.values), PARTNER, "Against its own training partner"),
]

fig, ax = plt.subplots(figsize=(9, 5.2), dpi=150)
fig.patch.set_facecolor(SURFACE)
ax.set_facecolor(SURFACE)
ax.axvspan(2100, iterations[-1] + 60, color=GRID, alpha=0.5, lw=0)
ax.text(2150, 4.62, "plateau", color=MUTED, fontsize=10)
for values, color, label in lines:
    ax.plot(iterations, values, color=color, lw=2, solid_capstyle="round",
            marker="o", ms=5, mec=SURFACE, mew=1.5, label=label)
    ax.annotate(label, (iterations[-1], values[-1]), xytext=(10, 0),
                textcoords="offset points", va="center", fontsize=10, color=INK)
ax.set_xlabel("Training checkpoint (iterations)", color=MUTED, fontsize=10)
ax.set_ylabel("Catch risk (prey caught per prey\nper 1,000 physics steps)",
              color=MUTED, fontsize=10)
ax.set_xlim(iterations[0] - 100, iterations[-1] + 60)
ax.set_ylim(1.8, 4.8)
ax.grid(axis="y", color=GRID, lw=1)
for side in ("top", "right"):
    ax.spines[side].set_visible(False)
for side in ("left", "bottom"):
    ax.spines[side].set_color(GRID)
ax.tick_params(colors=MUTED, labelsize=9)
ax.legend(loc="lower center", frameon=False, fontsize=9, labelcolor=INK)
fig.suptitle("Both species improve against each other, yet stay evenly matched",
             x=0.02, y=0.975, ha="left", color=INK, fontsize=12, fontweight="bold")
ax.set_title("Rising orange: better hunters.  Falling blue: prey better at escaping.  "
             "Flat green: still evenly matched.", loc="left", color=MUTED, fontsize=9)
fig.tight_layout(rect=(0, 0, 1, 0.975))
fig.subplots_adjust(right=0.74)
fig.savefig(sys.argv[2], facecolor=SURFACE)
print("wrote", sys.argv[2])
