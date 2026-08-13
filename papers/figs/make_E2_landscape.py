"""Paper E2 opener: the in-context-emergence claim landscape.
Single rendered PNG. Verified positioning (each work's stance), not fabricated numbers:
  Olsson et al. 2022 (2209.11895): induction/copy emerges (abrupt phase change).
  Lee et al. 2023 (2306.14892): in-context RL emerges (supervised pretraining).
  Wang et al. 2025 (2405.13861): in-context TD emerges in the forward pass.
  Schaeffer et al. 2023 (2304.15004): emergence can be a metric artifact (mirage).
  This work: induction emerges (~1850 steps, clean positive); in-context TD = calibration-negative,
    single-crossing detector 42/45 -> sustained 1/45 at tau=0.7 (fires on noise).
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

fig, ax = plt.subplots(figsize=(7.6, 3.8))

phenomena = ["induction /\ncopy", "in-context RL", "in-context TD"]
xs = [0, 1, 2]

# published positive claims (top row)
claims = {0: "Olsson 2022", 1: "Lee 2023", 2: "Wang 2025"}
for x in xs:
    ax.scatter([x], [1.0], s=150, marker="o", color="#1f77b4", edgecolor="white", lw=1.0, zorder=3)
    ax.annotate(claims[x] + "\n(claims emergence)", (x, 1.0), textcoords="offset points",
                xytext=(0, 12), ha="center", fontsize=8, color="#1f77b4")

# this work's calibrated outcome (bottom row)
ax.scatter([0], [0.0], s=150, marker="s", color="#2ca02c", edgecolor="white", lw=1.0, zorder=3)
ax.annotate("this work:\nemerges $\\approx$1850 steps\n(clean positive)", (0, 0.0),
            textcoords="offset points", xytext=(0, -14), ha="center", va="top",
            fontsize=8, color="#2ca02c")
ax.scatter([2], [0.0], s=170, marker="X", color="#d62728", edgecolor="white", lw=1.0, zorder=3)
ax.annotate("this work:\ncalibration-negative\ndetector 42/45 $\\to$ 1/45 sustained ($\\tau$=0.7)", (2, 0.0),
            textcoords="offset points", xytext=(0, -14), ha="center", va="top",
            fontsize=8, color="#d62728")
ax.annotate("(no calibrated\nnegative existed)", (1, 0.0), textcoords="offset points",
            xytext=(0, -14), ha="center", va="top", fontsize=7.6, color="#999999", style="italic")

# mirage caveat band spanning the row of claims
ax.text(1.0, 1.72, "Schaeffer et al. 2023: emergence can be a metric/measurement artifact (mirage)",
        ha="center", fontsize=8, color="#555555",
        bbox=dict(boxstyle="round,pad=0.3", fc="#fff7e6", ec="#e0c98a", lw=0.7))

ax.set_xticks(xs)
ax.set_xticklabels(phenomena, fontsize=9.5)
ax.set_yticks([0, 1])
ax.set_yticklabels(["this work\n(calibrated)", "published\nclaim"], fontsize=8.5)
ax.set_xlim(-0.5, 2.5)
ax.set_ylim(-0.75, 2.0)
ax.set_title("The in-context-emergence landscape: a calibrated negative where only claims existed",
             fontsize=10)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
fig.tight_layout()
fig.savefig("E2_landscape.png", dpi=200)
print("wrote E2_landscape.png")
