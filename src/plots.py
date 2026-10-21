import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

GRADE_LABELS = ["0 No DR", "1 Mild", "2 Moderate", "3 Severe", "4 PDR"]
BLUES = LinearSegmentedColormap.from_list("blues", ["#ffffff", "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#104281"])
INK, MUTED = "#1f1f1e", "#5f5e5a"
SERIES = [("#2a78d6", "o"), ("#eb6834", "s")]  # colour and marker, so the two lines differ without colour too
REFERENCE, GRID = "#b4b2a9", "#e6e5e0"


def _plain_axes(ax):
    ax.grid(color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(REFERENCE)
    ax.tick_params(colors=MUTED, labelsize=9, length=0)


def reliability_diagram(curves, out_path, min_count=20):
    # curves: list of (label, [(mean confidence, accuracy, count), ...])
    # bins with very few photos are left out of the picture, as one photo would show as 0% or 100%
    fig, ax = plt.subplots(figsize=(5.2, 5.0), dpi=200)
    ax.plot([0.2, 1], [0.2, 1], ls="--", lw=1, color=REFERENCE)
    for (label, bins), (color, marker) in zip(curves, SERIES):
        conf, acc, _ = zip(*[b for b in bins if b[2] >= min_count])
        ax.plot(conf, acc, color=color, lw=2, marker=marker, ms=6, label=label)
    ax.set_xlim(0.2, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel("Model confidence", fontsize=10, color=INK)
    ax.set_ylabel("Share correct", fontsize=10, color=INK)
    _plain_axes(ax)
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    fig.tight_layout()
    fig.savefig(out_path, facecolor="white")
    plt.close(fig)


def review_curve(flagged_pct, curves, out_path):
    # curves: list of (label, share caught in %); the dashed line is random flagging
    fig, ax = plt.subplots(figsize=(5.6, 4.6), dpi=200)
    ax.plot([0, max(flagged_pct)], [0, max(flagged_pct)], ls="--", lw=1, color=REFERENCE, label="Random choice")
    for (label, caught), (color, marker) in zip(curves, SERIES):
        ax.plot(flagged_pct, caught, color=color, lw=2, marker=marker, ms=4, markevery=5, label=label)
    ax.set_xlim(0, max(flagged_pct))
    ax.set_ylim(0, 100)
    ax.set_xlabel("Test photos sent for review (%)", fontsize=10, color=INK)
    ax.set_ylabel("Missed referrals caught (%)", fontsize=10, color=INK)
    _plain_axes(ax)
    ax.legend(frameon=False, fontsize=9, loc="lower right")
    fig.tight_layout()
    fig.savefig(out_path, facecolor="white")
    plt.close(fig)


def confusion_matrix_figure(cm, out_path):
    # colour shows the share of each true grade, the text shows the count and that share
    share = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(6.4, 5.6), dpi=200)
    ax.imshow(share, cmap=BLUES, vmin=0, vmax=1)
    n = cm.shape[0]
    for i in range(n):
        for j in range(n):
            dark = share[i, j] > 0.55
            ax.text(j, i - 0.1, f"{cm[i, j]}", ha="center", va="center", fontsize=11,
                    color="white" if dark else INK)
            pct = "<1%" if 0 < share[i, j] < 0.005 else f"{share[i, j]:.0%}"
            ax.text(j, i + 0.22, pct, ha="center", va="center", fontsize=8,
                    color="white" if dark else MUTED)
    # thin white gaps between cells
    ax.set_xticks(np.arange(-0.5, n), minor=True)
    ax.set_yticks(np.arange(-0.5, n), minor=True)
    ax.grid(which="minor", color="white", linewidth=2)
    ax.tick_params(which="minor", length=0)
    ax.set_xticks(range(n), GRADE_LABELS, fontsize=9, color=MUTED)
    ax.set_yticks(range(n), GRADE_LABELS, fontsize=9, color=MUTED)
    ax.tick_params(length=0)
    ax.set_xlabel("Predicted grade", fontsize=10, color=INK, labelpad=8)
    ax.set_ylabel("True grade", fontsize=10, color=INK, labelpad=8)
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.tight_layout()
    fig.savefig(out_path, facecolor="white")
    plt.close(fig)
