"""The two paper figures, with unchanged curve statistics and visual encodings."""

import os
from scripts.paper.records import ROOT

os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".mpl-cache"))
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update(
    {
        "figure.dpi": 110,
        "savefig.dpi": 200,
        "font.family": "serif",
        "font.size": 10,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "legend.frameon": False,
    }
)
COLORS = ["#ff7f0e", "#1f77b4", "#d62728"]


def arms(analysis):
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.6), sharey=True)
    handles = []
    for ax, opt in zip(axes, ("AdamW", "SGD"), strict=True):
        curves = analysis["arm_plot"][opt]
        x = curves["epochs"]
        mean, sd = np.asarray(curves["mean"]), np.asarray(curves["sd"])
        for i, (a, color, style) in enumerate(
            zip((0.5, 1, 2), COLORS, ("-", "--", ":"), strict=True)
        ):
            ax.fill_between(
                x,
                np.clip(mean[i] - sd[i], 0, 1),
                np.clip(mean[i] + sd[i], 0, 1),
                color=color,
                alpha=0.18,
                linewidth=0,
                zorder=1,
            )
            (line,) = ax.plot(
                x,
                mean[i],
                color=color,
                linewidth=1.5,
                zorder=3,
                linestyle=style,
                label=f"$m={a:g}$",
            )
            if opt == "AdamW":
                handles.append(line)
        ax.set_title("AdamW" if opt == "AdamW" else "SGD+M", fontsize=13, pad=30)
        ax.set_xlabel("Epoch", fontsize=12)
        ax.set_xlim(0, 200)
        ax.set_xticks([0, 50, 100, 150, 200])
        ax.set_ylim(0, 1)
        ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1])
        ax.tick_params(axis="both", labelsize=11)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(False)
    axes[0].set_ylabel("Arm selection frequency", fontsize=12)
    fig.legend(
        handles,
        [h.get_label() for h in handles],
        loc="upper center",
        bbox_to_anchor=(0.5, 0.78),
        ncol=3,
        fontsize=11,
        handlelength=2.4,
        columnspacing=1.2,
    )
    fig.subplots_adjust(top=0.68, bottom=0.16, left=0.07, right=0.99, wspace=0.12)
    return fig


def effects(analysis):
    methods = ("best_fixed", "uniform_random", "frequency_matched")
    colors = ("#1f77b4", "#ff7f0e", "#d62728")
    markers = ("o", "s", "^")
    offsets = (0.24, 0, -0.24)
    rows = ("asym20", "sym20", "sym40")
    handles = []
    values = [
        c["comparisons"]["aees_minus_" + m]["metrics"]["peak"]
        for k, c in analysis["cells"].items()
        if k.startswith("cifar100")
        for m in methods
    ]
    lo = min(v["diff_ci_low"] for v in values)
    hi = max(v["diff_ci_high"] for v in values)
    pad = 0.08 * (hi - lo)
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.6), sharex=True, sharey=True)
    for ax, opt in zip(axes, ("AdamW", "SGD"), strict=True):
        ax.axvline(0, color="#999999", linewidth=1, zorder=1)
        for row, regime in enumerate(rows):
            cell = analysis["cells"][f"cifar100/{regime}/{opt}"]
            for method, color, marker, offset in zip(
                methods, colors, markers, offsets, strict=True
            ):
                stats = cell["comparisons"]["aees_minus_" + method]["metrics"]["peak"]
                v = stats["diff_mean"]
                h = ax.errorbar(
                    v,
                    2 - row + offset,
                    xerr=[[v - stats["diff_ci_low"]], [stats["diff_ci_high"] - v]],
                    fmt=marker,
                    color=color,
                    markersize=6,
                    markeredgecolor="black",
                    markeredgewidth=0.5,
                    linewidth=1.5,
                    capsize=2.5,
                    zorder=3,
                )
                if opt == "AdamW" and row == 0:
                    handles.append(h)
        ax.set_title("AdamW" if opt == "AdamW" else "SGD+M", fontsize=13, pad=30)
        ax.set_xlabel("AEES - control (pp)", fontsize=12)
        ax.set_xlim(lo - pad, hi + pad)
        ax.set_xticks([-1, 0, 2, 4, 6])
        ax.set_ylim(-0.6, 2.6)
        ax.tick_params(axis="both", labelsize=11)
        ax.tick_params(axis="y", length=0)
        ax.grid(False)
        ax.spines[["top", "right", "left"]].set_visible(False)
    axes[0].set_yticks([2, 1, 0])
    axes[0].set_yticklabels(["Asym. 20%", "Sym. 20%", "Sym. 40%"])
    fig.legend(
        handles,
        ["Best fixed", "Uniform random", "Frequency matched"],
        loc="upper center",
        bbox_to_anchor=(0.5, 0.78),
        ncol=3,
        fontsize=11,
        handlelength=1.2,
        columnspacing=1.4,
    )
    fig.subplots_adjust(top=0.68, bottom=0.18, left=0.14, right=0.99, wspace=0.12)
    return fig


def save(analysis, label, path):
    fig = arms(analysis) if label == "fig:trajectory-actions" else effects(analysis)
    for suffix in (".pdf", ".png"):
        fig.savefig(
            path.with_suffix(suffix),
            bbox_inches="tight",
            pad_inches=0.05,
            metadata={"Creator": "feedback-utility-experiments"},
        )
    plt.close(fig)
