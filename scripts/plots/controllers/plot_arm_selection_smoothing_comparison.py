"""Compare 5-, 11-, and 21-episode windows for AdamW and SGD+M.

Run from the repository root:
    python -m scripts.plots.controllers.plot_arm_selection_smoothing_comparison

Writes individual paper figures and a six-panel comparison using the same
selection data and per-seed centered moving averages as plot_arm_selection.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from scripts.plots.controllers import plot_arm_selection as arm
from scripts.plots._style import save_figure, write_summary


WINDOWS = (5, 11, 21)
SETTING = "cifar_sym40_optimizer_paired"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-root", type=Path,
                        default=Path("archived_results/cifar_noisy"))
    parser.add_argument("--out-dir", type=Path,
                        default=Path("reproduced_artifacts/figures/controllers/smoothing_comparison"))
    args = parser.parse_args()

    for window in WINDOWS:
        rc = arm.main([
            "--setting", SETTING,
            "--runs-root", str(args.runs_root),
            "--out-dir", str(args.out_dir),
            "--smoothing-window", str(window),
        ])
        if rc:
            return rc

    spec = arm.SETTING_SPEC[SETTING]
    fig, axes = plt.subplots(3, 2, figsize=(11, 9.3), sharex=True, sharey=True)
    summary = [
        f"runs-root: {args.runs_root}",
        f"windows (episodes): {WINDOWS}",
        "Lines: mean across seeds after per-seed centered moving averages.",
        "Bands: mean +/- 1 sample SD across seeds, clipped to [0, 1].",
        "Boundary windows average only available episodes.",
        "",
    ]
    for column, (setting, title) in enumerate(zip(
        spec["component_settings"], spec["panel_titles"]
    )):
        items = arm._load_runs(args.runs_root, setting)
        arm_values = sorted(items[0][1]["controller_logs"]["lr_controller_logs"]["arm_values"])
        _, n_episodes, seeds = arm._arm_fractions(items, "selected_lr_values", arm_values)
        total_epochs = items[0][1]["total_epochs"]
        episodes_per_epoch = n_episodes / float(total_epochs)
        x = np.arange(1, n_episodes + 1) / episodes_per_epoch
        summary.append(f"{title}: {n_episodes} episodes, {total_epochs} epochs, seeds={seeds}")

        for row, window in enumerate(WINDOWS):
            per_seed = arm._per_seed_smoothed_fractions(
                items, "selected_lr_values", arm_values, n_episodes, window
            )
            mean = per_seed.mean(axis=0)
            std = per_seed.std(axis=0, ddof=1)
            ax = axes[row, column]
            for index, (value, color) in enumerate(zip(arm_values, arm._PAPER_ARM_COLORS)):
                ax.fill_between(
                    x, np.clip(mean[index] - std[index], 0, 1),
                    np.clip(mean[index] + std[index], 0, 1),
                    color=color, alpha=0.18, linewidth=0,
                )
                ax.plot(x, mean[index], color=color, linewidth=1.3,
                        linestyle=arm._PAPER_ARM_LINESTYLES[index],
                        label=f"m={value:.1f}")
            epoch_window = window / episodes_per_epoch
            ax.set_title(f"{title} | {window} episodes (~{epoch_window:.1f} epochs)",
                         fontsize=12, pad=9)
            ax.set_xlim(0, 200)
            ax.set_ylim(0, 1)
            ax.set_xticks([0, 50, 100, 150, 200])
            ax.set_yticks(np.linspace(0, 1, 6))
            ax.tick_params(labelsize=10)
            ax.grid(False)
            if column == 0:
                ax.set_ylabel("Arm selection frequency", fontsize=11)
            if row == 2:
                ax.set_xlabel("Epoch", fontsize=11)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.suptitle("Arm selection: smoothing-window comparison", fontsize=15, y=0.99)
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.96),
               ncol=3, fontsize=11, frameon=False)
    fig.subplots_adjust(top=0.875, bottom=0.07, left=0.075, right=0.985,
                        hspace=0.32, wspace=0.12)
    name = "arm_selection_cifar_sym40_smoothing_comparison"
    pdf, png = save_figure(fig, args.out_dir, name)
    write_summary(args.out_dir, name, summary)
    print(f"wrote {pdf}")
    print(f"wrote {png}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
