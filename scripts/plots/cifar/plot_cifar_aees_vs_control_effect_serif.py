#!/usr/bin/env python3
"""CIFAR-100 AEES-minus-control paired effect plot, serif-font variant.

Uses the same data sources, bootstrap machinery, manuscript validation,
layout, colors, and sizing as plot_cifar_aees_vs_control_effect.py. This
paper variant applies the shared serif typography from scripts.plots._style
and distinct marker shapes so controls remain identifiable without color.
It writes to a separate output name so both variants can coexist.

CLI:
    uv run python -m scripts.plots.cifar.plot_cifar_aees_vs_control_effect_serif

Outputs (on success):
    cifar_aees_vs_control_effect_serif.pdf
    cifar_aees_vs_control_effect_serif.png
    cifar_aees_vs_control_effect_serif.summary.txt

Outputs (on failure):
    cifar_aees_vs_control_effect_serif.MISSING.md
"""

from __future__ import annotations

import argparse
import pathlib
import sys
import traceback
from typing import Any

import matplotlib.pyplot as plt

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
SCRIPTS_TABLES_DIR = PROJECT_ROOT / "scripts" / "tables"
if str(SCRIPTS_TABLES_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_TABLES_DIR))

# Importing this applies the shared thesis/paper rcParams (serif font family,
# pdf.fonttype 42, etc.) at import time -- the same styling module used by
# scripts/plots/controllers/plot_arm_selection.py, so this figure's typography
# matches arm_selection_cifar_sym40_optimizer_paired.pdf exactly.
from scripts.plots import _style  # noqa: F401,E402

from scripts.tables._paired_stats import compare_paired  # noqa: E402
from scripts.tables.aggregate_cifar_attribution_controls import (  # noqa: E402
    aggregate as aggregate_open_loop_controls,
)
import make_cifar_noisy_fixed_lr_ablation_tables as fixed_lr  # noqa: E402


NAME = "cifar_aees_vs_control_effect_serif"

TASK_ORDER = ["asym20", "sym20", "sym40"]
ROW_LABELS = {"asym20": "Asym. 20%", "sym20": "Sym. 20%", "sym40": "Sym. 40%"}
OPTIMIZER_ORDER = ["AdamW", "SGD"]
OPTIMIZER_TITLE = {"AdamW": "AdamW", "SGD": "SGD+M"}

# Exact colors from scripts/plots/controllers/plot_arm_selection.py's
# _PAPER_ARM_COLORS = ["#ff7f0e", "#1f77b4", "#d62728"] (orange/blue/red, for
# m=0.5/1.0/2.0 respectively). That palette has no green; frequency-matched
# uses the palette's red in its place rather than inventing a new shade.
COLOR_BEST_FIXED = "#1f77b4"    # blue
COLOR_UNIFORM_RANDOM = "#ff7f0e"  # orange
COLOR_FREQUENCY_MATCHED = "#d62728"  # red

METHODS = ["best_fixed", "uniform_random", "frequency_matched"]
METHOD_LABEL = {
    "best_fixed": "Best fixed",
    "uniform_random": "Uniform random",
    "frequency_matched": "Frequency matched",
}
METHOD_COLOR = {
    "best_fixed": COLOR_BEST_FIXED,
    "uniform_random": COLOR_UNIFORM_RANDOM,
    "frequency_matched": COLOR_FREQUENCY_MATCHED,
}
METHOD_MARKER = {
    "best_fixed": "o",
    "uniform_random": "s",
    "frequency_matched": "^",
}
# Small vertical offsets within each noise row so the three controls' CIs
# don't overlap visually. Row spacing is 1.0, so +-0.24 leaves clear gaps.
METHOD_OFFSET = {"best_fixed": 0.24, "uniform_random": 0.0, "frequency_matched": -0.24}

# Validation targets (existing paper values); used only as a sanity check,
# never substituted for a computed value.
EXPECTED_PP = {
    ("asym20", "AdamW"): {"best_fixed": 3.24, "uniform_random": 0.51, "frequency_matched": 0.03},
    ("asym20", "SGD"): {"best_fixed": 5.07, "uniform_random": -0.55, "frequency_matched": 0.47},
    ("sym20", "AdamW"): {"best_fixed": 3.40, "uniform_random": -0.40, "frequency_matched": 1.16},
    ("sym20", "SGD"): {"best_fixed": 5.04, "uniform_random": -0.21, "frequency_matched": -0.13},
    ("sym40", "AdamW"): {"best_fixed": 2.53, "uniform_random": 1.99, "frequency_matched": 0.53},
    ("sym40", "SGD"): {"best_fixed": 4.78, "uniform_random": -0.09, "frequency_matched": 1.72},
}
VALIDATION_TOLERANCE_PP = 0.05


def _compute_best_fixed_effects(
    cifar_noisy_root: pathlib.Path,
) -> dict[tuple[str, str], dict[str, float]]:
    """AEES-minus-best-fixed paired effect per (task, optimizer).

    Reuses collect_runs/aggregate_runs/best_fixed_for/index_runs_by_seed from
    make_cifar_noisy_fixed_lr_ablation_tables.py unmodified, so the fixed
    multiplier selected here is identical to the one in the existing
    supplementary table (selected once per noise/optimizer cell by mean peak
    validation accuracy across seeds, never per seed).
    """
    runs = fixed_lr.collect_runs(cifar_noisy_root)
    if not runs:
        raise ValueError(f"No noisy CIFAR runs found under {cifar_noisy_root}")
    agg = fixed_lr.aggregate_runs(runs)
    by_seed = fixed_lr.index_runs_by_seed(runs)

    out: dict[tuple[str, str], dict[str, float]] = {}
    for task in TASK_ORDER:
        for optimizer in OPTIMIZER_ORDER:
            aees_seeds = by_seed.get((task, optimizer, "AEES"), {})
            if not aees_seeds:
                raise ValueError(f"Missing AEES runs for {task}/{optimizer}")
            best_fixed = fixed_lr.best_fixed_for(agg, task, optimizer, "best_val_mean")
            if best_fixed is None:
                raise ValueError(f"Missing fixed-multiplier runs for {task}/{optimizer}")
            best_fixed_method, _ = best_fixed
            fixed_seeds = by_seed.get((task, optimizer, best_fixed_method), {})
            shared = sorted(set(aees_seeds) & set(fixed_seeds))
            if not shared:
                raise ValueError(
                    f"No shared seeds between AEES and {best_fixed_method} "
                    f"for {task}/{optimizer}"
                )
            aees_vals = [aees_seeds[s].best_val for s in shared]
            fixed_vals = [fixed_seeds[s].best_val for s in shared]
            cmp = compare_paired(aees_vals, fixed_vals, scale=100.0)
            out[(task, optimizer)] = {
                "diff_mean": cmp.diff_mean,
                "diff_ci_low": cmp.diff_ci_low,
                "diff_ci_high": cmp.diff_ci_high,
                "best_fixed_method": best_fixed_method,
            }
    return out


def _compute_open_loop_effects(
    cifar_noisy_root: pathlib.Path,
    runpod_root: pathlib.Path,
) -> dict[tuple[str, str], dict[str, dict[str, float]]]:
    """AEES-minus-{uniform-random, frequency-matched} paired effects.

    Delegates to the existing scripts/tables/aggregate_cifar_attribution_controls
    aggregation unmodified, then extracts the peak_validation_accuracy paired
    comparisons per (task, optimizer).
    """
    summary = aggregate_open_loop_controls(
        aees_root=cifar_noisy_root, controls_root=runpod_root
    )
    if not summary["complete"]:
        incomplete = [c for c in summary["cells"] if not c["complete"]]
        raise ValueError(
            "Incomplete open-loop control cells: "
            + "; ".join(
                f"{c['noise_regime']}/{c['optimizer']}: {c['missing_run_ids']}"
                for c in incomplete
            )
        )

    out: dict[tuple[str, str], dict[str, dict[str, float]]] = {}
    for cell in summary["cells"]:
        key = (cell["noise_regime"], cell["optimizer"])
        cell_out: dict[str, dict[str, float]] = {}
        for method_key, comparison_key in (
            ("uniform_random", "aees_minus_uniform_random"),
            ("frequency_matched", "aees_minus_frequency_matched"),
        ):
            s = cell["comparisons"][comparison_key]["metrics"][
                "peak_validation_accuracy"
            ]["summary_percentage_points"]
            cell_out[method_key] = {
                "diff_mean": s["diff_mean"],
                "diff_ci_low": s["diff_ci_low"],
                "diff_ci_high": s["diff_ci_high"],
            }
        out[key] = cell_out
    return out


def _validate(
    effects: dict[tuple[str, str], dict[str, dict[str, float]]]
) -> list[str]:
    """Compare computed effects against known-good manuscript values.

    Returns a list of human-readable mismatch descriptions (empty if every
    computed mean is within VALIDATION_TOLERANCE_PP of the expected value).
    """
    mismatches: list[str] = []
    for (task, optimizer), expected in EXPECTED_PP.items():
        computed = effects[(task, optimizer)]
        for method, expected_val in expected.items():
            actual_val = computed[method]["diff_mean"]
            if abs(actual_val - expected_val) > VALIDATION_TOLERANCE_PP:
                mismatches.append(
                    f"{task}/{optimizer}/{method}: computed={actual_val:.2f} "
                    f"expected={expected_val:.2f} (tolerance={VALIDATION_TOLERANCE_PP})"
                )
    return mismatches


def _build_figure(
    effects: dict[tuple[str, str], dict[str, dict[str, float]]]
) -> plt.Figure:
    all_lo = [effects[k][m]["diff_ci_low"] for k in effects for m in METHODS]
    all_hi = [effects[k][m]["diff_ci_high"] for k in effects for m in METHODS]
    span = max(all_hi) - min(all_lo)
    pad = 0.08 * span
    xlim = (min(all_lo) - pad, max(all_hi) + pad)

    n_rows = len(TASK_ORDER)
    row_y = {task: n_rows - 1 - i for i, task in enumerate(TASK_ORDER)}

    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.6), sharey=True, sharex=True)

    legend_handles: list[Any] = []
    legend_labels: list[str] = []
    for ax, optimizer in zip(axes, OPTIMIZER_ORDER):
        ax.axvline(0.0, color="#999999", linewidth=1.0, zorder=1)
        for task in TASK_ORDER:
            y0 = row_y[task]
            cell = effects[(task, optimizer)]
            for method in METHODS:
                stats = cell[method]
                y = y0 + METHOD_OFFSET[method]
                lo_err = stats["diff_mean"] - stats["diff_ci_low"]
                hi_err = stats["diff_ci_high"] - stats["diff_mean"]
                color = METHOD_COLOR[method]
                container = ax.errorbar(
                    stats["diff_mean"], y,
                    xerr=[[lo_err], [hi_err]],
                    fmt=METHOD_MARKER[method], color=color, markersize=6,
                    markeredgecolor="black", markeredgewidth=0.5,
                    linewidth=1.5, capsize=2.5, zorder=3,
                )
                if optimizer == OPTIMIZER_ORDER[0] and task == TASK_ORDER[0]:
                    legend_handles.append(container)
                    legend_labels.append(METHOD_LABEL[method])

        ax.set_title(OPTIMIZER_TITLE[optimizer], fontsize=13, pad=30)
        ax.set_xlabel("AEES - control (pp)", fontsize=12)
        ax.set_xlim(*xlim)
        ax.set_xticks([-1, 0, 2, 4, 6])
        ax.set_ylim(-0.6, n_rows - 1 + 0.6)
        ax.tick_params(axis="both", labelsize=11)
        ax.grid(False)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_visible(False)

    axes[0].set_yticks([row_y[task] for task in TASK_ORDER])
    axes[0].set_yticklabels([ROW_LABELS[task] for task in TASK_ORDER])
    axes[0].tick_params(axis="y", length=0)
    axes[1].tick_params(axis="y", labelleft=False, length=0)

    fig.legend(
        legend_handles, legend_labels,
        loc="upper center", bbox_to_anchor=(0.5, 0.78),
        ncol=len(legend_labels), frameon=False, fontsize=11,
        handlelength=1.2, columnspacing=1.4,
    )

    fig.subplots_adjust(top=0.68, bottom=0.18, left=0.14, right=0.99, wspace=0.12)
    return fig


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cifar-noisy-root", type=pathlib.Path,
        default=pathlib.Path("archived_results/cifar_noisy"),
        help="Root directory with AEES + fixed-multiplier CIFAR-100 runs.",
    )
    parser.add_argument(
        "--runpod-root", type=pathlib.Path,
        default=pathlib.Path("archived_results/aees-runpod-results"),
        help="Root directory with uniform-random + frequency-matched open-loop control runs.",
    )
    parser.add_argument(
        "--out-dir", type=pathlib.Path,
        default=pathlib.Path("reproduced_artifacts/figures/cifar"),
        help="Directory in which to write the cifar_aees_vs_control_effect_serif outputs.",
    )
    args = parser.parse_args(argv)

    cifar_noisy_root = args.cifar_noisy_root
    if not cifar_noisy_root.is_absolute():
        cifar_noisy_root = PROJECT_ROOT / cifar_noisy_root
    runpod_root = args.runpod_root
    if not runpod_root.is_absolute():
        runpod_root = PROJECT_ROOT / runpod_root

    try:
        best_fixed_effects = _compute_best_fixed_effects(cifar_noisy_root)
        open_loop_effects = _compute_open_loop_effects(cifar_noisy_root, runpod_root)

        effects: dict[tuple[str, str], dict[str, dict[str, float]]] = {}
        for key in best_fixed_effects:
            effects[key] = {
                "best_fixed": best_fixed_effects[key],
                **open_loop_effects[key],
            }

        mismatches = _validate(effects)
        if mismatches:
            reason = (
                "Computed AEES-minus-control effects do not match the known-good "
                "manuscript values within tolerance "
                f"({VALIDATION_TOLERANCE_PP} pp):\n\n"
                + "\n".join(f"- {m}" for m in mismatches)
            )
            write_missing_local(args.out_dir, NAME, reason)
            return 1

        fig = _build_figure(effects)
    except Exception as exc:
        reason = (
            f"Failed to build {NAME}.\n\n"
            f"Exception: {exc!r}\n\n"
            f"Traceback:\n```\n{traceback.format_exc()}```\n"
        )
        write_missing_local(args.out_dir, NAME, reason)
        return 1

    out_dir = args.out_dir
    if not out_dir.is_absolute():
        out_dir = PROJECT_ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = out_dir / f"{NAME}.pdf"
    png_path = out_dir / f"{NAME}.png"
    fig.savefig(pdf_path, bbox_inches="tight", pad_inches=0.05)
    fig.savefig(png_path, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)

    summary_lines: list[str] = []
    summary_lines.append(f"cifar-noisy-root: {cifar_noisy_root}")
    summary_lines.append(f"runpod-root: {runpod_root}")
    summary_lines.append(
        "bootstrap: paired percentile bootstrap of seed-level differences, "
        "10000 resamples, 95% CI (2.5/97.5 percentiles)"
    )
    summary_lines.append("")
    summary_lines.append("Validation against known-good manuscript values: PASSED "
                          f"(tolerance={VALIDATION_TOLERANCE_PP} pp)")
    summary_lines.append("")
    for task in TASK_ORDER:
        for optimizer in OPTIMIZER_ORDER:
            cell = effects[(task, optimizer)]
            summary_lines.append(f"=== {ROW_LABELS[task]} / {OPTIMIZER_TITLE[optimizer]} ===")
            bf = cell["best_fixed"]
            summary_lines.append(
                f"  best_fixed (vs {bf['best_fixed_method']}): "
                f"{bf['diff_mean']:+.2f} [{bf['diff_ci_low']:+.2f}, {bf['diff_ci_high']:+.2f}] pp"
            )
            for method in ("uniform_random", "frequency_matched"):
                m = cell[method]
                summary_lines.append(
                    f"  {method}: {m['diff_mean']:+.2f} "
                    f"[{m['diff_ci_low']:+.2f}, {m['diff_ci_high']:+.2f}] pp"
                )
            summary_lines.append("")
    summary_lines.append("outputs:")
    summary_lines.append(f"  - {pdf_path}")
    summary_lines.append(f"  - {png_path}")

    summary_path = out_dir / f"{NAME}.summary.txt"
    summary_path.write_text("\n".join(summary_lines) + "\n")

    print(f"wrote {pdf_path}")
    print(f"wrote {png_path}")
    print(f"wrote {summary_path}")
    return 0


def write_missing_local(out_dir: pathlib.Path, name: str, reason: str) -> pathlib.Path:
    if not out_dir.is_absolute():
        out_dir = PROJECT_ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{name}.MISSING.md"
    path.write_text(f"# {name} — missing\n\n{reason}\n")
    print(reason, file=sys.stderr)
    return path


if __name__ == "__main__":
    sys.exit(main())
