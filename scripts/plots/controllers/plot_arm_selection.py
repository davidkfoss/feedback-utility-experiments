"""Selected-arm trajectory over training (stacked-area, Task 6).

Two settings produced by one CLI invocation each:
    --setting cifar_sym40    → LR-multiplier axis on noisy CIFAR-100 40% symmetric
                                (AdamW, AEES-LR flagship adamw_aees_ep200_lr05102).
    --setting agnews_noisy   → σ and LR axes on noisy AG News 20% symmetric
                                (AEES-Dual + warmup-linear flagship). Two-panel
                                figure: both axes shown because the σ-only
                                presentation hid that the LR axis is also
                                near-uniform.

For each setting and active axis, every episode is binned by the candidate-arm
value the controller picked at that episode. We stack the cross-seed
proportions to visualize whether the controller settled on one arm.

A centered rolling mean, with a window of roughly 5% of the episode count, is
applied for display only. The summary file reports raw proportions. The smoothing
preserves the columns-sum-to-1 property by averaging full episode slices.
The AdamW-vs-SGD+M paper figure uses an 11-episode window by default.
Use --smoothing-window to set an explicit positive odd window in episodes
(1 disables smoothing). Explicit-window outputs have a _w<window> suffix.

Typical reproduction commands:
    uv run python -m scripts.plots.controllers.plot_arm_selection \\
        --runs-root archived_results/cifar_noisy \\
        --out-dir reproduced_artifacts/figures/controllers \\
        --setting cifar_sym40

    uv run python -m scripts.plots.controllers.plot_arm_selection \\
        --runs-root archived_results/noisy_agnews \\
        --out-dir reproduced_artifacts/figures/controllers \\
        --setting agnews_noisy

Outputs on success:
    <out-dir>/arm_selection_<setting>.pdf
    <out-dir>/arm_selection_<setting>.png
    <out-dir>/arm_selection_<setting>.summary.txt

On failure:
    <out-dir>/arm_selection_<setting>.MISSING.md
"""

from __future__ import annotations

import argparse
import pathlib
import sys
import traceback
from typing import Any

import numpy as np
from matplotlib.transforms import Bbox

from scripts.plots._common import (
    RunInfo,
    identify,
    walk_runs,
)
from scripts.plots._style import (
    arm_palette,
    save_figure,
    write_missing,
    write_summary,
)

import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Per-setting / per-axis configuration. Each setting holds a list of axes; a
# multi-axis setting renders one stacked-area panel per axis (side-by-side,
# shared y).
# ---------------------------------------------------------------------------


AXIS_SPECS: dict[str, dict[str, Any]] = {
    "lr": {
        "base_key": "aees_lr",
        "selected_values_key": "selected_lr_values",
        "controller_logs_key": "lr_controller_logs",
        "axis_title": "LR multiplier",
        "arm_label_fmt": lambda v: f"m={float(v):.1f}",
    },
    "noise": {
        "base_key": "aees_noise",
        "selected_values_key": "selected_noise_values",
        "controller_logs_key": "noise_controller_logs",
        "axis_title": "Gradient noise σ",
        "arm_label_fmt": lambda v: ("σ=0" if float(v) == 0.0 else f"σ={v:g}"),
    },
}


SETTING_SPEC: dict[str, dict[str, Any]] = {
    "cifar_sym40": {
        "name": "arm_selection_cifar_sym40",
        "axes": ["lr"],
    },
    "cifar_sym40_sgd": {
        # Same layout as cifar_sym40 (AdamW) but for the SGD+M flagship, so
        # the two optimizers' arm-selection behavior can be compared.
        "name": "arm_selection_cifar_sym40_sgd",
        "axes": ["lr"],
    },
    "cifar_sym40_cosine": {
        # Same layout as cifar_sym40 but for the Cosine + AEES flagship.
        # Used to show that the controller's preferred arm pattern is
        # consistent with or modulated by the global decay schedule.
        "name": "arm_selection_cifar_sym40_cosine",
        "axes": ["lr"],
    },
    "cifar_sym40_paired": {
        # Side-by-side comparison: AEES-LR alone vs Cosine + AEES on the
        # 40% symmetric label noise setting. Renders the arm-selection
        # trajectories for both runs in a single two-panel figure, sharing
        # the y-axis so the AEES-alone vs Cosine+AEES contrast (preference
        # cements vs preference inverts) reads in a single glance.
        "name": "arm_selection_cifar_sym40_paired",
        "axes": ["lr"],  # both panels show the LR axis
        "component_settings": ["cifar_sym40", "cifar_sym40_cosine"],
        "panel_titles": ["AEES (no scheduler)", "Cosine + AEES"],
    },
    "cifar_sym40_optimizer_paired": {
        # LNCS-paper figure: AdamW vs SGD+M arm-selection trajectories,
        # side-by-side with a single shared legend and paper-scale fonts.
        # See _run_optimizer_paired for the dedicated layout.
        "name": "arm_selection_cifar_sym40_optimizer_paired",
        "axes": ["lr"],
        "component_settings": ["cifar_sym40", "cifar_sym40_sgd"],
        "panel_titles": ["AdamW", "SGD+M"],
        "paper_style": True,
        "smoothing_window": 11,
    },
    "agnews_noisy": {
        # Two panels: σ axis on the left, LR axis on the right. Showing both
        # is critical for the AG News chapter — the σ-only view hid that the
        # LR axis is also near-uniform on this task.
        "name": "arm_selection_agnews_noisy",
        "axes": ["noise", "lr"],
    },
}


# Default archived-results subfolder per setting (used when --runs-root is
# omitted). The cifar settings share archived_results/cifar_noisy; the AG News
# setting reads archived_results/noisy_agnews.
SETTING_RUNS_SUBDIR: dict[str, str] = {
    "cifar_sym40": "cifar_noisy",
    "cifar_sym40_sgd": "cifar_noisy",
    "cifar_sym40_cosine": "cifar_noisy",
    "cifar_sym40_paired": "cifar_noisy",
    "cifar_sym40_optimizer_paired": "cifar_noisy",
    "agnews_noisy": "noisy_agnews",
}


# ---------------------------------------------------------------------------
# Run-filtering predicates (per setting).
# ---------------------------------------------------------------------------


def _predicate_cifar_sym40(info: RunInfo, _record: dict[str, Any]) -> bool:
    if info.task != "cifar100":
        return False
    if info.noise_setting != "sym40":
        return False
    if info.optimizer != "AdamW":
        return False
    if info.path is None:
        return False
    if info.path.name != "adamw_aees_ep200_lr05102":
        return False
    return True


def _predicate_cifar_sym40_sgd(info: RunInfo, _record: dict[str, Any]) -> bool:
    if info.task != "cifar100":
        return False
    if info.noise_setting != "sym40":
        return False
    if info.optimizer != "SGD":
        return False
    if info.path is None:
        return False
    if info.path.name != "sgd_aees_ep200_lr05102_base01":
        return False
    return True


def _predicate_cifar_sym40_cosine(info: RunInfo, _record: dict[str, Any]) -> bool:
    if info.task != "cifar100":
        return False
    if info.noise_setting != "sym40":
        return False
    if info.optimizer != "AdamW":
        return False
    if info.path is None:
        return False
    if info.path.name != "adamw_cosine_aees_ep200_lr05102":
        return False
    return True


def _predicate_agnews_noisy(info: RunInfo, record: dict[str, Any]) -> bool:
    if info.task != "agnews":
        return False
    if info.noise_setting != "sym20":
        return False
    if info.path is None:
        return False
    if "aees_warmup_linear" not in info.path.name:
        return False
    cfg = record.get("config", {})
    lr_cands = cfg.get("lr_candidates") or []
    noise_cands = cfg.get("noise_candidates") or []
    # Need both axes active to plot both panels.
    if len(lr_cands) <= 1 or len(noise_cands) <= 1:
        return False
    return True


PREDICATES = {
    "cifar_sym40": _predicate_cifar_sym40,
    "cifar_sym40_sgd": _predicate_cifar_sym40_sgd,
    "cifar_sym40_cosine": _predicate_cifar_sym40_cosine,
    "agnews_noisy": _predicate_agnews_noisy,
}


# ---------------------------------------------------------------------------
# Loading.
# ---------------------------------------------------------------------------


def _load_runs(
    runs_root: pathlib.Path, setting: str
) -> list[tuple[RunInfo, dict[str, Any]]]:
    """Walk runs_root and filter to the flagship file(s) for the chosen setting."""
    pred = PREDICATES[setting]
    out: list[tuple[RunInfo, dict[str, Any]]] = []
    for path, record in walk_runs(runs_root):
        if record.get("controller_logs") is None:
            continue
        info = identify(record, path)
        if not pred(info, record):
            continue
        out.append((info, record))
    return out


# ---------------------------------------------------------------------------
# Arm-fraction matrix.
# ---------------------------------------------------------------------------


def _arm_fractions(
    items: list[tuple[RunInfo, dict[str, Any]]],
    selected_key: str,
    arm_values: list[float],
) -> tuple[np.ndarray, int, list[int]]:
    """Stack-friendly fraction matrix.

    Returns:
        fractions: ndarray of shape [n_arms, n_episodes], each column sums to 1.
        n_episodes: episode count after truncating to the shortest seed.
        seeds: sorted seed list.
    """
    items_sorted = sorted(
        items, key=lambda iv: (iv[0].seed if iv[0].seed is not None else -1)
    )
    seeds = [info.seed for info, _ in items_sorted]
    selected_lists = [r["episode_logs"][selected_key] for _, r in items_sorted]
    lengths = [len(s) for s in selected_lists]
    n_episodes = min(lengths)
    if n_episodes <= 0:
        raise ValueError(
            f"degenerate episode count after truncation: {n_episodes}")

    n_seeds = len(items_sorted)
    n_arms = len(arm_values)

    selected = np.array(
        [s[:n_episodes] for s in selected_lists], dtype=float
    )  # [n_seeds, n_episodes]

    arms_arr = np.asarray(arm_values, dtype=float)
    fractions = np.zeros((n_arms, n_episodes), dtype=float)
    for a_idx, a_val in enumerate(arms_arr):
        match = np.isclose(selected, a_val, atol=1e-9)
        fractions[a_idx, :] = match.sum(axis=0) / n_seeds

    col_sums = fractions.sum(axis=0)
    if not np.allclose(col_sums, 1.0, atol=1e-9):
        bad = np.where(~np.isclose(col_sums, 1.0, atol=1e-9))[0]
        raise ValueError(
            f"selected values include arm not in arm_values; "
            f"{len(bad)} episodes have column-sum != 1, "
            f"first bad column index = {int(bad[0])} sum = {col_sums[bad[0]]:.6f}"
        )

    return fractions, n_episodes, seeds


# ---------------------------------------------------------------------------
# Smoothing. Ported from scripts/plots_gpt/controllers/plot_arm_selection.py
# (window = ~5% of episode count, odd). Visual-only — summary reports raw
# proportions.
# ---------------------------------------------------------------------------


def _display_window(n_episodes: int, requested_window: int | None = None) -> int:
    """Use an explicit window or the default odd window of roughly 5%."""
    if requested_window is not None:
        if requested_window < 1 or requested_window % 2 == 0:
            raise ValueError("smoothing window must be a positive odd integer")
        return requested_window
    window = max(5, int(round(n_episodes * 0.05)))
    return window if window % 2 == 1 else window + 1


def _per_seed_smoothed_fractions(
    items: list[tuple[RunInfo, dict[str, Any]]],
    selected_key: str,
    arm_values: list[float],
    n_episodes: int,
    window: int,
) -> np.ndarray:
    """[n_seeds, n_arms, n_episodes] of smoothed per-seed selection fractions.

    For each seed, each arm-row is a 0/1 indicator vector over episodes (1 if
    that seed picked that arm in that episode), centered-rolling-meaned with
    the same window used for the stacked-area display. The resulting per-seed
    fractions allow drawing cross-seed mean ± SD bands per arm, which is the
    correct view for the "did the controller learn a preference?" question:
    the SD band shows how consistent the preference is across seeds.
    """
    items_sorted = sorted(
        items, key=lambda iv: (iv[0].seed if iv[0].seed is not None else -1)
    )
    n_seeds = len(items_sorted)
    n_arms = len(arm_values)
    arms_arr = np.asarray(arm_values, dtype=float)

    indicators = np.zeros((n_seeds, n_arms, n_episodes), dtype=float)
    for s_idx, (_, record) in enumerate(items_sorted):
        s_vals = np.asarray(
            record["episode_logs"][selected_key][:n_episodes], dtype=float
        )
        for a_idx, a_val in enumerate(arms_arr):
            indicators[s_idx, a_idx, :] = np.isclose(
                s_vals, a_val, atol=1e-9
            ).astype(float)

    if window <= 1:
        return indicators

    half = window // 2
    smoothed = np.empty_like(indicators)
    for i in range(n_episodes):
        lo = max(0, i - half)
        hi = min(n_episodes, i + half + 1)
        smoothed[:, :, i] = indicators[:, :, lo:hi].mean(axis=2)
    return smoothed


def _centered_rolling_mean(fractions: np.ndarray, window: int) -> np.ndarray:
    """Centered rolling mean along axis=1 (time). Renormalises columns to 1.

    Renormalisation is a no-op when the input columns already sum to 1 and
    every output bin uses the same time window, but we keep the explicit
    divide so the function tolerates ragged windows at the boundaries without
    drifting away from 1.
    """
    if window <= 1:
        return fractions
    n_arms, n_eps = fractions.shape
    out = np.empty_like(fractions, dtype=float)
    half = window // 2
    for i in range(n_eps):
        lo = max(0, i - half)
        hi = min(n_eps, i + half + 1)
        out[:, i] = fractions[:, lo:hi].mean(axis=1)
    col_sums = out.sum(axis=0, keepdims=True)
    return np.divide(
        out, col_sums, out=np.zeros_like(out), where=col_sums > 0
    )


# ---------------------------------------------------------------------------
# Per-seed disagreement diagnostic (raw, unsmoothed).
# ---------------------------------------------------------------------------


def _seed_disagreement(
    items: list[tuple[RunInfo, dict[str, Any]]],
    selected_key: str,
    arm_values: list[float],
    n_episodes: int,
) -> list[tuple[int, dict[float, float], float]]:
    items_sorted = sorted(
        items, key=lambda iv: (iv[0].seed if iv[0].seed is not None else -1)
    )
    selected_lists = [
        np.asarray(r["episode_logs"][selected_key][:n_episodes], dtype=float)
        for _, r in items_sorted
    ]
    arms_arr = np.asarray(arm_values, dtype=float)
    per_seed: list[dict[float, float]] = []
    for s_vals in selected_lists:
        seed_fracs = {}
        for a_val in arms_arr:
            seed_fracs[float(a_val)] = float(
                np.mean(np.isclose(s_vals, a_val, atol=1e-9))
            )
        per_seed.append(seed_fracs)
    group = {
        float(a): float(np.mean([s[float(a)] for s in per_seed])) for a in arms_arr
    }
    out: list[tuple[int, dict[float, float], float]] = []
    for (info, _), seed_fracs in zip(items_sorted, per_seed):
        l1 = sum(abs(seed_fracs[k] - group[k]) for k in group)
        out.append((info.seed, seed_fracs, l1))
    return out


# ---------------------------------------------------------------------------
# Plot construction.
# ---------------------------------------------------------------------------


def _make_multi_panel(n_panels: int):
    """N-panel figure with shared y."""
    if n_panels == 1:
        fig, ax = plt.subplots(figsize=(6.5, 4.0))
        return fig, [ax]
    width = 3.5 * n_panels + 0.5  # ~3.5in per panel
    fig, axes = plt.subplots(
        1, n_panels, sharey=True, figsize=(width, 4.0)
    )
    return fig, list(axes)


def _draw_per_arm_lines(
    ax,
    per_seed_smoothed: np.ndarray,
    arm_values: list[float],
    axis_spec: dict[str, Any],
    representative_record: dict[str, Any],
    *,
    show_title: bool = True,
    label_fontsize: float | None = None,
    tick_fontsize: float | None = None,
    legend_fontsize: float = 8,
) -> None:
    """Per-arm cross-seed mean trajectory with ±1 SD band; uniform reference.

    Reads more directly than the stacked-area display when the controller
    *does* learn a preference: one line lifts above the 1/K uniform line
    while the others drop below, and the SD band shows whether seeds agree.
    Used for the cifar_sym40 setting; the agnews_noisy setting keeps the
    stacked area because the story there is that nothing concentrates.
    """
    n_seeds, n_arms, n_episodes = per_seed_smoothed.shape
    mean = per_seed_smoothed.mean(axis=0)
    std = (per_seed_smoothed.std(axis=0, ddof=1)  # sample std
           if n_seeds > 1 else np.zeros_like(mean))
    # Override the standard arm_palette here: this figure is the only place
    # in the thesis that puts three LR-arm trajectories on a single set of
    # axes (the other arm-selection figure uses stacked areas), and the
    # light-to-dark single-hue palette becomes ambiguous when the three lines
    # are stacked at similar y. Use hue-separated colors instead so each arm
    # is unambiguously identifiable in print and at a glance. Order:
    # m=0.5 (preferred) → orange, m=1.0 (neutral) → blue, m=2.0 (suppressed)
    # → red. Falls back to arm_palette for n_arms ≠ 3 or non-aees_lr.
    if axis_spec["base_key"] == "aees_lr" and n_arms == 3:
        colors = ["#ff7f0e", "#1f77b4", "#d62728"]
    else:
        colors = arm_palette(axis_spec["base_key"], n_arms)

    # x-axis: convert episode index to epoch using the run's total_epochs,
    # so the figure shares its time unit with every other CIFAR plot.
    total_epochs = representative_record.get("total_epochs")
    if total_epochs and n_episodes >= 2:
        eps_per_epoch = n_episodes / float(total_epochs)
        x = np.arange(1, n_episodes + 1) / eps_per_epoch
        x_label = "Epoch"
        x_lim = (0.0, float(total_epochs))
    else:
        x = np.arange(1, n_episodes + 1, dtype=float)
        x_label = "Episode"
        x_lim = (1.0, float(n_episodes))

    for a_idx, (arm_val, color) in enumerate(zip(arm_values, colors)):
        lo = np.clip(mean[a_idx] - std[a_idx], 0.0, 1.0)
        hi = np.clip(mean[a_idx] + std[a_idx], 0.0, 1.0)
        ax.fill_between(x, lo, hi, color=color,
                        alpha=0.18, linewidth=0, zorder=1)
        ax.plot(
            x, mean[a_idx],
            color=color, linewidth=1.8, zorder=3,
            label=axis_spec["arm_label_fmt"](arm_val),
        )

    # Uniform-random reference line. If the controller learns anything, lines
    # diverge from this line; if it does not, lines hug it.

    ax.set_xlabel(x_label, fontsize=label_fontsize)
    ax.set_ylabel("Per-episode selection fraction", fontsize=label_fontsize)
    ax.set_xlim(*x_lim)
    ax.set_ylim(0.0, 1.0)
    if tick_fontsize is not None:
        ax.tick_params(axis="both", labelsize=tick_fontsize)
    if show_title:
        ax.set_title(axis_spec["axis_title"], fontsize=10, pad=8)
    ax.legend(
        loc="upper right", bbox_to_anchor=(1.0, 1.04),
        frameon=False, fontsize=legend_fontsize,
    )


def _draw_panel(
    ax,
    fractions_display: np.ndarray,
    n_episodes: int,
    arm_values: list[float],
    axis_spec: dict[str, Any],
    representative_record: dict[str, Any],
    is_first_panel: bool,
):
    x = np.arange(1, n_episodes + 1)
    colors = arm_palette(axis_spec["base_key"], len(arm_values))
    labels = [axis_spec["arm_label_fmt"](v) for v in arm_values]

    ax.stackplot(x, *fractions_display, colors=colors, labels=labels)
    ax.set_xlabel("Episode")
    if is_first_panel:
        ax.set_ylabel("Seed fraction")
    ax.set_xlim(1, n_episodes)
    ax.set_ylim(0.0, 1.0)
    ax.margins(x=0, y=0)
    ax.set_title(axis_spec["axis_title"], fontsize=10, pad=12)

    ax.legend(
        loc="lower center",
        ncol=len(arm_values),
        bbox_to_anchor=(0.5, -0.30),
        frameon=False,
    )

    total_epochs = representative_record["total_epochs"]
    if total_epochs >= 1 and n_episodes >= 2:
        eps_per_epoch = n_episodes / total_epochs

        def ep_to_epoch(episode_idx):
            return episode_idx / eps_per_epoch

        def epoch_to_ep(epoch):
            return epoch * eps_per_epoch

        if total_epochs <= 10:
            epoch_ticks = list(range(1, total_epochs + 1))
        else:
            step = max(10, int(round(total_epochs / 5 / 10)) * 10)
            epoch_ticks = list(range(step, total_epochs + 1, step))

        secax = ax.secondary_xaxis("top", functions=(ep_to_epoch, epoch_to_ep))
        secax.set_xticks(epoch_ticks)
        secax.set_xlim(ep_to_epoch(0), ep_to_epoch(n_episodes))
        secax.set_xlabel("Epoch")
        secax.grid(False)


# ---------------------------------------------------------------------------
# Summary text and entry point.
# ---------------------------------------------------------------------------


def _settled_callout(
    fractions: np.ndarray, arm_values: list[float], axis: str
) -> str:
    """One-line objective callout for the summary (raw proportions)."""
    n_arms, n_eps = fractions.shape
    overall = fractions.mean(axis=1)
    dominant_idx = int(np.argmax(overall))
    dominant_frac = float(overall[dominant_idx])
    dominant_val = arm_values[dominant_idx]
    settled = dominant_frac >= 0.70

    extra = ""
    q = n_eps // 4
    if q > 0:
        first_q = fractions[:, :q].mean(axis=1)
        last_q = fractions[:, -q:].mean(axis=1)
        first_high = float(first_q[-1])
        last_high = float(last_q[-1])
        first_low = float(first_q[0])
        last_low = float(last_q[0])
        delta_high = last_high - first_high
        delta_low = last_low - first_low
        if axis == "noise":
            if delta_high <= -0.15 and delta_low >= 0.15:
                extra = (
                    f" σ pattern: EARLY-HIGH / LATE-LOW "
                    f"(high-σ share {first_high:.2f} → {last_high:.2f}, "
                    f"low-σ share {first_low:.2f} → {last_low:.2f})."
                )
            elif delta_high >= 0.15 and delta_low <= -0.15:
                extra = (
                    f" σ pattern: EARLY-LOW / LATE-HIGH "
                    f"(high-σ share {first_high:.2f} → {last_high:.2f}, "
                    f"low-σ share {first_low:.2f} → {last_low:.2f})."
                )
            else:
                extra = (
                    f" σ pattern: no clear early/late skew "
                    f"(high-σ share {first_high:.2f} → {last_high:.2f}, "
                    f"low-σ share {first_low:.2f} → {last_low:.2f})."
                )
        elif axis == "lr":
            extra = (
                f" LR drift early→late: m={arm_values[-1]:g} share "
                f"{first_high:.2f}→{last_high:.2f}, m={arm_values[0]:g} share "
                f"{first_low:.2f}→{last_low:.2f}."
            )

    if settled:
        return (
            f"settled? YES — dominant arm {dominant_val:g} averages "
            f"{dominant_frac:.2f} across all episodes.{extra}"
        )
    return (
        f"settled? NO — top arm {dominant_val:g} only averages "
        f"{dominant_frac:.2f}; bands stay mixed.{extra}"
    )


def _quarter_means(
    fractions: np.ndarray, arm_values: list[float]
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Mean fraction per arm: (overall, first-25%, middle-50%, last-25%)."""
    n_eps = fractions.shape[1]
    q = max(n_eps // 4, 1)
    return (
        fractions.mean(axis=1),
        fractions[:, :q].mean(axis=1),
        fractions[:, q: 3 * q].mean(axis=1),
        fractions[:, -q:].mean(axis=1),
    )


def _run_paired(args, spec: dict[str, Any], runs_root: pathlib.Path) -> int:
    """Render a two-panel arm-selection figure comparing two component settings.

    Each panel shows per-arm trajectories (mean ± SD across seeds) for one
    component setting. Both panels share the y-axis so the cross-context
    contrast (preference cements vs preference inverts) reads directly off
    the page.
    """
    name = spec["name"]
    component_keys: list[str] = spec["component_settings"]
    panel_titles: list[str] = spec["panel_titles"]
    axis_key: str = spec["axes"][0]
    axis_spec = AXIS_SPECS[axis_key]

    try:
        # Load runs and compute per-axis data for each component.
        per_component: list[dict[str, Any]] = []
        for comp_key in component_keys:
            items = _load_runs(runs_root, comp_key)
            if not items:
                write_missing(
                    args.out_dir, name,
                    f"No flagship runs found for component '{comp_key}' "
                    f"under {runs_root.resolve()}.\n",
                )
                return 1
            items_sorted = sorted(
                items, key=lambda iv: (
                    iv[0].seed if iv[0].seed is not None else -1
                )
            )
            ctrl_key = axis_spec["controller_logs_key"]
            arm_value_sets = {
                tuple(r["controller_logs"][ctrl_key]["arm_values"])
                for _, r in items
            }
            if len(arm_value_sets) != 1:
                raise ValueError(
                    f"inconsistent {ctrl_key}.arm_values across seeds for "
                    f"component '{comp_key}': {sorted(arm_value_sets)}"
                )
            (arm_values_tuple,) = arm_value_sets
            arm_values = sorted(arm_values_tuple)

            fractions, n_episodes, seeds = _arm_fractions(
                items, axis_spec["selected_values_key"], arm_values
            )
            window = _display_window(n_episodes, args.smoothing_window)
            per_seed = _per_seed_smoothed_fractions(
                items,
                axis_spec["selected_values_key"],
                arm_values,
                n_episodes,
                window,
            )
            per_component.append({
                "comp_key": comp_key,
                "items": items,
                "items_sorted": items_sorted,
                "arm_values": arm_values,
                "fractions": fractions,
                "per_seed": per_seed,
                "n_episodes": n_episodes,
                "seeds": seeds,
                "window": window,
                "representative_record": items_sorted[0][1],
            })

        # Two-panel figure, shared y so the cross-context contrast reads
        # directly. ~3.5in per panel matches _make_multi_panel for n=2.
        fig, panels = plt.subplots(
            1, 2, sharey=True, figsize=(7.5, 4.0)
        )
        for ax, comp, title in zip(panels, per_component, panel_titles):
            _draw_per_arm_lines(
                ax,
                comp["per_seed"],
                comp["arm_values"],
                axis_spec,
                comp["representative_record"],
            )
            # Override the per-axis title set by _draw_per_arm_lines so the
            # panel headers identify the scheduler context, not just the axis.
            ax.set_title(title, fontsize=10, pad=8)

        # Drop redundant y-label from the right panel (sharey already
        # suppresses tick labels but the ylabel call inside _draw_per_arm_lines
        # sets it on both axes).
        panels[1].set_ylabel("")

        fig.tight_layout()
    except Exception as exc:
        reason = (
            f"Failed to build {name}.\n\n"
            f"Exception: {exc!r}\n\n"
            f"Traceback:\n```\n{traceback.format_exc()}```\n"
        )
        write_missing(args.out_dir, name, reason)
        return 1

    pdf_path, png_path = save_figure(fig, args.out_dir, name)

    repo_root = pathlib.Path(__file__).resolve().parents[3]

    def _rel(p: pathlib.Path) -> str:
        try:
            return str(p.resolve().relative_to(repo_root))
        except ValueError:
            return str(p.resolve())

    summary_lines: list[str] = []
    summary_lines.append(f"runs-root: {_rel(runs_root)}")
    summary_lines.append(f"setting: {spec['name']}")
    summary_lines.append(f"component_settings: {component_keys}")
    summary_lines.append("")
    for comp, title in zip(per_component, panel_titles):
        summary_lines.append(
            f"=== Panel: {title} (component={comp['comp_key']}) ===")
        summary_lines.append(f"n_seeds: {len(comp['seeds'])}")
        summary_lines.append(f"n_episodes_after_truncation: {comp['n_episodes']}")
        summary_lines.append(
            f"smoothing_window (display only): {comp['window']} "
            "(centered rolling mean)"
        )
        for info, _ in comp["items_sorted"]:
            summary_lines.append(f"  - {_rel(info.path)} (seed={info.seed})")
        overall, first_q, mid, last_q = _quarter_means(
            comp["fractions"], comp["arm_values"]
        )
        summary_lines.append("")
        summary_lines.append("Mean fraction per arm (raw, unsmoothed):")
        for a_idx, a_val in enumerate(comp["arm_values"]):
            summary_lines.append(
                f"  arm {a_val:g}: overall={overall[a_idx]:.3f}  "
                f"first25%={first_q[a_idx]:.3f}  middle50%={mid[a_idx]:.3f}  "
                f"last25%={last_q[a_idx]:.3f}  "
                f"drift(last-first)={last_q[a_idx] - first_q[a_idx]:+.3f}"
            )
        summary_lines.append("")

    summary_lines.append("outputs:")
    summary_lines.append(f"  - {pdf_path}")
    summary_lines.append(f"  - {png_path}")
    write_summary(args.out_dir, name, summary_lines)

    print(f"wrote {pdf_path}")
    print(f"wrote {png_path}")
    return 0


# LNCS-paper two-panel figure: fixed hue-separated colors, matching the order
# used elsewhere in this file for the 3-arm LR axis (m=0.5, m=1.0, m=2.0).
_PAPER_ARM_COLORS = ["#ff7f0e", "#1f77b4", "#d62728"]
# Keep the multipliers identifiable when the figure is printed without color.
_PAPER_ARM_LINESTYLES = ["-", "--", ":"]


def _run_optimizer_paired(
    args, spec: dict[str, Any], runs_root: pathlib.Path
) -> int:
    """Render the AdamW-vs-SGD+M arm-selection comparison for the LNCS paper.

    Distinct from `_run_paired` (which compares AEES-alone vs Cosine+AEES on
    a single optimizer): here the two panels are the same AEES-LR flagship on
    two different optimizers, styled to fit the paper's column width, with a
    single shared legend instead of a per-panel one. Reuses the exact same
    fraction/smoothing pipeline (`_arm_fractions`, `_per_seed_smoothed_fractions`,
    `_display_window`) as every other setting in this file, with an
    11-episode default window and paper-specific drawing code.
    """
    name = spec["name"]
    component_keys: list[str] = spec["component_settings"]
    panel_titles: list[str] = spec["panel_titles"]
    axis_key: str = spec["axes"][0]
    axis_spec = AXIS_SPECS[axis_key]

    try:
        per_component: list[dict[str, Any]] = []
        for comp_key in component_keys:
            items = _load_runs(runs_root, comp_key)
            if not items:
                write_missing(
                    args.out_dir, name,
                    f"No flagship runs found for component '{comp_key}' "
                    f"under {runs_root.resolve()}.\n",
                )
                return 1
            items_sorted = sorted(
                items, key=lambda iv: (
                    iv[0].seed if iv[0].seed is not None else -1
                )
            )
            ctrl_key = axis_spec["controller_logs_key"]
            arm_value_sets = {
                tuple(r["controller_logs"][ctrl_key]["arm_values"])
                for _, r in items
            }
            if len(arm_value_sets) != 1:
                raise ValueError(
                    f"inconsistent {ctrl_key}.arm_values across seeds for "
                    f"component '{comp_key}': {sorted(arm_value_sets)}"
                )
            (arm_values_tuple,) = arm_value_sets
            arm_values = sorted(arm_values_tuple)

            fractions, n_episodes, seeds = _arm_fractions(
                items, axis_spec["selected_values_key"], arm_values
            )
            window = _display_window(
                n_episodes,
                args.smoothing_window if args.smoothing_window is not None
                else spec.get("smoothing_window"),
            )
            per_seed = _per_seed_smoothed_fractions(
                items,
                axis_spec["selected_values_key"],
                arm_values,
                n_episodes,
                window,
            )
            per_component.append({
                "comp_key": comp_key,
                "items_sorted": items_sorted,
                "arm_values": arm_values,
                "fractions": fractions,
                "per_seed": per_seed,
                "n_episodes": n_episodes,
                "seeds": seeds,
                "window": window,
                "representative_record": items_sorted[0][1],
            })

        if len({tuple(c["arm_values"]) for c in per_component}) != 1:
            raise ValueError(
                "AdamW and SGD+M components must share identical arm_values "
                f"for a shared legend: "
                f"{[c['arm_values'] for c in per_component]}"
            )

        # No target physical size: the caller scales this PDF with
        # width=\linewidth in LaTeX. Instead aim for a wide, compact
        # composition (~2.7x wider than tall) so the plotting area stays
        # wide rather than tall/narrow, with normal-scale (not oversized)
        # type relative to the panels.
        fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.6), sharey=True)

        line_handles: list[Any] | None = None
        line_labels: list[str] | None = None
        for ax, comp, title in zip(axes, per_component, panel_titles):
            arm_values = comp["arm_values"]
            per_seed = comp["per_seed"]
            n_seeds = per_seed.shape[0]
            mean = per_seed.mean(axis=0)
            std = (per_seed.std(axis=0, ddof=1)
                   if n_seeds > 1 else np.zeros_like(mean))

            n_episodes = comp["n_episodes"]
            total_epochs = comp["representative_record"].get("total_epochs")
            if total_epochs and n_episodes >= 2:
                eps_per_epoch = n_episodes / float(total_epochs)
                x = np.arange(1, n_episodes + 1) / eps_per_epoch
            else:
                x = np.arange(1, n_episodes + 1, dtype=float)

            handles = []
            labels = []
            for a_idx, (arm_val, color) in enumerate(
                zip(arm_values, _PAPER_ARM_COLORS)
            ):
                lo = np.clip(mean[a_idx] - std[a_idx], 0.0, 1.0)
                hi = np.clip(mean[a_idx] + std[a_idx], 0.0, 1.0)
                ax.fill_between(x, lo, hi, color=color,
                                alpha=0.18, linewidth=0, zorder=1)
                label = axis_spec["arm_label_fmt"](arm_val)
                line, = ax.plot(
                    x, mean[a_idx], color=color, linewidth=1.5, zorder=3,
                    linestyle=_PAPER_ARM_LINESTYLES[a_idx],
                    label=label,
                )
                handles.append(line)
                labels.append(label)
            if line_handles is None:
                line_handles, line_labels = handles, labels

            ax.set_title(title, fontsize=13, pad=30)
            ax.set_xlabel("Epoch", fontsize=12)
            ax.set_xlim(0, 200)
            ax.set_xticks([0, 50, 100, 150, 200])
            ax.set_ylim(0.0, 1.0)
            ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
            ax.tick_params(axis="both", labelsize=11)
            ax.grid(False)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

        axes[0].set_ylabel("Arm selection frequency", fontsize=12)
        axes[1].set_ylabel("")
        axes[1].tick_params(axis="y", labelleft=False)

        fig.legend(
            line_handles, line_labels,
            loc="upper center", bbox_to_anchor=(0.5, 0.78),
            ncol=len(line_labels), frameon=False, fontsize=11,
            handlelength=2.4, columnspacing=1.2,
        )

        fig.subplots_adjust(top=0.68, bottom=0.16, left=0.07, right=0.99, wspace=0.12)
    except Exception as exc:
        reason = (
            f"Failed to build {name}.\n\n"
            f"Exception: {exc!r}\n\n"
            f"Traceback:\n```\n{traceback.format_exc()}```\n"
        )
        write_missing(args.out_dir, name, reason)
        return 1

    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = out_dir / f"{name}.pdf"
    png_path = out_dir / f"{name}.png"
    # Plain bbox_inches="tight" clips the first glyphs of the rotated
    # y-axis label on the left edge for this figure (the auto-computed tight
    # bbox under-measures rotated-text extent when a fig-level legend is
    # also present). Compute the tight bbox explicitly and pad only the
    # left side generously; other sides get a small uniform pad.
    fig.canvas.draw()
    tight_bbox = fig.get_tightbbox(fig.canvas.get_renderer())
    pad_small = 0.03
    pad_left = 0.18
    export_bbox = Bbox.from_extents(
        tight_bbox.x0 - pad_left, tight_bbox.y0 - pad_small,
        tight_bbox.x1 + pad_small, tight_bbox.y1 + pad_small,
    )
    fig.savefig(pdf_path, bbox_inches=export_bbox)
    fig.savefig(png_path, bbox_inches=export_bbox)
    plt.close(fig)

    repo_root = pathlib.Path(__file__).resolve().parents[3]

    def _rel(p: pathlib.Path) -> str:
        try:
            return str(p.resolve().relative_to(repo_root))
        except ValueError:
            return str(p.resolve())

    summary_lines: list[str] = []
    summary_lines.append(f"runs-root: {_rel(runs_root)}")
    summary_lines.append(f"setting: {spec['name']}")
    summary_lines.append(f"component_settings: {component_keys}")
    summary_lines.append("")
    for comp, title in zip(per_component, panel_titles):
        summary_lines.append(
            f"=== Panel: {title} (component={comp['comp_key']}) ===")
        summary_lines.append(f"n_seeds: {len(comp['seeds'])}")
        summary_lines.append(f"n_episodes_after_truncation: {comp['n_episodes']}")
        summary_lines.append(
            f"smoothing_window (display only): {comp['window']} "
            "(centered rolling mean)"
        )
        for info, _ in comp["items_sorted"]:
            summary_lines.append(f"  - {_rel(info.path)} (seed={info.seed})")
        overall, first_q, mid, last_q = _quarter_means(
            comp["fractions"], comp["arm_values"]
        )
        summary_lines.append("")
        summary_lines.append("Mean fraction per arm (raw, unsmoothed):")
        for a_idx, a_val in enumerate(comp["arm_values"]):
            summary_lines.append(
                f"  arm {a_val:g}: overall={overall[a_idx]:.3f}  "
                f"first25%={first_q[a_idx]:.3f}  middle50%={mid[a_idx]:.3f}  "
                f"last25%={last_q[a_idx]:.3f}  "
                f"drift(last-first)={last_q[a_idx] - first_q[a_idx]:+.3f}"
            )
        summary_lines.append("")

    summary_lines.append("outputs:")
    summary_lines.append(f"  - {pdf_path}")
    summary_lines.append(f"  - {png_path}")
    write_summary(args.out_dir, name, summary_lines)

    print(f"wrote {pdf_path}")
    print(f"wrote {png_path}")
    return 0


def _run_setting(
    args: argparse.Namespace, setting: str, runs_root: pathlib.Path
) -> int:
    spec = dict(SETTING_SPEC[setting])
    if args.smoothing_window is not None:
        spec["name"] += f"_w{args.smoothing_window}"
    name = spec["name"]
    axis_keys: list[str] = spec["axes"]

    # Paired-figure path: load each component setting independently and
    # render their per-arm-lines into a shared two-panel figure. Bypasses
    # the per-axis loop used by the other settings.
    if "component_settings" in spec:
        if spec.get("paper_style"):
            return _run_optimizer_paired(args, spec, runs_root)
        return _run_paired(args, spec, runs_root)

    try:
        items = _load_runs(runs_root, setting)
        if not items:
            write_missing(
                args.out_dir, name,
                f"No flagship runs found for setting '{setting}' under "
                f"{runs_root.resolve()}.\n",
            )
            return 1

        items_sorted = sorted(
            items, key=lambda iv: (
                iv[0].seed if iv[0].seed is not None else -1)
        )

        # Per-axis: arm values, raw fractions, smoothed display fractions.
        per_axis: dict[str, dict[str, Any]] = {}
        for axis_key in axis_keys:
            axis_spec = AXIS_SPECS[axis_key]
            ctrl_key = axis_spec["controller_logs_key"]
            arm_value_sets = {
                tuple(r["controller_logs"][ctrl_key]["arm_values"])
                for _, r in items
            }
            if len(arm_value_sets) != 1:
                raise ValueError(
                    f"inconsistent {ctrl_key}.arm_values across seeds: "
                    f"{sorted(arm_value_sets)}"
                )
            (arm_values_tuple,) = arm_value_sets
            arm_values = sorted(arm_values_tuple)

            fractions, n_episodes, seeds = _arm_fractions(
                items, axis_spec["selected_values_key"], arm_values
            )
            window = _display_window(n_episodes, args.smoothing_window)
            display_fractions = _centered_rolling_mean(fractions, window)

            per_axis[axis_key] = {
                "axis_spec": axis_spec,
                "arm_values": arm_values,
                "fractions": fractions,           # raw, for summary
                "display": display_fractions,     # smoothed, for plot
                "n_episodes": n_episodes,
                "seeds": seeds,
                "window": window,
            }

        representative_info, representative_record = items_sorted[0]
        fig, panels = _make_multi_panel(len(axis_keys))

        # Both cifar_sym40 settings (AEES alone and Cosine + AEES) use per-arm
        # trajectories with ±SD bands — the right view when the controller
        # *does* learn a preference. agnews_noisy keeps the stacked-area
        # display because its story is that the bands stay roughly equal
        # across all arms.
        use_lines = setting in (
            "cifar_sym40", "cifar_sym40_sgd", "cifar_sym40_cosine",
        )

        for ax, axis_key in zip(panels, axis_keys):
            entry = per_axis[axis_key]
            if use_lines:
                per_seed = _per_seed_smoothed_fractions(
                    items,
                    entry["axis_spec"]["selected_values_key"],
                    entry["arm_values"],
                    entry["n_episodes"],
                    entry["window"],
                )
                # The single-panel cifar_sym40 figure has no companion panel,
                # so the axis title is redundant with the caption; drop it
                # and size up the remaining text for print readability.
                single_panel = len(axis_keys) == 1
                _draw_per_arm_lines(
                    ax,
                    per_seed,
                    entry["arm_values"],
                    entry["axis_spec"],
                    representative_record,
                    show_title=not single_panel,
                    label_fontsize=14 if single_panel else None,
                    tick_fontsize=12 if single_panel else None,
                    legend_fontsize=13 if single_panel else 8,
                )
            else:
                _draw_panel(
                    ax,
                    entry["display"],
                    entry["n_episodes"],
                    entry["arm_values"],
                    entry["axis_spec"],
                    representative_record,
                    is_first_panel=(axis_key == axis_keys[0]),
                )

        if use_lines:
            fig.tight_layout()
        else:
            fig.subplots_adjust(bottom=0.24, top=0.85, wspace=0.10)
    except Exception as exc:
        reason = (
            f"Failed to build {name}.\n\n"
            f"Exception: {exc!r}\n\n"
            f"Traceback:\n```\n{traceback.format_exc()}```\n"
        )
        write_missing(args.out_dir, name, reason)
        return 1

    pdf_path, png_path = save_figure(fig, args.out_dir, name)

    repo_root = pathlib.Path(__file__).resolve().parents[3]

    def _rel(p: pathlib.Path) -> str:
        try:
            return str(p.resolve().relative_to(repo_root))
        except ValueError:
            return str(p.resolve())

    summary_lines: list[str] = []
    summary_lines.append(f"runs-root: {_rel(runs_root)}")
    summary_lines.append(f"setting: {setting}")
    summary_lines.append(f"axes plotted: {axis_keys}")
    summary_lines.append("")
    summary_lines.append("Files loaded:")
    for info, _ in items_sorted:
        summary_lines.append(f"  - {_rel(info.path)} (seed={info.seed})")
    summary_lines.append("")

    for axis_key in axis_keys:
        entry = per_axis[axis_key]
        axis_spec = entry["axis_spec"]
        arm_values = entry["arm_values"]
        fractions = entry["fractions"]

        summary_lines.append(
            f"=== Axis: {axis_key} ({axis_spec['axis_title']}) ===")
        summary_lines.append(f"n_seeds: {len(entry['seeds'])}")
        summary_lines.append(
            f"n_episodes_after_truncation: {entry['n_episodes']}")
        summary_lines.append(f"arm_values: {arm_values}")
        summary_lines.append(
            f"smoothing_window (display only): {entry['window']} "
            "(centered rolling mean)"
        )

        overall, first_q, mid, last_q = _quarter_means(fractions, arm_values)
        summary_lines.append("")
        summary_lines.append("Mean fraction per arm (raw, unsmoothed):")
        for a_idx, a_val in enumerate(arm_values):
            summary_lines.append(
                f"  arm {a_val:g}: overall={overall[a_idx]:.3f}  "
                f"first25%={first_q[a_idx]:.3f}  middle50%={mid[a_idx]:.3f}  "
                f"last25%={last_q[a_idx]:.3f}  "
                f"drift(last-first)={last_q[a_idx] - first_q[a_idx]:+.3f}"
            )

        disagreement = _seed_disagreement(
            items, axis_spec["selected_values_key"], arm_values, entry["n_episodes"]
        )
        summary_lines.append("")
        summary_lines.append(
            "Per-seed mean-fraction (and L1 distance from group mean):"
        )
        for seed, fracs_dict, l1 in disagreement:
            frac_str = " ".join(f"{k:g}:{v:.2f}" for k,
                                v in fracs_dict.items())
            summary_lines.append(f"  seed {seed}: {frac_str}  L1={l1:.3f}")

        callout = _settled_callout(fractions, arm_values, axis_key)
        summary_lines.append("")
        summary_lines.append(f"callout: {callout}")
        summary_lines.append("")

    summary_lines.append("outputs:")
    summary_lines.append(f"  - {_rel(pdf_path)}")
    summary_lines.append(f"  - {_rel(png_path)}")

    summary_path = write_summary(args.out_dir, name, summary_lines)
    print(f"wrote {pdf_path}")
    print(f"wrote {png_path}")
    print(f"wrote {summary_path}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--runs-root", type=pathlib.Path, default=None,
        help="Root directory containing the flagship runs. Defaults to the "
             "archived_results subfolder for the chosen setting.",
    )
    parser.add_argument(
        "--out-dir", type=pathlib.Path,
        default=pathlib.Path("reproduced_artifacts/figures/controllers"),
        help="Directory to write the arm_selection_<setting> outputs.",
    )
    parser.add_argument(
        "--setting", choices=sorted(SETTING_SPEC.keys()), default=None,
        help="Which setting to plot. Omit to emit all settings.",
    )
    parser.add_argument(
        "--smoothing-window", type=int, default=None,
        help="Centered moving-average window in episodes: a positive odd "
             "integer, or 1 for no smoothing. Default: 11 episodes for "
             "cifar_sym40_optimizer_paired; approximately 5%% of episodes "
             "otherwise. Adds _w<window> to output filenames.",
    )
    args = parser.parse_args(argv)
    if args.smoothing_window is not None and (
        args.smoothing_window < 1 or args.smoothing_window % 2 == 0
    ):
        parser.error("--smoothing-window must be a positive odd integer")

    settings = [args.setting] if args.setting else sorted(SETTING_SPEC.keys())
    rc = 0
    for setting in settings:
        if args.runs_root is not None:
            runs_root = args.runs_root
        else:
            runs_root = pathlib.Path("archived_results") / SETTING_RUNS_SUBDIR[setting]
        if _run_setting(args, setting, runs_root) != 0:
            rc = 1
    return rc


if __name__ == "__main__":
    sys.exit(main())
