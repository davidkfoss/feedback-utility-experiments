"""Populate the submitted paper's table layouts from computed values."""

import re
from scripts.paper.records import ROOT


def table_values(analysis, label):
    cells = analysis["cells"]
    values = []

    def add(value, digits=2):
        values.append(f"{value:.{digits}f}")

    def pm(stats, digits=2):
        add(stats["mean"], digits)
        add(stats["sd"], digits)

    def paired(stats):
        for k in ("diff_mean", "diff_ci_low", "diff_ci_high"):
            add(stats[k])
        add(stats["wins"], 0)
        add(stats["n"], 0)

    if label in ("tab:agnews-noise", "tab:appendix-agnews-controls"):
        cell = cells["agnews/sym20/AdamW"]
        order = (
            ["fixed0", "aees", cell["best_fixed"], "uniform_random"]
            if label == "tab:agnews-noise"
            else ["fixed0", "fixed005", "fixed01", "uniform_random", "aees"]
        )
        for policy in order:
            if policy in ("fixed0", "fixed005", "fixed01"):
                values.append(
                    {"fixed0": "0", "fixed005": "0.005", "fixed01": "0.01"}[policy]
                )
            for metric in ("peak", "final", "drop", "peak_epoch"):
                pm(cell["policies"][policy][metric])
        return values
    for regime in ("asym20", "sym20", "sym40"):
        for opt in ("AdamW", "SGD"):
            cell = cells[f"cifar100/{regime}/{opt}"]
            policies = cell["policies"]
            if label in ("tab:cifar-fixed", "tab:cifar-ur-fixed"):
                best = cell["best_fixed"]
                add({"fixed05": 0.5, "fixed10": 1.0, "fixed20": 2.0}[best], 1)
                pm(policies[best]["peak"])
                method, comparison = (
                    ("aees", "aees_minus_best_fixed")
                    if label == "tab:cifar-fixed"
                    else ("uniform_random", "ur_minus_best_fixed")
                )
                pm(policies[method]["peak"])
                paired(cell["comparisons"][comparison]["metrics"]["peak"])
            elif label == "tab:cifar-open-loop":
                pm(policies["aees"]["peak"], 1)
                for policy in ("uniform_random", "frequency_matched"):
                    pm(policies[policy]["peak"], 1)
                    paired(
                        cell["comparisons"]["aees_minus_" + policy]["metrics"]["peak"]
                    )
            elif label == "tab:appendix-all-fixed":
                for policy in ("fixed05", "fixed10", "fixed20", "aees"):
                    pm(policies[policy]["peak"])
            elif label == "tab:appendix-cifar-controller":
                for key in ("shares", "median_rewards"):
                    for value in cell["controller"][key]:
                        add(value, 3)
            elif label == "tab:appendix-fm-ranges":
                for lo, hi in cell["fm_count_ranges"]:
                    add(lo, 0)
                    if lo != hi:
                        add(hi, 0)
            else:
                raise ValueError(f"Unknown table: {label}")
    return values


def render(analysis, label):
    name = (
        "control_questions.tex"
        if label == "tab:audit-controls"
        else label.replace(":", "_") + ".tex"
    )
    template = (ROOT / "paper/templates" / name).read_text()
    if label == "tab:audit-controls":
        return template
    values = table_values(analysis, label)
    slots = re.findall(r"@@v\d{3}@@", template)
    if len(slots) != len(values):
        raise ValueError(f"{label}: expected {len(slots)} cells, got {len(values)}")
    for slot, value in zip(slots, values, strict=True):
        template = template.replace(slot, value)
    return template
