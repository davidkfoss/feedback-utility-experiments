"""Full-precision seed outcomes, paired effects and controller diagnostics."""

from dataclasses import asdict
from statistics import mean, stdev, median
from collections import defaultdict
import numpy as np
from scripts.tables._paired_stats import compare_paired

METRICS = ("peak", "final", "drop", "peak_epoch")


def outcome(payload):
    vals = payload["val_accuracies"]
    peak, final = payload["best_val_accuracy"], payload["final_val_accuracy"]
    return dict(
        peak=peak,
        final=final,
        drop=peak - final,
        peak_epoch=max(range(len(vals)), key=lambda i: vals[i]) + 1,
    )


def diagnostic(payloads, axis, arms):
    actions = [
        v for p in payloads for v in p["episode_logs"][f"selected_{axis}_values"]
    ]
    rewards = [v for p in payloads for v in p["episode_logs"]["episode_rewards"]]
    if len(actions) != len(rewards):
        raise ValueError("Action/reward lengths differ")
    return {
        "arms": arms,
        "episode_count": len(actions),
        "shares": [actions.count(a) / len(actions) for a in arms],
        "median_rewards": [
            median(r for v, r in zip(actions, rewards, strict=True) if v == a)
            for a in arms
        ],
    }


def arm_curves(payloads, window=11):
    if window < 1 or window % 2 != 1:
        raise ValueError("Smoothing window must be a positive odd integer")
    values = np.asarray(
        [p["episode_logs"]["selected_lr_values"] for p in payloads], dtype=float
    )
    if values.shape != (5, 391):
        raise ValueError(f"Expected five complete 391-episode traces: {values.shape}")
    indicators = np.stack(
        [np.isclose(values, a, atol=1e-9) for a in (0.5, 1, 2)], axis=1
    ).astype(float)
    smoothed = np.empty_like(indicators)
    for i in range(391):
        smoothed[:, :, i] = indicators[
            :, :, max(0, i - window // 2) : min(391, i + window // 2 + 1)
        ].mean(axis=2)
    return {
        "window": window,
        "mean": smoothed.mean(axis=0).tolist(),
        "sd": smoothed.std(axis=0, ddof=1).tolist(),
        "epochs": (np.arange(1, 392) / (391 / 200)).tolist(),
    }


def aggregate(records):
    groups = defaultdict(dict)
    for r, p in records:
        key = "/".join(r[k] for k in ("dataset", "regime", "optimizer"))
        groups[key].setdefault(r["policy"], {})[r["seed"]] = (r, p)
    result = {
        "schema_version": 1,
        "seeds": list(range(5)),
        "statistics": {
            "bootstrap_resamples": 10000,
            "bootstrap_seed": 0,
            "interval": "paired percentile 2.5/97.5",
            "sd": "sample (ddof=1)",
            "best_fixed": "selected once by mean peak; held fixed in bootstrap",
        },
        "cells": {},
        "arm_plot": {},
        "seed_outcomes": [],
    }
    for key, policies in groups.items():
        summaries = {}
        outcomes = {}
        for policy, by_seed in policies.items():
            if sorted(by_seed) != list(range(5)):
                raise ValueError(f"Incomplete seed pairing: {key}/{policy}")
            rows = [outcome(by_seed[s][1]) for s in range(5)]
            outcomes[policy] = rows
            summaries[policy] = {}
            for metric in METRICS:
                scale = 1 if metric == "peak_epoch" else 100
                vals = [r[metric] for r in rows]
                summaries[policy][metric] = dict(
                    mean=mean(vals) * scale, sd=stdev(vals) * scale
                )
            for s, row in enumerate(rows):
                result["seed_outcomes"].append(
                    {"id": by_seed[s][0]["id"], "seed": s, **row}
                )
        candidates = (
            ["fixed05", "fixed10", "fixed20"]
            if key.startswith("cifar100")
            else ["fixed0", "fixed005", "fixed01"]
        )
        best = max(candidates, key=lambda p: summaries[p]["peak"]["mean"])
        comparisons = {}
        pairs = [
            ("aees", best, "aees_minus_best_fixed"),
            ("uniform_random", best, "ur_minus_best_fixed"),
            ("aees", "uniform_random", "aees_minus_uniform_random"),
        ]
        if "frequency_matched" in policies:
            pairs.append(("aees", "frequency_matched", "aees_minus_frequency_matched"))
        for a, b, name in pairs:
            comparisons[name] = {"method": a, "control": b, "metrics": {}}
            for metric in METRICS:
                left = [r[metric] for r in outcomes[a]]
                right = [r[metric] for r in outcomes[b]]
                stats = asdict(
                    compare_paired(
                        left, right, scale=1 if metric == "peak_epoch" else 100
                    )
                )
                # p-values are not reported in this paper; avoid non-finite JSON for constant vectors.
                stats.pop("p_value")
                stats["wins"] = sum(x > y for x, y in zip(left, right, strict=True))
                stats["paired_differences"] = [
                    x - y for x, y in zip(left, right, strict=True)
                ]
                comparisons[name]["metrics"][metric] = stats
        payloads = [policies["aees"][s][1] for s in range(5)]
        is_cifar = key.startswith("cifar100")
        cell = {
            "best_fixed": best,
            "policies": summaries,
            "comparisons": comparisons,
            "controller": diagnostic(
                payloads,
                "lr" if is_cifar else "noise",
                [0.5, 1, 2] if is_cifar else [0, 0.005, 0.01],
            ),
        }
        if is_cifar:
            counts = [
                policies["frequency_matched"][s][1]["config"]["open_loop_policy"][
                    "integer_counts"
                ]
                for s in range(5)
            ]
            cell["fm_count_ranges"] = [
                [min(c[str(a)] for c in counts), max(c[str(a)] for c in counts)]
                for a in (0.5, 1.0, 2.0)
            ]
            if "/sym40/" in key:
                result["arm_plot"][key.split("/")[-1]] = arm_curves(payloads)
        result["cells"][key] = cell
    return result
