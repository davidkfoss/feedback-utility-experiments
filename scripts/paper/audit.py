"""Reconstruct released schedules through the original control implementation."""

import random
from scripts.paper.records import donor_view
from experiments.utils.cifar_attribution_controls import (
    build_frequency_matched_plan,
    build_uniform_random_plan,
    load_validated_aees_trace,
)


def audit(records, data):
    state = random.getstate()
    checked = []
    with donor_view(data) as root:
        for entry, p in records:
            if entry["dataset"] != "cifar100":
                continue
            if entry["policy"] == "aees":
                load_validated_aees_trace(
                    trace_root=root,
                    noise_regime=entry["regime"],
                    optimizer=entry["optimizer"],
                    seed=entry["seed"],
                )
            if entry["policy"] not in ("frequency_matched", "uniform_random"):
                continue
            meta = p["config"]["open_loop_policy"]
            logs = p["episode_logs"]
            kwargs = dict(episode_count=391, schedule_seed=meta["schedule_seed"])
            if entry["policy"] == "frequency_matched":
                plan = build_frequency_matched_plan(
                    trace_root=root,
                    noise_regime=entry["regime"],
                    optimizer=entry["optimizer"],
                    target_seed=entry["seed"],
                    **kwargs,
                )
            else:
                plan = build_uniform_random_plan(**kwargs)
            expected = plan.to_metadata()
            for field in (
                "source_seeds",
                "probabilities",
                "integer_counts",
                "planned_lr_indices",
                "planned_lr_multipliers",
            ):
                if expected[field] != meta[field]:
                    raise ValueError(
                        f"{entry['id']}: {field} differs from reconstructed schedule"
                    )
            values = list(plan.lr_values)
            if (
                values != meta["executed_lr_multipliers"]
                or values != logs["selected_lr_values"]
            ):
                raise ValueError(
                    f"{entry['id']}: planned/executed/logged sequences differ"
                )
            if (
                list(plan.lr_indices) != logs["selected_lr_indices"]
                or p["total_steps"] != 78200
            ):
                raise ValueError(
                    f"{entry['id']}: action indices or training horizon differ"
                )
            if logs["episode_start_steps"] != list(range(0, 78200, 200)) or logs[
                "episode_end_steps"
            ] != list(range(199, 78200, 200)):
                raise ValueError(f"{entry['id']}: episode boundaries differ")
            checked.append(entry["id"])
    if state != random.getstate():
        raise ValueError("Schedule construction changed global Python RNG")
    if len(checked) != 60:
        raise ValueError("Expected 60 control runs")
    return {
        "passed": True,
        "controls_checked": len(checked),
        "fm_runs": 30,
        "aees_donors_checked": 30,
        "global_rng_unchanged": True,
        "run_ids": checked,
    }
