"""Resolve released training configurations; execute only with --execute."""

import argparse
from contextlib import nullcontext
import importlib.metadata
import json
from pathlib import Path
import shlex
import subprocess
import sys
from scripts.paper.records import ROOT, DATA, manifest, donor_view, load_records

COMMON_FLAGS = "epochs batch_size lr weight_decay episode_length lr_scheduler scheduler_t_max warmup_epochs lr_candidates noise_candidates structured_control_mode context_mode context_trend_window context_trend_epsilon reward_epsilon reward_instability_lambda reward_clip_min reward_clip_max label_noise_rate seed".split()


def command(entry, output, trace_root):
    config = entry["config"]
    cifar = entry["dataset"] == "cifar100"
    cmd = [
        sys.executable,
        "-m",
        "experiments.task_cifar100" if cifar else "experiments.task_agnews",
    ]
    keys = COMMON_FLAGS + (
        "control_mode optimizer momentum label_noise_type".split()
        if cifar
        else "method label_noise_seed max_length".split()
    )
    for k in keys:
        value = config.get(k)
        if value is None:
            continue
        cmd.extend(
            [
                "--" + k.replace("_", "-"),
                ",".join(map(str, value)) if isinstance(value, list) else str(value),
            ]
        )
    if "schedule_seed" in entry:
        cmd.extend(["--schedule-seed", str(entry["schedule_seed"])])
    if entry["policy"] == "frequency_matched":
        cmd.extend(["--aees-trace-root", str(trace_root)])
    cmd.extend(["--output", str(output)])
    return cmd


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-id", action="append")
    p.add_argument("--dataset", choices=["cifar100", "agnews"])
    p.add_argument("--output-root", type=Path, default=ROOT / "results/iconip2026")
    p.add_argument(
        "--donor-data",
        type=Path,
        default=DATA,
        help="Flat released inputs used for FM donors",
    )
    p.add_argument("--execute", action="store_true")
    args = p.parse_args(argv)
    all_runs = manifest()["runs"]
    if args.run_id and set(args.run_id) - {r["id"] for r in all_runs}:
        p.error("Unknown run ID")
    selected = [
        r
        for r in all_runs
        if (not args.run_id or r["id"] in args.run_id)
        and (not args.dataset or r["dataset"] == args.dataset)
    ]
    if not selected:
        p.error("No runs selected")
    if not args.execute:
        for r in selected:
            print(
                json.dumps(
                    {
                        "run_id": r["id"],
                        "config": r["config"],
                        "schedule_seed": r.get("schedule_seed"),
                        "execute": f"uv run --frozen --extra training python -m scripts.paper.training --run-id {r['id']} --execute",
                    }
                )
            )
        return 0
    # Refuse to write training output inside released inputs or over prior runs.
    if args.output_root.resolve().is_relative_to((ROOT / "archived_results").resolve()):
        p.error("Training outputs must be outside archived_results")
    for r in selected:
        if (args.output_root / r["file"]).exists():
            p.error(f"Result already exists: {r['file']}; choose a new output root")
    needs_fm = any(r["policy"] == "frequency_matched" for r in selected)
    if needs_fm:
        load_records(args.donor_data)
    args.output_root.mkdir(parents=True, exist_ok=True)
    with donor_view(args.donor_data) if needs_fm else nullcontext(None) as traces:
        for r in selected:
            cmd = command(r, args.output_root / r["file"], traces)
            metadata = {
                "run_id": r["id"],
                "command": cmd,
                "source_result_sha256": r["sha256"],
                "code_commit": subprocess.check_output(
                    ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
                ).strip(),
                "packages": {
                    n: importlib.metadata.version(n)
                    for n in (
                        "pulseopt",
                        "torch",
                        "torchvision",
                        "transformers",
                        "datasets",
                    )
                },
            }
            (args.output_root / (r["id"] + ".provenance.json")).write_text(
                json.dumps(metadata, indent=2) + "\n"
            )
            print(shlex.join(cmd), flush=True)
            subprocess.run(cmd, cwd=ROOT, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
