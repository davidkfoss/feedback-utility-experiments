"""Regenerate registered paper artifacts from released results, without training."""

import argparse
import csv
import hashlib
import importlib.metadata
import json
import platform
import subprocess
from pathlib import Path
from scripts.paper.records import ROOT, DATA, OUTPUT, MANIFEST, load_records
from scripts.paper.aggregate import aggregate
from scripts.paper.audit import audit
from scripts.paper.tables import render


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, default=DATA)
    p.add_argument("--out", type=Path, default=OUTPUT)
    p.add_argument("--artifact", action="append")
    p.add_argument(
        "--kind",
        choices=["all", "tables", "figures", "analysis", "audit"],
        default="all",
    )
    args = p.parse_args(argv)
    registry = json.loads((ROOT / "paper/artifacts.json").read_text())["artifacts"]
    if args.artifact:
        unknown = set(args.artifact) - {r["label"] for r in registry}
        if unknown:
            p.error(f"Unknown artifact labels: {sorted(unknown)}")
    records = load_records(args.data)
    analysis = aggregate(records)
    args.out.mkdir(parents=True, exist_ok=True)
    write_json(args.out / "analysis.json", analysis)
    with (args.out / "seed_outcomes.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(analysis["seed_outcomes"][0]))
        writer.writeheader()
        writer.writerows(analysis["seed_outcomes"])
    if args.kind in ("all", "audit"):
        write_json(args.out / "control_audit.json", audit(records, args.data))
    selected = [
        r
        for r in registry
        if (not args.artifact or r["label"] in args.artifact)
        and (args.kind == "all" or args.kind == r["kind"])
    ]
    for r in selected:
        path = args.out / r["output"]
        path.parent.mkdir(parents=True, exist_ok=True)
        if r["kind"] == "tables":
            path.write_text(render(analysis, r["label"]))
        else:
            from scripts.paper.plots import save

            save(analysis, r["label"], path)
        print(f"Generated {r['label']}: {r['output']}")
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        commit = None
    provenance = {
        "manifest_sha256": hashlib.sha256(MANIFEST.read_bytes()).hexdigest(),
        "code_commit": commit,
        "python": platform.python_version(),
        "packages": {
            n: importlib.metadata.version(n) for n in ("numpy", "scipy", "matplotlib")
        },
        "artifacts": [r["label"] for r in selected],
        "run_count": len(records),
        "source_code_sha256": {
            str(f.relative_to(ROOT)): hashlib.sha256(f.read_bytes()).hexdigest()
            for f in sorted((ROOT / "scripts").rglob("*.py"))
        },
    }
    write_json(args.out / "generation.json", provenance)
    print(f"Computed full-precision analysis from {len(records)} runs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
