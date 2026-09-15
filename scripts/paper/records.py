"""Explicit, checksum-verified paper inputs; never discover runs by filename heuristics."""

from __future__ import annotations
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import tempfile

ROOT = Path(__file__).resolve().parents[2]
DATA = Path(os.environ.get("ICONIP_DATA", ROOT / "archived_results/iconip2026"))
OUTPUT = ROOT / "reproduced_artifacts/iconip2026"
MANIFEST = ROOT / "paper/experiments.json"


def manifest():
    value = json.loads(MANIFEST.read_text())
    runs = value["runs"]
    keys = [
        (r["dataset"], r["regime"], r["optimizer"], r["policy"], r["seed"])
        for r in runs
    ]
    for field in ("id", "file"):
        if len({r[field] for r in runs}) != len(runs):
            raise ValueError(f"Duplicate manifest {field}")
    if len(set(keys)) != len(keys):
        raise ValueError("Duplicate logical run")
    expected = {
        ("cifar100", regime, opt, policy, s)
        for regime in ("asym20", "sym20", "sym40")
        for opt in ("AdamW", "SGD")
        for policy in (
            "aees",
            "fixed05",
            "fixed10",
            "fixed20",
            "uniform_random",
            "frequency_matched",
        )
        for s in range(5)
    }
    expected |= {
        ("agnews", "sym20", "AdamW", policy, s)
        for policy in ("aees", "fixed0", "fixed005", "fixed01", "uniform_random")
        for s in range(5)
    }
    if set(keys) != expected or value["seeds"] != list(range(5)):
        raise ValueError("Manifest does not contain the complete paired paper matrix")
    for r in runs:
        if Path(r["file"]).name != r["file"] or not r["file"].endswith(".json"):
            raise ValueError(f"Unsafe or non-flat result filename: {r['file']}")
    return value


def load_records(data=DATA):
    data = Path(data)
    spec = manifest()
    expected = {r["file"] for r in spec["runs"]}
    actual = {p.name for p in data.iterdir()} if data.exists() else set()
    if actual != expected:
        raise ValueError(
            f"Results inventory mismatch: missing={sorted(expected - actual)}, unexpected={sorted(actual - expected)}"
        )
    records = []
    for entry in spec["runs"]:
        raw = (data / entry["file"]).read_bytes()
        if (
            len(raw) != entry["size_bytes"]
            or hashlib.sha256(raw).hexdigest() != entry["sha256"]
        ):
            raise ValueError(f"Checksum mismatch: {entry['file']}")
        payload = json.loads(raw)
        config = payload["config"]
        if payload["seed"] != entry["seed"] or any(
            config.get(k) != v for k, v in entry["config"].items()
        ):
            raise ValueError(f"Configuration mismatch: {entry['id']}")
        vals = payload["val_accuracies"]
        if (
            len(vals) != config["epochs"]
            or payload["best_val_accuracy"] != max(vals)
            or payload["final_val_accuracy"] != vals[-1]
        ):
            raise ValueError(
                f"Incomplete or inconsistent validation metrics: {entry['id']}"
            )
        records.append((entry, payload))
    return records


@contextmanager
def donor_view(data=DATA):
    """Adapt flat release paths to the unchanged training trace validator."""
    from experiments.utils.cifar_attribution_controls import AEES_ARTIFACT_NAMES

    with tempfile.TemporaryDirectory(prefix="iconip-donors-") as tmp:
        root = Path(tmp)
        for r in manifest()["runs"]:
            if r["dataset"] == "cifar100" and r["policy"] == "aees":
                target = (
                    root
                    / f"cifar100_{r['regime']}_seed{r['seed']}"
                    / AEES_ARTIFACT_NAMES[r["optimizer"]]
                )
                target.parent.mkdir(parents=True, exist_ok=True)
                target.symlink_to((Path(data) / r["file"]).resolve())
        yield root
