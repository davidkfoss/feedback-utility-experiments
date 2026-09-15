"""Check all resolved training configurations without loading data or training."""

import dataclasses
import sys
from pathlib import Path
import pytest

torch = pytest.importorskip("torch", reason="Optional training environment required")
pytest.importorskip("pulseopt")
from scripts.paper.records import manifest
from scripts.paper.training import command
from experiments import task_agnews, task_cifar100


def test_all_paper_commands_preserve_recorded_training_config(monkeypatch):
    for entry in manifest()["runs"]:
        cmd = command(
            entry, Path("/tmp/unused-result.json"), Path("/tmp/unused-donors")
        )
        monkeypatch.setattr(sys, "argv", [cmd[2], *cmd[3:]])
        module = task_cifar100 if entry["dataset"] == "cifar100" else task_agnews
        actual = dataclasses.asdict(module.parse_args())
        for key, value in entry["config"].items():
            if key in actual:
                if (
                    entry["config"].get("control_mode") == "baseline"
                    and key in ("lr_candidates", "noise_candidates")
                    and value is None
                ):
                    assert actual[key] == ([1.0] if key == "lr_candidates" else [0.0])
                    continue
                assert actual[key] == value, (entry["id"], key, actual[key], value)
        if "schedule_seed" in entry:
            assert actual["schedule_seed"] == entry["schedule_seed"]


def test_tiny_nlp_updates_and_partial_episode_match_original(monkeypatch):
    import json
    from experiments import nlp_common
    from scripts.paper.records import ROOT
    from tests.nlp_toy import run_toy

    expected = json.loads((ROOT / "tests/reference/nlp_toy.json").read_text())

    def equal(actual, reference):
        if isinstance(reference, dict):
            assert actual.keys() == reference.keys()
            for k in reference:
                equal(actual[k], reference[k])
        elif isinstance(reference, (list, tuple)):
            assert len(actual) == len(reference)
            for a, b in zip(actual, reference, strict=True):
                equal(a, b)
        elif isinstance(reference, float):
            assert actual == pytest.approx(reference, rel=1e-6, abs=1e-8)
        else:
            assert actual == reference

    for key, reference in expected.items():
        method, noise = key.split(":")
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "check",
                "--method",
                method,
                "--noise-candidates",
                noise,
                "--lr-candidates",
                "1",
                "--lr-scheduler",
                "warmup_linear",
            ],
        )
        equal(run_toy(nlp_common), reference)
