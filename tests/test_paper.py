"""Numerical and provenance regressions against the pre-refactor analysis."""

import ast
import hashlib
import json
import numpy as np
import pytest
from scripts.paper.records import ROOT, DATA, load_records, manifest
from scripts.paper.aggregate import aggregate
from scripts.paper.tables import table_values
from scripts.paper.audit import audit


@pytest.fixture(scope="module")
def analysis():
    if not DATA.exists():
        pytest.skip("Run make paper-data for full released-result regression tests")
    return aggregate(load_records())


def test_nlp_function_bodies_unchanged():
    expected = json.loads((ROOT / "tests/reference/nlp_function_asts.json").read_text())
    tree = ast.parse((ROOT / "experiments/nlp_common.py").read_text())
    actual = {
        n.name: hashlib.sha256(
            ast.dump(n, include_attributes=False).encode()
        ).hexdigest()
        for n in tree.body
        if isinstance(n, ast.FunctionDef)
    }
    assert actual == expected
    assert not (ROOT / "experiments/task_sst2.py").exists()


def test_manifest_complete_and_artifacts_resolve():
    runs = manifest()["runs"]
    ids = {r["id"] for r in runs}
    artifacts = json.loads((ROOT / "paper/artifacts.json").read_text())["artifacts"]
    assert len(runs) == 205 and len(artifacts) == 11
    assert sum(a["kind"] == "figures" for a in artifacts) == 2
    assert all(set(a["run_ids"]) <= ids for a in artifacts)


def test_all_displayed_table_values_match_submitted_paper(analysis):
    for path in (ROOT / "tests/reference").glob("tab_*.values.json"):
        label = path.name.removesuffix(".values.json").replace("tab_", "tab:", 1)
        expected = json.loads(path.read_text())
        actual = table_values(analysis, label)
        exceptions = json.loads(
            (ROOT / "paper/display_discrepancies.json").read_text()
        ).get(label, {})
        assert len(actual) == len(expected)
        for i, (generated, paper) in enumerate(zip(actual, expected, strict=True)):
            if str(i) in exceptions:
                assert (generated, paper) == (
                    exceptions[str(i)]["generated"],
                    exceptions[str(i)]["paper"],
                )
            else:
                assert generated == paper, (label, i, generated, paper)


def test_effects_and_curves_match_original_scripts(analysis):
    legacy = json.loads((ROOT / "tests/reference/legacy_analysis.json").read_text())
    for key, effects in legacy["effects"].items():
        cell = analysis["cells"]["cifar100/" + key]
        for method, values in effects.items():
            stats = cell["comparisons"]["aees_minus_" + method]["metrics"]["peak"]
            for field in ("diff_mean", "diff_ci_low", "diff_ci_high"):
                assert stats[field] == pytest.approx(values[field], abs=1e-12), (
                    key,
                    method,
                    field,
                )
        ur = cell["comparisons"]["ur_minus_best_fixed"]["metrics"]["peak"]
        for field, value in legacy["ur_fixed"][key].items():
            if field != "p_value":
                assert ur[field] == pytest.approx(value, abs=1e-12)
    for opt, curves in legacy["arm_plot"].items():
        for field in ("mean", "sd"):
            np.testing.assert_allclose(
                analysis["arm_plot"][opt][field], curves[field], rtol=0, atol=1e-14
            )
    keys = {
        "fixed0": "adamw_wl",
        "fixed005": "fixed_005",
        "fixed01": "fixed_01",
        "uniform_random": "random_sigma",
        "aees": "aees_noise_wl",
    }
    for policy, key in keys.items():
        current = analysis["cells"]["agnews/sym20/AdamW"]["policies"][policy]
        for metric, old in [
            ("peak", "best"),
            ("final", "final"),
            ("drop", "drop"),
            ("peak_epoch", "peak_epoch"),
        ]:
            assert current[metric]["mean"] == pytest.approx(
                legacy["agnews"][key][old + "_mean"], abs=1e-12
            )
            assert current[metric]["sd"] == pytest.approx(
                legacy["agnews"][key][old + "_std"], abs=1e-12
            )


def test_full_control_audit():
    if not DATA.exists():
        pytest.skip("Run make paper-data for full control audit")
    report = audit(load_records(), DATA)
    assert report["controls_checked"] == 60 and report["global_rng_unchanged"]


def test_corrupted_archive_is_rejected(tmp_path):
    from scripts.paper.archive import unpack

    path = tmp_path / "bad.zip"
    path.write_bytes(b"invalid")
    with pytest.raises(ValueError, match="checksum"):
        unpack(path, tmp_path / "results", "0" * 64)


def test_foreign_zip_entries_are_rejected(tmp_path):
    import zipfile
    from scripts.paper.archive import unpack, sha256

    path = tmp_path / "bad.zip"
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("../escaped.json", "{}")
    with pytest.raises(ValueError, match="exactly"):
        unpack(path, tmp_path / "results", sha256(path))
    assert not (tmp_path / "escaped.json").exists()
