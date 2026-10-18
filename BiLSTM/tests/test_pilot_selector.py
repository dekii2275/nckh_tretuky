"""Pilot sampling and persistence tests using synthetic data only."""

import csv
import io
import json
import sys
from collections import Counter
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import pilot_selector as server  # noqa: E402
from pilot_catalog import (  # noqa: E402
    REPO,
    digest_json,
    load_catalog,
    max_missing_run,
    measure_raw,
    pipeline_hashes,
    quality_band,
)
from pilot_sampling import propose, selection_summary, validate_draft  # noqa: E402


@pytest.fixture
def catalog():
    rows = []
    for subject in range(8):
        for activity in range(10):
            for band, missing in [
                ("low_missing", 0.02),
                ("medium_missing", 0.10),
                ("high_missing", 0.30),
            ]:
                rows.append(
                    {
                        "clip_id": f"a{activity}_s{subject}_D1_{band}",
                        "subject_id": f"s{subject}",
                        "session_id": f"D{subject % 3 + 1}",
                        "activity": f"activity_{activity}",
                        "activity_id": str(activity),
                        "num_frames": 100,
                        "status": "ready",
                        "missingness_band": band,
                        "missing_joint_ratio": missing,
                        "max_missing_run": 3,
                        "source_gap_count": 0,
                        "mean_confidence": 0.8,
                        "source_sha256": "a" * 64,
                    }
                )
    result = {
        "schema_version": "mmasd_pilot_catalog_v1",
        "config_path": "BiLSTM/configs/skeleton_audit_v0_2.yaml",
        "config_sha256": "b" * 64,
        "pipeline_sha256": {},
        "rows": rows,
        "complete_population": True,
        "measured_size": len(rows),
        "population_size": len(rows),
    }
    result["catalog_sha256"] = digest_json(result)
    return result


def test_proposal_is_reproducible_balanced_and_never_claims_manual_review(catalog):
    before = deepcopy(catalog)
    first, second = propose(catalog, 40, 42), propose(catalog, 40, 42)
    assert first == second and catalog == before
    info = selection_summary(first, catalog)
    assert info["selected"] == 40 and info["groups"] == {"tuning": 20, "evaluation": 20}
    assert not info["subject_overlap"] and info["reviewed"] == 0 and not info["can_finalize"]
    assert set(info["missingness_bands"]) == {"low_missing", "medium_missing", "high_missing"}
    assert all(len(activities) == 10 for activities in info["activity_coverage"].values())
    assert all(
        count == 2 for counts in info["activity_coverage"].values() for count in counts.values()
    )


@pytest.mark.parametrize("target", [20, 21, 30, 49, 50])
def test_proposal_honors_target_without_duplicates_or_group_leakage(catalog, target):
    draft = propose(catalog, target, 7)
    info = selection_summary(draft, catalog)
    assert len(draft["selected"]) == target
    assert len({item["clip_id"] for item in draft["selected"]}) == target
    assert not info["subject_overlap"]


def test_exclusions_are_preserved_and_shortage_is_not_hidden(catalog):
    first = propose(catalog)
    removed = [{"clip_id": item["clip_id"], "reason": "Not suitable"} for item in first["selected"]]
    second = propose(catalog, excluded=removed)
    assert not {item["clip_id"] for item in second["selected"]} & {
        item["clip_id"] for item in removed
    }
    assert second["excluded"] == removed
    reduced = deepcopy(catalog)
    reduced["rows"] = [row for row in catalog["rows"] if row["activity"] == "activity_0"][:6]
    for index, row in enumerate(reduced["rows"]):
        row["subject_id"] = f"s{index % 2}"
    draft = propose(reduced, 40)
    assert len(draft["selected"]) < 40
    assert not selection_summary(draft, reduced)["can_finalize"]


def test_review_reasons_and_disjoint_subjects_are_required_to_finalize(catalog):
    draft = propose(catalog)
    for item in draft["selected"]:
        item["reviewed"] = True
    assert selection_summary(draft, catalog)["can_finalize"]
    draft["selected"][0]["reason"] = "  "
    assert not selection_summary(draft, catalog)["can_finalize"]
    draft["selected"][0]["reason"] = "Valid reason"
    lookup = {row["clip_id"]: row for row in catalog["rows"]}
    subject = lookup[draft["selected"][0]["clip_id"]]["subject_id"]
    same_subject = [
        item for item in draft["selected"] if lookup[item["clip_id"]]["subject_id"] == subject
    ]
    assert len(same_subject) > 1
    same_subject[0]["group"] = "evaluation" if same_subject[1]["group"] == "tuning" else "tuning"
    assert subject in selection_summary(draft, catalog)["subject_overlap"]


@pytest.mark.parametrize(
    "change", ["unknown", "duplicate", "catalog", "group", "review_type", "quarantined"]
)
def test_malformed_imports_are_rejected(catalog, change):
    draft = propose(catalog)
    if change == "unknown":
        draft["selected"][0]["clip_id"] = "not-in-catalogue"
    elif change == "duplicate":
        draft["selected"].append(deepcopy(draft["selected"][0]))
    elif change == "catalog":
        draft["catalog_sha256"] = "different"
    elif change == "group":
        draft["selected"][0]["group"] = "test"
    elif change == "review_type":
        draft["selected"][0]["reviewed"] = "true"
    else:
        next(row for row in catalog["rows"] if row["clip_id"] == draft["selected"][0]["clip_id"])[
            "status"
        ] = "quarantined"
    with pytest.raises(ValueError):
        validate_draft(draft, catalog)


def test_missing_runs_include_edges_but_do_not_bridge_absent_source_files():
    valid = np.array([[False, True], [False, False], [False, True], [True, False]])
    assert max_missing_run(valid, [10, 11, 12, 13]) == 3
    assert max_missing_run(valid, [10, 11, 20, 21]) == 2
    assert quality_band(0.05) == "low_missing" and quality_band(0.20) == "medium_missing"
    assert quality_band(0.21) == "high_missing"


def test_metrics_keep_absence_unknown_and_ignore_motion_across_source_gaps():
    raw = np.zeros((3, 25, 3), dtype=float)
    raw[:, :, :2] = np.arange(25)[:, None] + [10, 20]
    raw[:, :, 2] = 0.8
    raw[1, :, :2] += 10000
    valid = np.ones((3, 25), dtype=bool)
    result = measure_raw(raw, valid, [0, 10, 20], [1, 2, 1], 0.1)
    assert result["source_gap_count"] == 18 and result["large_step_joint_events"] == 0
    raw[:] = np.nan
    result = measure_raw(raw, np.zeros_like(valid), [0, 1, 2], [0, 0, 0], 0.1)
    assert result["mean_confidence"] is None and result["low_confidence_ratio"] is None
    assert result["target_missing_frames"] == 3 and result["missing_joint_ratio"] == 1


def test_catalog_checksum_detects_metric_changes(tmp_path, catalog):
    path = tmp_path / "catalog.json"
    path.write_text(json.dumps(catalog))
    assert load_catalog(path)["catalog_sha256"] == catalog["catalog_sha256"]
    catalog["rows"][0]["missing_joint_ratio"] = 0.99
    path.write_text(json.dumps(catalog))
    with pytest.raises(ValueError, match="checksum"):
        load_catalog(path)


def test_pipeline_hashes_cover_inherited_config():
    path = REPO / "BiLSTM/configs/improved_v7_focal_loss.yaml"
    hashes = pipeline_hashes(path)
    assert all(
        "BiLSTM/configs/" + name in hashes
        for name in [
            "paper_repro.yaml",
            "paper_repro_tracker_v2.yaml",
            "improved_v7_focal_loss.yaml",
        ]
    )


@pytest.fixture
def store(tmp_path, catalog):
    (tmp_path / "catalog.json").write_text(json.dumps(catalog))
    return server.PilotStore(tmp_path, check_pipeline=False)


def test_drafts_survive_restart_and_stale_revision_cannot_overwrite(store):
    state = store.state()
    state["draft"]["selected"][0]["reason"] = 'a quote " and a newline\nkept'
    saved = store.save(state["draft"], state["revision"])
    restarted = server.PilotStore(store.directory, check_pipeline=False)
    assert restarted.state()["draft"] == saved["draft"]
    with pytest.raises(server.ConflictError):
        store.save(state["draft"], state["revision"])
    assert store.state()["draft"] == saved["draft"]


def test_bad_import_does_not_replace_saved_draft(store):
    before = store.draft_path.read_bytes()
    state = store.state()
    state["draft"]["catalog_sha256"] = "wrong"
    with pytest.raises(ValueError):
        store.save(state["draft"], state["revision"])
    assert store.draft_path.read_bytes() == before


@pytest.mark.parametrize(
    "value", [None, [], "not a draft", {"selected": None}, {"selected": [], "excluded": None}]
)
def test_import_rejects_wrong_json_shapes(catalog, value):
    with pytest.raises(ValueError):
        validate_draft(value, catalog)


def test_final_export_rechecks_source_and_writes_two_disjoint_manifests(store, monkeypatch):
    state = store.state()
    with pytest.raises(ValueError, match="Chưa thể chốt"):
        store.finalize(state["revision"])
    for item in state["draft"]["selected"]:
        item["reviewed"] = True
    state = store.save(state["draft"], state["revision"])
    checked = []
    monkeypatch.setattr(store, "observation", lambda clip_id: checked.append(clip_id))
    exported = store.finalize(state["revision"])
    assert len(checked) == 40
    directory = store.state_dir / "exports" / exported["export_id"]
    with (directory / "manifest.csv").open(encoding="utf-8-sig", newline="") as file:
        rows = list(csv.DictReader(file))
    assert len(rows) == 40 and Counter(row["group"] for row in rows) == {
        "tuning": 20,
        "evaluation": 20,
    }
    tuning = {row["subject_id"] for row in rows if row["group"] == "tuning"}
    evaluation = {row["subject_id"] for row in rows if row["group"] == "evaluation"}
    assert not tuning & evaluation
    payload = json.loads((directory / "selection.json").read_text())
    assert payload["config"]["tracking"]["mode"] == "tracklet_v2"
    assert payload["manifest"][0]["source_sha256"] == "a" * 64
    assert all((directory / name).exists() for name in ["tuning.csv", "evaluation.csv"])
    assert store.finalize(state["revision"])["export_id"] != exported["export_id"]


def test_export_refuses_changed_source(store, monkeypatch):
    state = store.state()
    for item in state["draft"]["selected"]:
        item["reviewed"] = True
    state = store.save(state["draft"], state["revision"])

    def changed_source(_):
        raise server.ConflictError("source changed")

    monkeypatch.setattr(store, "observation", changed_source)
    with pytest.raises(server.ConflictError):
        store.finalize(state["revision"])
    assert not (store.state_dir / "exports").exists()


def test_csv_preserves_quotes_and_neutralizes_spreadsheet_formulas():
    rows = [
        {"clip_id": "clip", "reason": 'quote "hello"\nnext line'},
        {"clip_id": "another", "reason": "=1+1"},
    ]
    exported = server.csv_text(rows)
    parsed = list(csv.DictReader(io.StringIO(exported.lstrip("\ufeff"))))
    assert parsed[0] == rows[0] and parsed[1]["reason"] == "'=1+1"
