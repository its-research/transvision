from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from tools.event_track_v2x.audit_v2v4real_overlap import main
from tools.event_track_v2x.prepare_v2v4real_inputs import prepare_native_inputs
from transvision.models.event_track_v2x.v2v4real_inputs import V2V4RealInputError
from transvision.models.event_track_v2x.v2v4real_overlap import (
    SESSION_MAP_KIND, audit_native_overlap, validate_development_folds,
)


def _prepare_split(tmp_path, split, specs):
    source = tmp_path / f"raw-{split}"
    ego = {}
    for sequence, (_, salt) in specs.items():
        ego[sequence] = "0"
        for cav in ("0", "1"):
            folder = source / sequence / cav
            folder.mkdir(parents=True)
            for key in ("00000", "00001"):
                (folder / f"{key}.yaml").write_text("lidar_pose: [0, 0, 0, 0, 0, 0]\nvehicles: {}\n")
                # Exact-byte fingerprints are tested; no PCD decoding is claimed.
                (folder / f"{key}.pcd").write_bytes(f"synthetic-pcd:{salt}:{cav}:{key}".encode())
    evidence = tmp_path / f"source-{split}.txt"
    evidence.write_text("synthetic source, no official data")
    output = tmp_path / f"prepared-{split}"
    receipt = prepare_native_inputs(source, output, dataset_split=split, ego_agents=ego, source_evidence=evidence)
    return output / "inputs", receipt["input_manifest_sha256"]


def _fixture(tmp_path, train=None, test=None):
    train = train or {"train-a": ("session-a", "a"), "train-b": ("session-b", "b")}
    test = test or {"test-c": ("session-c", "c")}
    train_path, train_sha = _prepare_split(tmp_path, "train", train)
    test_path, test_sha = _prepare_split(tmp_path, "test", test)
    evidence = b"Synthetic explicit session evidence, not an official provenance certificate."
    mapping = {"kind": SESSION_MAP_KIND, "dataset": "V2V4Real",
               "train_manifest_sha256": train_sha, "test_manifest_sha256": test_sha,
               "session_evidence_sha256": hashlib.sha256(evidence).hexdigest(),
               "sequences": {"train": {s: x[0] for s, x in train.items()},
                             "test": {s: x[0] for s, x in test.items()}}}
    kwargs = dict(train_manifest_sha256=train_sha, test_manifest_sha256=test_sha,
                  session_map=mapping, session_evidence=evidence)
    return train_path, test_path, kwargs


def _audit(fixture):
    train, test, kwargs = fixture
    return audit_native_overlap(train, test, **kwargs)


def test_disjoint_checked_inputs_do_not_become_official_or_training_eligible(tmp_path):
    fixture = _fixture(tmp_path)
    report = _audit(fixture)
    assert report["checked_overlap_absent"] is True
    assert report["counts"]["train"]["source_frame_count"] == 8
    assert report["counts"]["test"]["source_frame_count"] == 4
    for key in ("raw_annotations_read", "test_performance_viewed", "source_payloads_modified", "test_cohort_modified",
                "official_split_membership_verified", "session_provenance_verified", "training_eligibility_verified", "paper_eligible"):
        assert report[key] is False
    assert "point_reordering_or_reencoding" in report["unchecked"]
    validate_development_folds(report, {"train-a": "fold-0", "train-b": "fold-1"})
    assert report == _audit(fixture)


@pytest.mark.parametrize("signal", ["sequence", "session", "pcd"])
def test_each_cross_split_signal_blocks_independence(tmp_path, signal):
    train = {"train-a": ("session-a", "a"), "train-b": ("session-b", "b")}
    tests = {"sequence": {"train-a": ("other-session", "other-bytes")},
             "session": {"different-name": ("session-a", "other-bytes")},
             "pcd": {"renamed-clip": ("other-session", "a")}}
    fixture = _fixture(tmp_path, train, tests[signal])
    report = _audit(fixture)
    assert not report["checked_overlap_absent"]
    assert report["directly_affected_train_sequences"] == ["train-a"]
    assert report["affected_train_component_closure"] == ["train-a"]
    assert bool(report["shared_sequence_ids"]) == (signal == "sequence")
    assert bool(report["shared_sessions"]) == (signal == "session")
    assert bool(report["shared_pcd_bytes"]) == (signal == "pcd")
    if signal == "pcd":
        assert len(report["shared_pcd_bytes"]) == 4
        assert sum(len(m["train"]) for m in report["shared_pcd_bytes"]) == 4
    with pytest.raises(V2V4RealInputError, match="unresolved"):
        validate_development_folds(report, {"train-a": "0", "train-b": "1"})


def test_train_prefix_testoutput_is_not_itself_an_exclusion_signal(tmp_path):
    name = "testoutput_CAV_data_2022-03-17-12-04-22_0"
    report = _audit(_fixture(tmp_path, {name: ("train-session", "train")}, {"test-other": ("test-session", "test")}))
    assert report["checked_overlap_absent"]
    assert report["train_development_components"][0]["sequences"] == [name]


def test_transitive_session_content_closure_and_train_only_components(tmp_path):
    train = {"a": ("session-1", "bytes-a"), "b": ("session-1", "bytes-b"),
             "c": ("session-2", "bytes-b"), "d": ("session-3", "bytes-d")}
    fixture = _fixture(tmp_path, train, {"t": ("session-4", "bytes-a")})
    report = _audit(fixture)
    assert report["directly_affected_train_sequences"] == ["a"]
    assert report["affected_train_component_closure"] == ["a", "b", "c"]
    assert [g["sequences"] for g in report["train_development_components"]] == [["a", "b", "c"], ["d"]]
    # Change only test mapping; train component construction must not change.
    changed = copy.deepcopy(fixture[2])
    changed["session_map"]["sequences"]["test"]["t"] = "session-3"
    other = audit_native_overlap(fixture[0], fixture[1], **changed)
    assert other["train_development_components"] == report["train_development_components"]
    assert other["affected_train_component_closure"] == ["a", "b", "c", "d"]


def test_development_folds_must_keep_session_and_byte_components_together(tmp_path):
    train = {"a": ("session-1", "bytes-a"), "b": ("session-1", "bytes-b"),
             "c": ("session-2", "bytes-b"), "d": ("session-3", "bytes-d")}
    report = _audit(_fixture(tmp_path, train))
    validate_development_folds(report, {"a": "0", "b": "0", "c": "0", "d": "1"})
    with pytest.raises(V2V4RealInputError, match="crosses"):
        validate_development_folds(report, {"a": "0", "b": "0", "c": "1", "d": "1"})


@pytest.mark.parametrize("assignment", [{}, {"train-a": "0"}, {"train-a": "0", "train-b": "0"},
                                        {"train-a": "0", "train-b": "1", "test-c": "2"},
                                        {"train-a": 0, "train-b": 1}, {"train-a": " ", "train-b": "1"}])
def test_invalid_or_test_containing_fold_assignments_rejected(tmp_path, assignment):
    report = _audit(_fixture(tmp_path))
    with pytest.raises(V2V4RealInputError):
        validate_development_folds(report, assignment)


@pytest.mark.parametrize("mutation", [
    lambda x: x.update(dataset="SPD"), lambda x: x.update(extra="GT"),
    lambda x: x.update(train_manifest_sha256="0" * 64),
    lambda x: x.update(test_manifest_sha256="0" * 64),
    lambda x: x.update(session_evidence_sha256="0" * 64),
    lambda x: x["sequences"].update(val={}),
    lambda x: x["sequences"]["train"].pop("train-a"),
    lambda x: x["sequences"]["test"].update(extra="extra-session"),
    lambda x: x["sequences"]["train"].update({"train-a": " "}),
    lambda x: x["sequences"].update(test=None),
])
def test_incomplete_or_unbound_session_map_is_not_accepted(tmp_path, mutation):
    train, test, kwargs = _fixture(tmp_path)
    mutation(kwargs["session_map"])
    with pytest.raises(V2V4RealInputError):
        audit_native_overlap(train, test, **kwargs)


def test_cohorts_cannot_be_swapped_or_reused(tmp_path):
    train, test, kwargs = _fixture(tmp_path)
    with pytest.raises(V2V4RealInputError):
        audit_native_overlap(test, train, **kwargs)
    with pytest.raises(V2V4RealInputError, match="distinct"):
        audit_native_overlap(train, train, **kwargs)


def test_source_gt_not_read_and_input_files_not_changed(tmp_path, monkeypatch):
    fixture = _fixture(tmp_path)
    paths = [p for root in fixture[:2] for p in root.rglob("*") if p.is_file()]
    before = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    original_open = Path.open

    def guarded_open(path, *args, **kwargs):
        assert path.suffix != ".yaml", "overlap audit must not read raw GT"
        mode = args[0] if args else kwargs.get("mode", "r")
        assert not any(letter in mode for letter in ("w", "a", "+")), "overlap audit wrote a file"
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", guarded_open)
    _audit(fixture)
    after = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    assert after == before


def test_changed_pcd_and_corrupted_audit_block_reuse(tmp_path):
    fixture = _fixture(tmp_path)
    report = _audit(fixture)
    report["training_eligibility_verified"] = True
    with pytest.raises(V2V4RealInputError, match="digest"):
        validate_development_folds(report, {"train-a": "0", "train-b": "1"})
    path = next(fixture[1].rglob("*.pcd"))
    path.write_bytes(path.read_bytes() + b"tampered")
    with pytest.raises(V2V4RealInputError, match="size"):
        _audit(fixture)


@pytest.mark.parametrize("overlap", [False, True])
def test_cli_writes_new_audit_and_distinguishes_overlap_exit(tmp_path, capsys, overlap):
    fixture = _fixture(tmp_path, test={"test-c": ("other-session", "a" if overlap else "c")})
    train, test, kwargs = fixture
    mapping, evidence, output = tmp_path / "sessions.json", tmp_path / "sessions.txt", tmp_path / "audit.json"
    mapping.write_text(json.dumps(kwargs["session_map"]))
    evidence.write_bytes(kwargs["session_evidence"])
    args = ["--train-inputs", str(train), "--test-inputs", str(test),
            "--train-manifest-sha256", kwargs["train_manifest_sha256"],
            "--test-manifest-sha256", kwargs["test_manifest_sha256"],
            "--session-map", str(mapping), "--session-evidence", str(evidence), "--output", str(output)]
    assert main(args) == (3 if overlap else 0)
    summary = json.loads(capsys.readouterr().out)
    report = json.loads(output.read_bytes())
    assert report["report_sha256"] == summary["report_sha256"]
    assert report["checked_overlap_absent"] is not overlap
    original = output.read_bytes()
    assert main(args) == 2
    assert output.read_bytes() == original
    args[-1] = str(test / "audit.json")
    assert main(args) == 2
    assert not (test / "audit.json").exists()
