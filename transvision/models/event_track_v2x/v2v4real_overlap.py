"""GT-free native train/test overlap audit, bound to immutable input digests.

Equality of sequence/session IDs and exact PCD bytes are conservative exclusion
signals. Absence of those signals is NOT a proof of dataset provenance, absence
of re-encoded/near duplicates, or eligibility of pretrained weights.
"""

from __future__ import annotations

from collections import defaultdict
import hashlib
import json
from pathlib import Path
from typing import Any

from .v2v4real_inputs import V2V4RealInputError, load_prepared_frames


SESSION_MAP_KIND = "v2v4real_session_map_v1"
OVERLAP_KIND = "v2v4real_input_overlap_audit_v1"


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")


def _sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _name(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip()) and value == value.strip()


def _by_hash(records):
    result = defaultdict(list)
    for row in records:
        result[row["pcd_sha256"]].append({key: row[key] for key in ("sequence_id", "cav_id", "frame_key")})
    return dict(result)


def _train_components(sequences, session_map, content_groups):
    # Neither test records nor test sessions influence these grouping edges.
    parents = {sequence: sequence for sequence in sequences}

    def root(sequence):
        while parents[sequence] != sequence:
            parents[sequence] = parents[parents[sequence]]
            sequence = parents[sequence]
        return sequence

    def merge(group):
        roots = sorted({root(s) for s in group})
        for other in roots[1:]:
            parents[other] = roots[0]

    sessions = defaultdict(list)
    for sequence in sequences:
        sessions[session_map[sequence]].append(sequence)
    for group in sessions.values():
        merge(group)
    for rows in content_groups.values():
        merge(row["sequence_id"] for row in rows)
    groups = defaultdict(list)
    for sequence in sorted(sequences):
        groups[root(sequence)].append(sequence)
    return [groups[key] for key in sorted(groups)]


def audit_native_overlap(
    train_inputs: Path, test_inputs: Path, *, train_manifest_sha256: str,
    test_manifest_sha256: str, session_map: dict[str, Any], session_evidence: bytes,
) -> dict[str, Any]:
    """Read only verified pose/PCD projections; never raw YAML or test metrics.

    Session IDs must come from an explicit reviewed map, never from stripping
    filename suffixes. ``session_evidence`` bytes are hash-bound but their
    semantic truth cannot be validated automatically by this function.
    """
    if (not isinstance(session_map, dict) or set(session_map) != {
            "kind", "dataset", "train_manifest_sha256", "test_manifest_sha256",
            "session_evidence_sha256", "sequences"}
            or session_map["kind"] != SESSION_MAP_KIND or session_map["dataset"] != "V2V4Real"):
        raise V2V4RealInputError("invalid native session map schema")
    if not isinstance(session_evidence, bytes) or not session_evidence.strip():
        raise V2V4RealInputError("explicit nonempty session provenance evidence is required")
    if (session_map["train_manifest_sha256"] != train_manifest_sha256
            or session_map["test_manifest_sha256"] != test_manifest_sha256
            or session_map["session_evidence_sha256"] != _sha(session_evidence)):
        raise V2V4RealInputError("session map is not bound to the input/evidence digests")
    if Path(train_inputs).resolve() == Path(test_inputs).resolve():
        raise V2V4RealInputError("train and test must be distinct projection trees")
    manifests, records = {}, {}
    for split, path, digest in (("train", train_inputs, train_manifest_sha256),
                                ("test", test_inputs, test_manifest_sha256)):
        manifest, rows = load_prepared_frames(path, expected_manifest_sha256=digest)
        if manifest["dataset_split"] != split:
            raise V2V4RealInputError(f"{split} argument does not contain the declared native split")
        manifests[split], records[split] = manifest, rows
    sequence_maps = session_map["sequences"]
    if not isinstance(sequence_maps, dict) or set(sequence_maps) != {"train", "test"}:
        raise V2V4RealInputError("session map must contain exactly train and test")
    for split in ("train", "test"):
        if (not isinstance(sequence_maps[split], dict)
                or set(sequence_maps[split]) != set(manifests[split]["ego_agents"])
                or any(not _name(value) for value in sequence_maps[split].values())):
            raise V2V4RealInputError(f"session map must cover every {split} sequence exactly")

    shared_sequences = sorted(set(sequence_maps["train"]) & set(sequence_maps["test"]))
    shared_sessions = sorted(set(sequence_maps["train"].values()) & set(sequence_maps["test"].values()))
    hashes = {split: _by_hash(records[split]) for split in ("train", "test")}
    shared_hashes = sorted(set(hashes["train"]) & set(hashes["test"]))
    content_matches = [{"pcd_sha256": digest, "train": hashes["train"][digest], "test": hashes["test"][digest]}
                       for digest in shared_hashes]
    session_matches = [{"session_id": session,
                        "train_sequences": sorted(s for s, g in sequence_maps["train"].items() if g == session),
                        "test_sequences": sorted(s for s, g in sequence_maps["test"].items() if g == session)}
                       for session in shared_sessions]
    affected = set(shared_sequences)
    for match in session_matches:
        affected.update(match["train_sequences"])
    for match in content_matches:
        affected.update(row["sequence_id"] for row in match["train"])
    components = _train_components(sequence_maps["train"], sequence_maps["train"], hashes["train"])
    # Report the whole training leakage component. Do not silently remove it.
    affected_closure = sorted(s for group in components if affected.intersection(group) for s in group)
    group_records = [{"component_id": _sha(_canonical(group)), "sequences": group}
                     for group in components]
    report = {
        "kind": OVERLAP_KIND, "dataset": "V2V4Real",
        "train_manifest_sha256": train_manifest_sha256,
        "test_manifest_sha256": test_manifest_sha256,
        "session_map_sha256": _sha(_canonical(session_map)),
        "session_evidence_sha256": _sha(session_evidence),
        "counts": {split: {key: manifests[split][key] for key in (
            "sequence_count", "source_frame_count", "paired_frame_count")} for split in ("train", "test")},
        "shared_sequence_ids": shared_sequences, "shared_sessions": session_matches,
        "shared_pcd_bytes": content_matches,
        "directly_affected_train_sequences": sorted(affected),
        "affected_train_component_closure": affected_closure,
        "train_development_components": group_records,
        "checked_overlap_absent": not bool(affected),
        "raw_annotations_read": False, "test_performance_viewed": False,
        "source_payloads_modified": False, "test_cohort_modified": False,
        "official_split_membership_verified": False, "session_provenance_verified": False,
        "training_eligibility_verified": False, "paper_eligible": False,
        "unchecked": ["official_archive_membership_and_completeness", "session_evidence_truth",
                      "point_reordering_or_reencoding", "spatiotemporal_near_duplicates",
                      "pretrained_weight_training_and_selection_sources"],
    }
    report["report_sha256"] = _sha(_canonical(report))
    return report


def validate_development_folds(report: dict[str, Any], assignments: dict[str, str]) -> None:
    """Ensure a proposed train-only fold map never splits a leakage component.

    Passing this function is only a fold consistency check, not training
    authorization. Callers must keep the independently pinned audit digest.
    """
    if not isinstance(report, dict) or report.get("kind") != OVERLAP_KIND:
        raise V2V4RealInputError("expected a native overlap audit")
    payload = dict(report)
    digest = payload.pop("report_sha256", None)
    if digest != _sha(_canonical(payload)):
        raise V2V4RealInputError("overlap report digest mismatch")
    if report["checked_overlap_absent"] is not True:
        raise V2V4RealInputError("unresolved train/test overlap blocks development fold acceptance")
    components = report["train_development_components"]
    sequences = {sequence for group in components for sequence in group["sequences"]}
    if (not isinstance(assignments, dict) or set(assignments) != sequences
            or any(not _name(fold) for fold in assignments.values())
            or len(set(assignments.values())) < 2):
        raise V2V4RealInputError("fold assignments must cover all train sequences in at least two folds")
    for group in components:
        if len({assignments[sequence] for sequence in group["sequences"]}) != 1:
            raise V2V4RealInputError("a session/content-connected training component crosses folds")
