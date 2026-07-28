from __future__ import annotations

import copy
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

from tools.resilient_v2x.prepare_v2xset_pair import (
    CAMERA_IDS,
    CONDITION_NAMES,
    DISCLAIMER,
    INVENTORY_TYPE,
    PAIR_TYPE,
    TARGET_TYPE,
    V2XSetPairError,
    prepare_pair_manifest,
    write_pair_manifest,
)
from transvision.dataset.resilient_v2x_manifest import (
    canonical_json_bytes,
    content_sha256,
)


ROOT = Path(__file__).resolve().parents[2]
INTERVAL_US = 100_000


def _identity() -> list[list[float]]:
    return [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]


def _pose(x: float, y: float) -> list[list[float]]:
    value = _identity()
    value[0][3] = x
    value[1][3] = y
    return value


def _intrinsic() -> list[list[float]]:
    return [
        [800.0, 0.0, 640.0],
        [0.0, 800.0, 360.0],
        [0.0, 0.0, 1.0],
    ]


def _write_payload(root: Path, relative: str, content: bytes) -> dict[str, object]:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return {
        "relative_path": relative,
        "size_bytes": len(content),
        "sha256": hashlib.sha256(content).hexdigest(),
    }


def _agent(
    root: Path,
    frame: str,
    agent_id: str,
    numeric_id: int,
    agent_type: str,
    x: float,
    y: float,
    *,
    camera_count: int = 4,
    bad_lidar_hash: bool = False,
    bad_camera_intrinsic: bool = False,
) -> dict[str, object]:
    prefix = f"payloads/{frame}/{agent_id}"
    lidar = _write_payload(
        root,
        f"{prefix}/lidar.bin",
        f"lidar:{frame}:{agent_id}".encode(),
    )
    if bad_lidar_hash:
        lidar["sha256"] = "0" * 64
    cameras = []
    for camera_id in CAMERA_IDS[:camera_count]:
        image = _write_payload(
            root,
            f"{prefix}/{camera_id}.jpg",
            f"image:{frame}:{agent_id}:{camera_id}".encode(),
        )
        intrinsic = _intrinsic()
        if bad_camera_intrinsic and camera_id == CAMERA_IDS[0]:
            intrinsic[0][0] = -1.0
        cameras.append(
            {
                "camera_id": camera_id,
                "image": image,
                "intrinsic": intrinsic,
                "agent_from_camera": _identity(),
            }
        )
    return {
        "agent_id": agent_id,
        "numeric_id": numeric_id,
        "agent_type": agent_type,
        "world_from_agent": _pose(x, y),
        "lidar": {
            "payload": lidar,
            "agent_from_lidar": _identity(),
        },
        "cameras": cameras,
    }


def _frame(timestamp_us: int, agents: list[dict[str, object]]) -> dict[str, object]:
    return {"timestamp_us": timestamp_us, "agents": agents}


def _scene(
    scene_id: str,
    split: str,
    frames: list[dict[str, object]],
    *,
    fixed_ego: str | None = None,
) -> dict[str, object]:
    return {
        "scene_id": scene_id,
        "split": split,
        "standard_fixed_ego_agent_id": fixed_ego,
        "frames": frames,
    }


def _inventory(
    scenes: list[dict[str, object]],
    *,
    delays_ms: list[int] | None = None,
    history_limit: int = 1,
    frame_interval_us: int = INTERVAL_US,
) -> dict[str, object]:
    value: dict[str, object] = {
        "schema_version": 2,
        "artifact_type": INVENTORY_TYPE,
        "dataset_name": "V2XSet",
        "coordinate_unit": "meter",
        "timestamp_unit": "microsecond",
        "configured_delays_ms": [0, 100] if delays_ms is None else delays_ms,
        "history_limit": history_limit,
        "frame_interval_us": frame_interval_us,
        "scenes": scenes,
    }
    value["content_sha256"] = content_sha256(value)
    return value


def _write_inventory(
    path: Path,
    scenes: list[dict[str, object]],
    **kwargs: object,
) -> dict[str, object]:
    value = _inventory(scenes, **kwargs)
    path.write_bytes(canonical_json_bytes(value))
    return value


def _standard_agents(
    root: Path,
    label: str,
    *,
    av_two_x: float = 0.0,
    infra_three_x: float = 5.0,
    include_av_ten: bool = True,
    include_infra_seven: bool = True,
) -> list[dict[str, object]]:
    agents = [
        _agent(root, label, "av-two", 2, "av", av_two_x, 0.0),
        _agent(
            root,
            label,
            "infra-three",
            3,
            "infrastructure",
            infra_three_x,
            0.0,
        ),
    ]
    if include_av_ten:
        agents.append(_agent(root, label, "av-ten", 10, "av", 0.0, 1.0))
    if include_infra_seven:
        agents.append(
            _agent(
                root,
                label,
                "infra-seven",
                7,
                "infrastructure",
                -5.0,
                0.0,
            )
        )
    return agents


def test_train_uses_earliest_anchor_numeric_ego_nearest_infra_and_freezes_pair(
    tmp_path: Path,
) -> None:
    data_root = tmp_path / "data"
    data_root.mkdir()
    frames = [
        _frame(0, _standard_agents(data_root, "s0-t0")),
        _frame(INTERVAL_US, _standard_agents(data_root, "s0-t1")),
        _frame(2 * INTERVAL_US, _standard_agents(data_root, "s0-t2")),
        _frame(
            3 * INTERVAL_US,
            _standard_agents(data_root, "s0-t3")
            + [
                _agent(data_root, "s0-t3", "av-one", 1, "av", 0.0, 0.0),
                _agent(
                    data_root,
                    "s0-t3",
                    "infra-one",
                    1,
                    "infrastructure",
                    0.5,
                    0.0,
                ),
            ],
        ),
        _frame(
            4 * INTERVAL_US,
            _standard_agents(data_root, "s0-t4", infra_three_x=71.0),
        ),
        _frame(5 * INTERVAL_US, _standard_agents(data_root, "s0-t5")),
    ]
    inventory_path = tmp_path / "inventory.json"
    _write_inventory(inventory_path, [_scene("sequence-0", "train", frames)])

    manifest = prepare_pair_manifest(inventory_path, data_root)

    sequence = manifest["sequences"][0]
    assert sequence["anchor_timestamp_us"] == 2 * INTERVAL_US
    assert sequence["ego_agent_id"] == "av-two"
    assert sequence["ego_numeric_id"] == 2
    assert sequence["infrastructure_agent_id"] == "infra-three"
    assert sequence["infrastructure_numeric_id"] == 3
    assert sequence["anchor_distance_m"] == pytest.approx(5.0)
    assert [pair["timestamp_us"] for pair in manifest["pairs"]] == [
        2 * INTERVAL_US,
        3 * INTERVAL_US,
        5 * INTERVAL_US,
    ]
    assert {
        (pair["ego_agent_id"], pair["infrastructure_agent_id"])
        for pair in manifest["pairs"]
    } == {("av-two", "infra-three")}
    later = next(
        pair for pair in manifest["pairs"] if pair["timestamp_us"] == 3 * INTERVAL_US
    )
    assert later["target_eligibility_agent_ids"] == ["av-two", "infra-three"]
    assert {"av-one", "infra-one", "av-ten", "infra-seven"}.issubset(
        later["excluded_agent_ids"]
    )
    assert [
        item["timestamp_us"] for item in later["endpoint_coverage"]["ego_history"]
    ] == [3 * INTERVAL_US, 2 * INTERVAL_US]
    assert [
        item["timestamp_us"]
        for item in later["endpoint_coverage"]["infrastructure_histories"][1]["history"]
    ] == [2 * INTERVAL_US, INTERVAL_US]
    excluded = {item["timestamp_us"]: item for item in manifest["excluded_frames"]}
    assert excluded[0]["reasons"][0]["code"] == "before_sequence_anchor"
    assert excluded[INTERVAL_US]["reasons"][0]["code"] == ("before_sequence_anchor")
    failure = excluded[4 * INTERVAL_US]["reasons"][0]
    assert failure["code"] == "frozen_pair_ineligible"
    assert any(
        detail["code"] == "outside_max_collaborator_distance"
        and detail["distance_m"] == pytest.approx(71.0)
        for detail in failure["details"]
    )


def test_validation_keeps_input_standard_fixed_ego_and_never_substitutes(
    tmp_path: Path,
) -> None:
    data_root = tmp_path / "data"
    data_root.mkdir()
    first = _frame(
        0,
        [
            _agent(data_root, "v0", "av-one", 1, "av", 0.0, 0.0),
            _agent(
                data_root,
                "v0",
                "infra-nine",
                9,
                "infrastructure",
                1.0,
                0.0,
            ),
        ],
    )
    second = _frame(
        INTERVAL_US,
        [
            _agent(data_root, "v1", "av-one", 1, "av", 0.0, 0.0),
            _agent(data_root, "v1", "av-ten", 10, "av", 0.0, 0.0),
            _agent(
                data_root,
                "v1",
                "infra-nine",
                9,
                "infrastructure",
                5.0,
                0.0,
            ),
            _agent(
                data_root,
                "v1",
                "infra-two",
                2,
                "infrastructure",
                -5.0,
                0.0,
            ),
        ],
    )
    no_fixed_frames = [
        _frame(
            0,
            [
                _agent(data_root, "n0", "av-one", 1, "av", 0.0, 0.0),
                _agent(
                    data_root,
                    "n0",
                    "infra-one",
                    1,
                    "infrastructure",
                    1.0,
                    0.0,
                ),
            ],
        )
    ]
    inventory_path = tmp_path / "inventory.json"
    _write_inventory(
        inventory_path,
        [
            _scene(
                "validation-included",
                "validation",
                [first, second],
                fixed_ego="av-ten",
            ),
            _scene(
                "validation-excluded",
                "validation",
                no_fixed_frames,
                fixed_ego="av-ten",
            ),
        ],
        delays_ms=[0],
        history_limit=0,
    )

    manifest = prepare_pair_manifest(inventory_path, data_root)

    sequences = {item["scene_id"]: item for item in manifest["sequences"]}
    included = sequences["validation-included"]
    assert included["anchor_timestamp_us"] == INTERVAL_US
    assert included["ego_agent_id"] == "av-ten"
    assert included["infrastructure_agent_id"] == "infra-two"
    assert included["anchor_attempts"][0]["frame_reasons"][0]["code"] == (
        "standard_fixed_ego_missing_no_substitution"
    )
    assert included["anchor_attempts"][1]["ignored_av_agent_ids"] == ["av-one"]
    assert sequences["validation-excluded"]["status"] == "excluded_no_anchor"
    excluded = next(
        item
        for item in manifest["excluded_frames"]
        if item["scene_id"] == "validation-excluded"
    )
    audit = excluded["reasons"][1]["selection_audit"]
    assert audit["frame_reasons"][0]["code"] == (
        "standard_fixed_ego_missing_no_substitution"
    )


def test_all_delay_and_history_endpoints_are_required_with_exact_failure_trace(
    tmp_path: Path,
) -> None:
    data_root = tmp_path / "data"
    data_root.mkdir()
    frames = [
        _frame(
            0,
            [
                _agent(data_root, "e0", "av", 1, "av", 0.0, 0.0),
                _agent(
                    data_root,
                    "e0",
                    "infra",
                    1,
                    "infrastructure",
                    2.0,
                    0.0,
                    camera_count=3,
                ),
            ],
        ),
        _frame(
            INTERVAL_US,
            [
                _agent(data_root, "e1", "av", 1, "av", 0.0, 0.0),
                _agent(
                    data_root,
                    "e1",
                    "infra",
                    1,
                    "infrastructure",
                    2.0,
                    0.0,
                ),
            ],
        ),
        _frame(
            2 * INTERVAL_US,
            [
                _agent(data_root, "e2", "av", 1, "av", 0.0, 0.0),
                _agent(
                    data_root,
                    "e2",
                    "infra",
                    1,
                    "infrastructure",
                    2.0,
                    0.0,
                ),
            ],
        ),
        _frame(
            3 * INTERVAL_US,
            [
                _agent(data_root, "e3", "av", 1, "av", 0.0, 0.0),
                _agent(
                    data_root,
                    "e3",
                    "infra",
                    1,
                    "infrastructure",
                    2.0,
                    0.0,
                ),
            ],
        ),
    ]
    inventory_path = tmp_path / "inventory.json"
    _write_inventory(inventory_path, [_scene("endpoint-sequence", "train", frames)])

    manifest = prepare_pair_manifest(inventory_path, data_root)

    sequence = manifest["sequences"][0]
    assert sequence["anchor_timestamp_us"] == 3 * INTERVAL_US
    attempt = next(
        item
        for item in sequence["anchor_attempts"]
        if item["timestamp_us"] == 2 * INTERVAL_US
    )
    reasons = attempt["candidate_pairs"][0]["reasons"]
    endpoint_failure = next(
        reason
        for reason in reasons
        if reason["code"] == "agent_invalid_at_required_endpoint"
    )
    assert endpoint_failure["member"] == "infrastructure"
    assert endpoint_failure["delay_ms"] == 100
    assert endpoint_failure["horizon"] == 1
    assert endpoint_failure["required_timestamp_us"] == 0
    assert any(
        detail["code"] == "camera_set_mismatch"
        for detail in endpoint_failure["details"]
    )


def test_later_frame_missing_frozen_endpoint_is_excluded_without_reselection(
    tmp_path: Path,
) -> None:
    data_root = tmp_path / "data"
    data_root.mkdir()
    frames = [
        _frame(0, _standard_agents(data_root, "m0")),
        _frame(
            INTERVAL_US,
            [
                _agent(data_root, "m1", "av-two", 2, "av", 0.0, 0.0),
                _agent(data_root, "m1", "av-ten", 10, "av", 0.0, 0.0),
                _agent(
                    data_root,
                    "m1",
                    "infra-seven",
                    7,
                    "infrastructure",
                    1.0,
                    0.0,
                ),
            ],
        ),
    ]
    inventory_path = tmp_path / "inventory.json"
    _write_inventory(
        inventory_path,
        [_scene("missing-later", "train", frames)],
        delays_ms=[0],
        history_limit=0,
    )

    manifest = prepare_pair_manifest(inventory_path, data_root)

    assert manifest["sequences"][0]["infrastructure_agent_id"] == "infra-three"
    assert [pair["timestamp_us"] for pair in manifest["pairs"]] == [0]
    excluded = manifest["excluded_frames"][0]
    assert excluded["timestamp_us"] == INTERVAL_US
    failure = excluded["reasons"][0]
    assert failure["code"] == "frozen_pair_ineligible"
    assert failure["substitution_forbidden"] is True
    assert any(
        reason["code"] == "agent_missing_at_required_endpoint"
        and reason["agent_id"] == "infra-three"
        and reason["delay_ms"] == 0
        and reason["horizon"] == 0
        for reason in failure["details"]
    )


def test_invalid_unselected_agents_do_not_affect_fixed_pair_target_eligibility(
    tmp_path: Path,
) -> None:
    data_root = tmp_path / "data"
    data_root.mkdir()
    first = _frame(
        0,
        [
            _agent(data_root, "u0", "av-one", 1, "av", 0.0, 0.0, bad_lidar_hash=True),
            _agent(data_root, "u0", "av-two", 2, "av", 0.0, 0.0),
            _agent(
                data_root,
                "u0",
                "infra-one",
                1,
                "infrastructure",
                0.5,
                0.0,
                bad_camera_intrinsic=True,
            ),
            _agent(
                data_root,
                "u0",
                "infra-three",
                3,
                "infrastructure",
                3.0,
                0.0,
            ),
        ],
    )
    second = _frame(
        INTERVAL_US,
        [
            _agent(data_root, "u1", "av-one", 1, "av", 0.0, 0.0, bad_lidar_hash=True),
            _agent(data_root, "u1", "av-two", 2, "av", 0.0, 0.0),
            _agent(
                data_root,
                "u1",
                "infra-one",
                1,
                "infrastructure",
                0.1,
                0.0,
                bad_lidar_hash=True,
            ),
            _agent(
                data_root,
                "u1",
                "infra-three",
                3,
                "infrastructure",
                3.0,
                0.0,
            ),
        ],
    )
    inventory_path = tmp_path / "inventory.json"
    _write_inventory(
        inventory_path,
        [_scene("unselected-invalid", "train", [first, second])],
        delays_ms=[0],
        history_limit=0,
    )

    manifest = prepare_pair_manifest(inventory_path, data_root)

    assert manifest["pair_count"] == 2
    assert {
        (pair["ego_agent_id"], pair["infrastructure_agent_id"])
        for pair in manifest["pairs"]
    } == {("av-two", "infra-three")}
    assert all(
        pair["target_eligibility_agent_ids"] == ["av-two", "infra-three"]
        for pair in manifest["pairs"]
    )
    assert all(
        {"av-one", "infra-one"}.issubset(pair["excluded_agent_ids"])
        for pair in manifest["pairs"]
    )


def _single_pair_inventory(root: Path) -> list[dict[str, object]]:
    return [
        _scene(
            "single",
            "train",
            [
                _frame(
                    0,
                    [
                        _agent(root, "single", "av", 1, "av", 0.0, 0.0),
                        _agent(
                            root,
                            "single",
                            "infra",
                            1,
                            "infrastructure",
                            1.0,
                            0.0,
                        ),
                    ],
                )
            ],
        )
    ]


def _complete_delay_pair_inventory(root: Path) -> list[dict[str, object]]:
    frames = []
    for index in range(4):
        label = f"complete-delay-{index}"
        frames.append(
            _frame(
                index * INTERVAL_US,
                [
                    _agent(root, label, "av", 1, "av", 0.0, 0.0),
                    _agent(
                        root,
                        label,
                        "infra",
                        1,
                        "infrastructure",
                        1.0,
                        0.0,
                    ),
                ],
            )
        )
    return [_scene("complete-delay", "train", frames)]


def test_exact_70m_boundary_is_eligible(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    data_root.mkdir()
    frames = [
        _frame(
            0,
            [
                _agent(data_root, "edge", "av", 1, "av", 0.0, 0.0),
                _agent(
                    data_root,
                    "edge",
                    "infra",
                    1,
                    "infrastructure",
                    70.0,
                    0.0,
                ),
            ],
        )
    ]
    inventory_path = tmp_path / "inventory.json"
    _write_inventory(
        inventory_path,
        [_scene("edge-70m", "train", frames)],
        delays_ms=[0],
        history_limit=0,
    )

    manifest = prepare_pair_manifest(inventory_path, data_root)

    assert manifest["pair_count"] == 1
    assert manifest["pairs"][0]["infrastructure_distance_m"] == pytest.approx(70.0)


def test_nested_target_manifest_and_all_condition_references_are_invariant(
    tmp_path: Path,
) -> None:
    data_root = tmp_path / "data"
    data_root.mkdir()
    inventory_path = tmp_path / "inventory.json"
    inventory = _write_inventory(
        inventory_path,
        _complete_delay_pair_inventory(data_root),
        delays_ms=[0, 100, 200, 300],
        history_limit=0,
    )

    manifest = prepare_pair_manifest(inventory_path, data_root)
    target = manifest["target_manifest"]
    assert target["artifact_type"] == TARGET_TYPE
    assert target["content_sha256"] == content_sha256(target)
    assert target["sample_count"] == 1
    assert manifest["content_sha256"] == content_sha256(manifest)
    assert (
        manifest["input_summary"]["inventory_content_sha256"]
        == (inventory["content_sha256"])
    )
    references = manifest["condition_target_references"]
    assert len(references) == 4 * 3 + (len(CONDITION_NAMES) - 3)
    assert {item["target_manifest_content_sha256"] for item in references} == {
        target["content_sha256"]
    }
    assert {item["target_sample_count"] for item in references} == {1}
    assert manifest["derivation_ids"]["included_target_sample_ids"]["count"] == 1
    assert manifest["derivation_ids"]["included_sequence_ids"]["ids"] == [
        "complete-delay"
    ]


def test_publication_is_canonical_idempotent_and_rejects_condition_tampering(
    tmp_path: Path,
) -> None:
    data_root = tmp_path / "data"
    data_root.mkdir()
    inventory_path = tmp_path / "inventory.json"
    _write_inventory(
        inventory_path,
        _single_pair_inventory(data_root),
        delays_ms=[0],
        history_limit=0,
    )
    manifest = prepare_pair_manifest(inventory_path, data_root)
    output = tmp_path / "pair.json"

    write_pair_manifest(output, manifest)
    write_pair_manifest(output, manifest)

    raw = output.read_bytes()
    assert raw == canonical_json_bytes(manifest)
    tampered = copy.deepcopy(manifest)
    tampered["condition_target_references"][0]["target_manifest_content_sha256"] = (
        "f" * 64
    )
    tampered["content_sha256"] = content_sha256(tampered)
    with pytest.raises(V2XSetPairError, match="not invariant"):
        write_pair_manifest(tmp_path / "tampered.json", tampered)

    conflicting = copy.deepcopy(manifest)
    conflicting["input_summary"]["inventory_file_sha256"] = "f" * 64
    conflicting["content_sha256"] = content_sha256(conflicting)
    with pytest.raises(V2XSetPairError, match="destination conflict"):
        write_pair_manifest(output, conflicting)


def test_inventory_schema_fixed_ego_and_exact_delay_grid_are_strict(
    tmp_path: Path,
) -> None:
    data_root = tmp_path / "data"
    data_root.mkdir()
    scenes = _single_pair_inventory(data_root)

    train_with_fixed = copy.deepcopy(scenes)
    train_with_fixed[0]["standard_fixed_ego_agent_id"] = "av"
    path = tmp_path / "train-fixed.json"
    _write_inventory(path, train_with_fixed, delays_ms=[0], history_limit=0)
    with pytest.raises(V2XSetPairError, match="must be null"):
        prepare_pair_manifest(path, data_root)

    validation_without_fixed = copy.deepcopy(scenes)
    validation_without_fixed[0]["split"] = "validation"
    path = tmp_path / "validation-no-fixed.json"
    _write_inventory(path, validation_without_fixed, delays_ms=[0], history_limit=0)
    with pytest.raises(V2XSetPairError, match="canonical identifier"):
        prepare_pair_manifest(path, data_root)

    path = tmp_path / "off-grid.json"
    _write_inventory(
        path,
        scenes,
        delays_ms=[0, 50],
        history_limit=0,
        frame_interval_us=INTERVAL_US,
    )
    with pytest.raises(V2XSetPairError, match="map exactly"):
        prepare_pair_manifest(path, data_root)

    pretty = tmp_path / "pretty.json"
    pretty.write_text(json.dumps(_inventory([]), indent=2), encoding="utf-8")
    with pytest.raises(V2XSetPairError, match="not canonical"):
        prepare_pair_manifest(pretty, data_root)


def test_payload_mutation_is_detected_at_required_endpoint(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    data_root.mkdir()
    scenes = _single_pair_inventory(data_root)
    inventory_path = tmp_path / "inventory.json"
    _write_inventory(
        inventory_path,
        scenes,
        delays_ms=[0],
        history_limit=0,
    )
    ego = scenes[0]["frames"][0]["agents"][0]
    lidar_path = data_root / ego["lidar"]["payload"]["relative_path"]
    lidar_path.write_bytes(b"mutated after inventory")

    manifest = prepare_pair_manifest(inventory_path, data_root)

    assert manifest["pair_count"] == 0
    audit = manifest["sequences"][0]["anchor_attempts"][0]
    reasons = audit["candidate_pairs"][0]["reasons"]
    invalid = next(
        reason
        for reason in reasons
        if reason["code"] == "agent_invalid_at_required_endpoint"
        and reason["member"] == "ego"
    )
    assert any(
        detail["code"] == "invalid_lidar_payload"
        and detail["failure"] == "size_mismatch"
        for detail in invalid["details"]
    )


def test_cli_emits_strict_sequence_pair_manifest_and_summary(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    data_root.mkdir()
    inventory_path = tmp_path / "inventory.json"
    _write_inventory(
        inventory_path,
        _single_pair_inventory(data_root),
        delays_ms=[0],
        history_limit=0,
    )
    output = tmp_path / "pair.json"

    result = subprocess.run(
        [
            sys.executable,
            "tools/resilient_v2x/prepare_v2xset_pair.py",
            str(inventory_path),
            "--data-root",
            str(data_root),
            "--out",
            str(output),
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    summary = json.loads(result.stdout)
    manifest = json.loads(output.read_bytes())
    assert manifest["schema_version"] == 2
    assert manifest["artifact_type"] == PAIR_TYPE
    assert manifest["protocol_disclaimer"] == DISCLAIMER
    assert manifest["content_sha256"] == content_sha256(manifest)
    assert summary["pair_count"] == 1
    assert summary["standard_multi_agent_protocol_implemented"] is False
