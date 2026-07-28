from __future__ import annotations

import copy
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

from tools.resilient_v2x.export_v2xset_standard import (
    MANIFEST_TYPE,
    METHOD_ELIGIBILITY,
    TRACE_TYPE,
    TRUNCATION_RULE,
    V2XSetStandardError,
    export_standard_manifest,
    validate_standard_manifest,
    write_standard_manifest,
)
from transvision.dataset.resilient_v2x_manifest import (
    canonical_json_bytes,
    content_sha256,
)


ROOT = Path(__file__).resolve().parents[2]


def _write_lidar(root: Path, relative: str, content: bytes) -> dict[str, object]:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return {
        "relative_path": relative,
        "size_bytes": len(content),
        "sha256": hashlib.sha256(content).hexdigest(),
    }


def _delay(
    *,
    target_us: int,
    delay_ms: int,
    offset: int,
) -> dict[str, int]:
    source_us = target_us - delay_ms * 1000
    return {
        "generated_delay_ms": delay_ms,
        "source_timestamp_us": source_us,
        "arrival_timestamp_us": source_us + delay_ms * 1000,
        "causal_frame_offset": offset,
    }


def _record(
    root: Path,
    *,
    split: str,
    sample_id: str,
    target_us: int,
    selected_vehicle_index: int = 1,
    scene_id: str = "scene-1",
    epoch: int | None = 0,
    draw_index: int = 0,
) -> dict[str, object]:
    available = [
        {
            "agent_id": "veh-1",
            "agent_type": "vehicle",
            "distance_to_ego_m": 5.0,
        },
        {
            "agent_id": "veh-2",
            "agent_type": "vehicle",
            "distance_to_ego_m": 0.0,
        },
        {
            "agent_id": "infra-1",
            "agent_type": "infrastructure",
            "distance_to_ego_m": 20.0,
        },
        {
            "agent_id": "veh-3",
            "agent_type": "vehicle",
            "distance_to_ego_m": 70.0,
        },
        {
            "agent_id": "infra-far",
            "agent_type": "infrastructure",
            "distance_to_ego_m": 70.01,
        },
    ]
    ordered = ["veh-2", "infra-1", "veh-1", "veh-3"]
    admitted = []
    for index, agent_id in enumerate(ordered[:3]):
        available_agent = next(
            item for item in available if item["agent_id"] == agent_id
        )
        is_ego = agent_id == "veh-2"
        delay = _delay(
            target_us=target_us,
            delay_ms=0 if is_ego else 200,
            offset=0 if is_ego else 2,
        )
        admitted.append(
            {
                "agent_id": agent_id,
                "agent_type": available_agent["agent_type"],
                "distance_to_ego_m": available_agent["distance_to_ego_m"],
                "lidar": _write_lidar(
                    root,
                    f"lidar/{sample_id}/{agent_id}.bin",
                    f"{sample_id}:{agent_id}:{index}".encode(),
                ),
                "delay": delay,
            }
        )
    random = split == "train"
    return {
        "sample_id": sample_id,
        "scene_id": scene_id,
        "target_timestamp_us": target_us,
        "epoch": epoch if random else None,
        "draw_index": draw_index,
        "ego_selection": {
            "mode": "random" if random else "fixed",
            "candidate_vehicle_agent_ids": ["veh-1", "veh-2", "veh-3"],
            "selected_index": selected_vehicle_index,
            "rng_seed": 2026 if random else None,
            "rng_state_before_sha256": "a" * 64 if random else None,
        },
        "available_agents": available,
        "ordered_in_range_agent_ids": ordered,
        "admitted_agents": admitted,
        "truncated_agent_ids": ["veh-3"],
    }


def _trace(
    records: list[dict[str, object]],
    *,
    split: str = "train",
    nominal_latency_ms: int = 200,
    condition_kind: str = "source_aligned",
) -> dict[str, object]:
    trace: dict[str, object] = {
        "schema_version": 1,
        "artifact_type": TRACE_TYPE,
        "dataset_name": "V2XSet",
        "split": split,
        "dataset_identity": {
            "release_id": "v2xset-release-verified-locally",
            "release_inventory_sha256": "1" * 64,
            "split_id": f"coformer-{split}",
            "split_sha256": "2" * 64,
        },
        "loader_identity": {
            "implementation": "source V2XSet dataloader with trace hook",
            "repository_commit": "3" * 40,
            "config_path": "opencood/config/v2xset.yaml",
            "config_sha256": "4" * 64,
            "trace_hook": "opencood/data_utils/datasets/intermediate_fusion_dataset.py:__getitem__",
        },
        "selection_policy": {
            "communication_range_m": 70.0,
            "max_cav": 3,
            "agent_ordering_rule": "ego first, then loader dictionary iteration order",
            "agent_ordering_source": "intermediate_fusion_dataset.py:get_item_single_car",
            "truncation_rule": TRUNCATION_RULE,
            "truncation_source": "intermediate_fusion_dataset.py:max_cav_slice",
            "ego_selection_rule": (
                "random_per_epoch" if split == "train" else "fixed_per_scene"
            ),
            "ego_selection_source": "base_dataset.py:retrieve_base_data",
        },
        "delay_policy": {
            "nominal_latency_ms": nominal_latency_ms,
            "condition_kind": condition_kind,
            "unit": "millisecond",
            "generation_rule": "fixed delay from the executed configuration",
            "causal_frame_mapping_rule": "latest source whose delayed arrival is not after target",
            "uniform_across_remote_agents": True,
            "source_location": "base_dataset.py:time_delay_calculation",
        },
        "records": records,
    }
    trace["content_sha256"] = content_sha256(trace)
    return trace


def _write_trace(path: Path, trace: dict[str, object]) -> None:
    path.write_bytes(canonical_json_bytes(trace))


def _rehash(trace: dict[str, object]) -> dict[str, object]:
    trace["content_sha256"] = content_sha256(trace)
    return trace


def test_train_trace_preserves_random_ego_range_order_truncation_and_delays(
    tmp_path: Path,
) -> None:
    data_root = tmp_path / "data"
    data_root.mkdir()
    records = [
        _record(
            data_root,
            split="train",
            sample_id="sample-0",
            target_us=1_000_000,
            epoch=0,
            draw_index=0,
        ),
        _record(
            data_root,
            split="train",
            sample_id="sample-0",
            target_us=1_000_000,
            epoch=1,
            draw_index=1,
        ),
    ]
    trace = _trace(records)
    trace_path = tmp_path / "trace.json"
    _write_trace(trace_path, trace)

    manifest = export_standard_manifest(trace_path, data_root)

    assert manifest["artifact_type"] == MANIFEST_TYPE
    assert manifest["modalities"] == ["lidar"]
    assert manifest["record_count"] == 2
    assert manifest["scene_count"] == 1
    assert manifest["epoch_ids"] == [0, 1]
    assert manifest["selection_policy"]["communication_range_m"] == 70.0
    assert manifest["selection_policy"]["max_cav"] == 3
    first = manifest["records"][0]
    assert first["ego_selection"]["mode"] == "random"
    assert first["ego_selection"]["rng_seed"] == 2026
    assert first["ordered_in_range_agent_ids"] == [
        "veh-2",
        "infra-1",
        "veh-1",
        "veh-3",
    ]
    assert [agent["agent_id"] for agent in first["admitted_agents"]] == [
        "veh-2",
        "infra-1",
        "veh-1",
    ]
    assert first["truncated_agent_ids"] == ["veh-3"]
    assert [
        agent["delay"]["generated_delay_ms"] for agent in first["admitted_agents"]
    ] == [0, 200, 200]
    assert manifest["method_eligibility"] == METHOD_ELIGIBILITY
    assert manifest["method_eligibility"]["status"] == "N/A"
    assert manifest["measurement_boundary"] == {
        "contains_model_outputs": False,
        "contains_evaluator_outputs": False,
        "controlled_result_id": None,
    }
    assert manifest["content_sha256"] == content_sha256(manifest)


def test_validation_trace_requires_fixed_ego_per_scene(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    data_root.mkdir()
    records = [
        _record(
            data_root,
            split="validation",
            sample_id="val-0",
            target_us=1_000_000,
            epoch=None,
            draw_index=0,
        ),
        _record(
            data_root,
            split="validation",
            sample_id="val-1",
            target_us=2_000_000,
            epoch=None,
            draw_index=1,
        ),
    ]
    trace = _trace(records, split="validation")
    trace_path = tmp_path / "val-trace.json"
    _write_trace(trace_path, trace)

    manifest = export_standard_manifest(trace_path, data_root)

    assert manifest["split"] == "validation"
    assert manifest["epoch_ids"] == []
    for record in manifest["records"]:
        assert record["epoch"] is None
        assert record["ego_selection"]["mode"] == "fixed"
        assert record["ego_selection"]["rng_seed"] is None

    changed = copy.deepcopy(trace)
    changed["records"][1]["ego_selection"]["selected_index"] = 0
    changed["records"][1]["available_agents"][0]["distance_to_ego_m"] = 0.0
    changed["records"][1]["available_agents"][1]["distance_to_ego_m"] = 5.0
    changed["records"][1]["ordered_in_range_agent_ids"] = [
        "veh-1",
        "infra-1",
        "veh-2",
        "veh-3",
    ]
    changed["records"][1]["admitted_agents"][0]["agent_id"] = "veh-1"
    changed["records"][1]["admitted_agents"][0]["distance_to_ego_m"] = 0.0
    changed["records"][1]["admitted_agents"][0]["lidar"] = changed["records"][1][
        "admitted_agents"
    ][2]["lidar"]
    changed["records"][1]["admitted_agents"][2]["agent_id"] = "veh-2"
    changed["records"][1]["admitted_agents"][2]["distance_to_ego_m"] = 5.0
    changed["records"][1]["admitted_agents"][2]["lidar"] = trace["records"][1][
        "admitted_agents"
    ][0]["lidar"]
    _rehash(changed)
    changed_path = tmp_path / "changed.json"
    _write_trace(changed_path, changed)

    with pytest.raises(V2XSetStandardError, match="remain fixed"):
        export_standard_manifest(changed_path, data_root)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda trace: trace["selection_policy"].update(communication_range_m=69.0),
            "must be 70.0",
        ),
        (
            lambda trace: trace["selection_policy"].update(
                agent_ordering_rule="unknown loader order"
            ),
            "resolved string",
        ),
        (
            lambda trace: trace["selection_policy"].update(max_cav=1),
            "must be >= 2",
        ),
        (
            lambda trace: trace["delay_policy"].update(nominal_latency_ms=150),
            "one of 0, 200, 300",
        ),
    ],
)
def test_unresolved_or_scope_changing_protocol_is_rejected(
    tmp_path: Path,
    mutation,
    message: str,
) -> None:
    data_root = tmp_path / "data"
    data_root.mkdir()
    trace = _trace(
        [_record(data_root, split="train", sample_id="sample", target_us=1_000_000)]
    )
    mutation(trace)
    _rehash(trace)
    path = tmp_path / "trace.json"
    _write_trace(path, trace)

    with pytest.raises(V2XSetStandardError, match=message):
        export_standard_manifest(path, data_root)


def test_trace_fails_closed_on_order_truncation_and_delay_inconsistency(
    tmp_path: Path,
) -> None:
    data_root = tmp_path / "data"
    data_root.mkdir()
    trace = _trace(
        [_record(data_root, split="train", sample_id="sample", target_us=1_000_000)]
    )

    wrong_order = copy.deepcopy(trace)
    wrong_order["records"][0]["ordered_in_range_agent_ids"] = [
        "infra-1",
        "veh-2",
        "veh-1",
        "veh-3",
    ]
    _rehash(wrong_order)
    path = tmp_path / "wrong-order.json"
    _write_trace(path, wrong_order)
    with pytest.raises(V2XSetStandardError, match="start with the selected Ego"):
        export_standard_manifest(path, data_root)

    wrong_truncation = copy.deepcopy(trace)
    wrong_truncation["records"][0]["truncated_agent_ids"] = []
    _rehash(wrong_truncation)
    path = tmp_path / "wrong-truncation.json"
    _write_trace(path, wrong_truncation)
    with pytest.raises(V2XSetStandardError, match="ordered suffix"):
        export_standard_manifest(path, data_root)

    wrong_delay = copy.deepcopy(trace)
    wrong_delay["records"][0]["admitted_agents"][1]["delay"][
        "arrival_timestamp_us"
    ] -= 1
    _rehash(wrong_delay)
    path = tmp_path / "wrong-delay.json"
    _write_trace(path, wrong_delay)
    with pytest.raises(V2XSetStandardError, match="arrival must equal"):
        export_standard_manifest(path, data_root)


def test_supplemental_100_and_nonuniform_remote_delays_are_explicit(
    tmp_path: Path,
) -> None:
    data_root = tmp_path / "data"
    data_root.mkdir()
    target_us = 1_000_000
    supplemental_record = _record(
        data_root,
        split="train",
        sample_id="supplemental",
        target_us=target_us,
    )
    for agent in supplemental_record["admitted_agents"][1:]:
        agent["delay"] = _delay(
            target_us=target_us,
            delay_ms=100,
            offset=1,
        )
    supplemental = _trace(
        [supplemental_record],
        nominal_latency_ms=100,
        condition_kind="supplemental",
    )
    supplemental_path = tmp_path / "supplemental.json"
    _write_trace(supplemental_path, supplemental)

    supplemental_manifest = export_standard_manifest(
        supplemental_path,
        data_root,
    )
    assert supplemental_manifest["delay_policy"]["condition_kind"] == ("supplemental")
    assert supplemental_manifest["delay_policy"]["nominal_latency_ms"] == 100

    nonuniform_record = _record(
        data_root,
        split="train",
        sample_id="nonuniform",
        target_us=target_us,
    )
    nonuniform_record["admitted_agents"][1]["delay"] = _delay(
        target_us=target_us,
        delay_ms=100,
        offset=1,
    )
    nonuniform = _trace(
        [nonuniform_record],
        nominal_latency_ms=300,
    )
    nonuniform["delay_policy"]["uniform_across_remote_agents"] = False
    _rehash(nonuniform)
    nonuniform_path = tmp_path / "nonuniform.json"
    _write_trace(nonuniform_path, nonuniform)

    nonuniform_manifest = export_standard_manifest(nonuniform_path, data_root)
    assert [
        agent["delay"]["generated_delay_ms"]
        for agent in nonuniform_manifest["records"][0]["admitted_agents"]
    ] == [0, 100, 200]


def test_trace_rejects_payload_mutation_noncanonical_json_and_extra_results(
    tmp_path: Path,
) -> None:
    data_root = tmp_path / "data"
    data_root.mkdir()
    trace = _trace(
        [_record(data_root, split="train", sample_id="sample", target_us=1_000_000)]
    )
    path = tmp_path / "trace.json"
    _write_trace(path, trace)
    payload = trace["records"][0]["admitted_agents"][0]["lidar"]
    (data_root / payload["relative_path"]).write_bytes(b"mutated")
    with pytest.raises(V2XSetStandardError, match="payload size mismatch"):
        export_standard_manifest(path, data_root)

    pretty = tmp_path / "pretty.json"
    pretty.write_text(json.dumps(trace, indent=2), encoding="utf-8")
    with pytest.raises(V2XSetStandardError, match="not canonical"):
        export_standard_manifest(pretty, data_root)

    extra = copy.deepcopy(trace)
    extra["model_results"] = {"bev_ap_07": 99.0}
    _rehash(extra)
    extra_path = tmp_path / "extra.json"
    _write_trace(extra_path, extra)
    with pytest.raises(V2XSetStandardError, match="fields mismatch"):
        export_standard_manifest(extra_path, data_root)


def test_manifest_publication_is_idempotent_and_cannot_upgrade_method_status(
    tmp_path: Path,
) -> None:
    data_root = tmp_path / "data"
    data_root.mkdir()
    trace = _trace(
        [_record(data_root, split="train", sample_id="sample", target_us=1_000_000)]
    )
    path = tmp_path / "trace.json"
    _write_trace(path, trace)
    manifest = export_standard_manifest(path, data_root)
    output = tmp_path / "standard.json"

    write_standard_manifest(output, manifest)
    write_standard_manifest(output, manifest)
    assert output.read_bytes() == canonical_json_bytes(manifest)

    upgraded = copy.deepcopy(manifest)
    upgraded["method_eligibility"]["status"] = "eligible"
    upgraded["content_sha256"] = content_sha256(upgraded)
    with pytest.raises(V2XSetStandardError, match="must remain N/A"):
        validate_standard_manifest(upgraded)

    conflict = copy.deepcopy(manifest)
    conflict["dataset_identity"]["release_id"] = "different-release"
    conflict["content_sha256"] = content_sha256(conflict)
    with pytest.raises(V2XSetStandardError, match="destination conflict"):
        write_standard_manifest(output, conflict)


def test_cli_exports_only_a_manifest_and_reports_explicit_na(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    data_root.mkdir()
    trace = _trace(
        [_record(data_root, split="train", sample_id="sample", target_us=1_000_000)]
    )
    trace_path = tmp_path / "trace.json"
    output = tmp_path / "standard.json"
    _write_trace(trace_path, trace)

    result = subprocess.run(
        [
            sys.executable,
            "tools/resilient_v2x/export_v2xset_standard.py",
            str(trace_path),
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
    assert output.read_bytes() == canonical_json_bytes(manifest)
    assert summary["standard_loader_implemented_in_repository"] is False
    assert summary["resilient_v2x_method_eligibility"] == "N/A"
    assert summary["contains_model_results"] is False
    assert manifest["method_eligibility"]["status"] == "N/A"
