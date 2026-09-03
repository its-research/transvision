from __future__ import annotations

from dataclasses import dataclass
import json

import pytest

from tools.event_track_v2x import verify_run_result_registry
from tools.event_track_v2x import verify_supplemental_registry
from tools.event_track_v2x import verify_validation_registry
from transvision.models.event_track_v2x.network import (
    NetworkConditionId,
    condition_plan_v1,
)
from transvision.models.event_track_v2x.network_disturbance import (
    ConditionInputManifestV1,
)
from transvision.models.event_track_v2x.wire import canonical_json_bytes


@dataclass(frozen=True)
class _BaselineSummary:
    method_id: str = "baseline-a"
    robust_assa_at_64k: float = 0.61
    clean_amota: float = 0.70
    clean_hota: float = 0.69
    failure_count: int = 0


@dataclass(frozen=True)
class _SchedulerSummary:
    scheduler_id: str = "fifo-aoi"
    robust_assa_at_64k: float = 0.60
    actual_byte_auc: float = 0.59
    failure_count: int = 1


class _Registry:
    cells = (object(), object())

    def digest(self) -> str:
        return "a" * 64


class _SupplementalRegistry:
    cells = (object(), object(), object())

    def digest(self) -> str:
        return "d" * 64


class _Plan:
    content_sha256 = "b" * 64


class _MeasuredReceipt:
    content_sha256 = "a" * 64


class _Metrics:
    def digest(self) -> str:
        return "c" * 64

    def canonical_bytes(self) -> bytes:
        return b'{"kind":"publication_metrics_v1"}'


def test_verify_validation_registry_formats_canonical_summary(
    monkeypatch, tmp_path, capsys
) -> None:
    plan_path = tmp_path / "plan.json"
    registry_path = tmp_path / "validation.json"
    receipt_path = tmp_path / "measured-receipt.json"
    manifest_inventory_path = tmp_path / "condition-manifests.json"
    plan_path.write_bytes(b"canonical-plan")
    registry_path.write_bytes(b"canonical-validation")
    receipt_path.write_bytes(b"canonical-measured-receipt")
    manifest_inventory_path.write_bytes(b"canonical-manifest-inventory")
    plan = _Plan()
    registry = _Registry()
    receipt = _MeasuredReceipt()
    manifests = {"f" * 64: object()}
    events = []
    summaries = type(
        "Summaries",
        (),
        {
            "baselines": (_BaselineSummary(),),
            "schedulers": (_SchedulerSummary(),),
        },
    )()
    monkeypatch.setattr(
        verify_validation_registry,
        "decode_plan",
        lambda data: plan if data == b"canonical-plan" else None,
    )
    monkeypatch.setattr(
        verify_validation_registry,
        "decode_validation_result_registry",
        lambda data: registry if data == b"canonical-validation" else None,
    )
    monkeypatch.setattr(
        verify_validation_registry,
        "decode_measured_trace_receipt",
        lambda data: receipt if data == b"canonical-measured-receipt" else None,
    )
    monkeypatch.setattr(
        verify_validation_registry,
        "decode_condition_input_manifest_inventory_v1",
        lambda data: manifests if data == b"canonical-manifest-inventory" else None,
    )
    monkeypatch.setattr(
        verify_validation_registry,
        "validate_validation_result_external_inputs_v1",
        lambda value, value_plan, **kwargs: events.append(
            ("validate", value, value_plan, kwargs)
        ),
    )
    monkeypatch.setattr(
        verify_validation_registry,
        "derive_validation_summaries_v1",
        lambda value: (
            events.append(("derive", value)) or summaries if value is registry else None
        ),
    )

    assert (
        verify_validation_registry.main(
            [
                "--plan",
                str(plan_path),
                "--registry",
                str(registry_path),
                "--measured-trace-receipt",
                str(receipt_path),
                "--condition-input-manifest-inventory",
                str(manifest_inventory_path),
            ]
        )
        == 0
    )
    assert events == [
        (
            "validate",
            registry,
            plan,
            {
                "condition_input_manifests": manifests,
                "measured_trace_receipt": receipt,
            },
        ),
        ("derive", registry),
    ]
    output = capsys.readouterr().out
    document = json.loads(output)
    assert (
        output
        == json.dumps(
            document, ensure_ascii=False, separators=(",", ":"), sort_keys=True
        )
        + "\n"
    )
    assert document["registry_sha256"] == "a" * 64
    assert document["cell_count"] == 2
    assert document["experiment_plan_sha256"] == "b" * 64
    assert document["measured_trace_receipt_sha256"] == "a" * 64
    assert document["condition_input_manifest_count"] == 1
    assert document["baseline_summaries"][0]["method_id"] == "baseline-a"
    assert document["scheduler_summaries"][0]["actual_byte_auc"] == 0.59


def test_verify_run_registry_derives_and_writes_metrics(
    monkeypatch, tmp_path, capsys
) -> None:
    plan_path = tmp_path / "plan.json"
    registry_path = tmp_path / "registry.json"
    receipt_path = tmp_path / "measured-receipt.json"
    manifest_inventory_path = tmp_path / "condition-manifests.json"
    metrics_path = tmp_path / "out" / "metrics.json"
    plan_path.write_bytes(b"canonical-plan")
    registry_path.write_bytes(b"canonical-registry")
    receipt_path.write_bytes(b"canonical-measured-receipt")
    manifest_inventory_path.write_bytes(b"canonical-manifest-inventory")
    plan = _Plan()
    registry = _Registry()
    receipt = _MeasuredReceipt()
    manifests = {"f" * 64: object()}
    metrics = _Metrics()
    validated = []
    monkeypatch.setattr(
        verify_run_result_registry,
        "decode_plan",
        lambda data: plan if data == b"canonical-plan" else None,
    )
    monkeypatch.setattr(
        verify_run_result_registry,
        "decode_run_result_registry",
        lambda data: registry if data == b"canonical-registry" else None,
    )
    monkeypatch.setattr(
        verify_run_result_registry,
        "decode_measured_trace_receipt",
        lambda data: receipt if data == b"canonical-measured-receipt" else None,
    )
    monkeypatch.setattr(
        verify_run_result_registry,
        "decode_condition_input_manifest_inventory_v1",
        lambda data: manifests if data == b"canonical-manifest-inventory" else None,
    )
    monkeypatch.setattr(
        verify_run_result_registry,
        "validate_run_result_external_inputs_v1",
        lambda value, value_plan, **kwargs: validated.append(
            (value, value_plan, kwargs)
        ),
    )
    monkeypatch.setattr(
        verify_run_result_registry,
        "derive_publication_metrics_v1",
        lambda value, value_plan: metrics,
    )

    assert (
        verify_run_result_registry.main(
            [
                "--plan",
                str(plan_path),
                "--registry",
                str(registry_path),
                "--measured-trace-receipt",
                str(receipt_path),
                "--condition-input-manifest-inventory",
                str(manifest_inventory_path),
                "--metrics-output",
                str(metrics_path),
            ]
        )
        == 0
    )
    assert validated == [
        (
            registry,
            plan,
            {
                "condition_input_manifests": manifests,
                "measured_trace_receipt": receipt,
            },
        )
    ]
    assert metrics_path.read_bytes() == metrics.canonical_bytes()
    document = json.loads(capsys.readouterr().out)
    assert document == {
        "cell_count": 2,
        "condition_input_manifest_count": 1,
        "experiment_plan_sha256": "b" * 64,
        "kind": "run_result_registry_verification_v1",
        "measured_trace_receipt_sha256": "a" * 64,
        "publication_metrics_sha256": "c" * 64,
        "run_result_registry_sha256": "a" * 64,
        "schema_version": 1,
    }


def test_metrics_output_refuses_to_overwrite_existing_file(
    monkeypatch, tmp_path
) -> None:
    path = tmp_path / "metrics.json"
    path.write_bytes(b"existing")

    try:
        verify_run_result_registry._write_new(path, b"replacement")
    except FileExistsError:
        pass
    else:  # pragma: no cover - documents the fail-closed overwrite contract.
        raise AssertionError("existing metrics file was overwritten")
    assert path.read_bytes() == b"existing"


def test_verify_supplemental_registry_wires_strict_load_and_validator(
    monkeypatch, tmp_path, capsys
) -> None:
    plan_path = tmp_path / "plan.json"
    registry_path = tmp_path / "supplemental.json"
    receipt_path = tmp_path / "measured-receipt.json"
    manifest_inventory_path = tmp_path / "condition-manifests.json"
    plan_path.write_bytes(b"canonical-plan")
    registry_path.write_bytes(b"canonical-supplemental-registry")
    receipt_path.write_bytes(b"canonical-measured-receipt")
    manifest_inventory_path.write_bytes(b"canonical-manifest-inventory")
    plan = _Plan()
    registry = _SupplementalRegistry()
    receipt = _MeasuredReceipt()
    manifests = {"f" * 64: object()}
    validated = []
    monkeypatch.setattr(
        verify_supplemental_registry,
        "decode_plan",
        lambda data: plan if data == b"canonical-plan" else None,
    )
    monkeypatch.setattr(
        verify_supplemental_registry,
        "decode_paper_supplemental_registry",
        lambda data: registry if data == b"canonical-supplemental-registry" else None,
    )
    monkeypatch.setattr(
        verify_supplemental_registry,
        "decode_measured_trace_receipt",
        lambda data: receipt if data == b"canonical-measured-receipt" else None,
    )
    monkeypatch.setattr(
        verify_supplemental_registry,
        "_decode_condition_input_manifest_inventory",
        lambda data: manifests if data == b"canonical-manifest-inventory" else None,
    )
    monkeypatch.setattr(
        verify_supplemental_registry,
        "validate_paper_supplemental_registry_v1",
        lambda value, value_plan, **kwargs: validated.append(
            (value, value_plan, kwargs)
        ),
    )

    assert (
        verify_supplemental_registry.main(
            [
                "--plan",
                str(plan_path),
                "--registry",
                str(registry_path),
                "--measured-trace-receipt",
                str(receipt_path),
                "--condition-input-manifest-inventory",
                str(manifest_inventory_path),
                "--raw-sensor-detector-cache-sha256",
                "e" * 64,
            ]
        )
        == 0
    )
    assert validated == [
        (
            registry,
            plan,
            {
                "condition_input_manifests": manifests,
                "measured_trace_receipt": receipt,
                "raw_sensor_detector_cache_sha256": "e" * 64,
            },
        )
    ]
    output = capsys.readouterr().out
    document = json.loads(output)
    assert (
        output
        == json.dumps(
            document, ensure_ascii=False, separators=(",", ":"), sort_keys=True
        )
        + "\n"
    )
    assert document == {
        "cell_count": 3,
        "condition_input_manifest_count": 1,
        "experiment_plan_sha256": "b" * 64,
        "kind": "paper_supplemental_registry_verification_v1",
        "measured_trace_receipt_sha256": "a" * 64,
        "nonranking_scenario_ids": [
            "full_send_diagnostic",
            "synchronous_oracle_upper_bound",
        ],
        "primary_sota_gate_eligible": False,
        "registry_sha256": "d" * 64,
        "schema_version": 1,
    }


def test_verify_supplemental_registry_does_not_emit_success_on_failure(
    monkeypatch, tmp_path, capsys
) -> None:
    plan_path = tmp_path / "plan.json"
    registry_path = tmp_path / "supplemental.json"
    receipt_path = tmp_path / "measured-receipt.json"
    manifest_inventory_path = tmp_path / "condition-manifests.json"
    plan_path.write_bytes(b"canonical-plan")
    registry_path.write_bytes(b"canonical-supplemental-registry")
    receipt_path.write_bytes(b"canonical-measured-receipt")
    manifest_inventory_path.write_bytes(b"canonical-manifest-inventory")
    monkeypatch.setattr(verify_supplemental_registry, "decode_plan", lambda _: _Plan())
    monkeypatch.setattr(
        verify_supplemental_registry,
        "decode_paper_supplemental_registry",
        lambda _: _SupplementalRegistry(),
    )
    monkeypatch.setattr(
        verify_supplemental_registry,
        "decode_measured_trace_receipt",
        lambda _: _MeasuredReceipt(),
    )
    monkeypatch.setattr(
        verify_supplemental_registry,
        "_decode_condition_input_manifest_inventory",
        lambda _: {"f" * 64: object()},
    )
    monkeypatch.setattr(
        verify_supplemental_registry,
        "validate_paper_supplemental_registry_v1",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            ValueError("supplemental registry invalid")
        ),
    )

    with pytest.raises(ValueError, match="supplemental registry invalid"):
        verify_supplemental_registry.main(
            [
                "--plan",
                str(plan_path),
                "--registry",
                str(registry_path),
                "--measured-trace-receipt",
                str(receipt_path),
                "--condition-input-manifest-inventory",
                str(manifest_inventory_path),
                "--raw-sensor-detector-cache-sha256",
                "e" * 64,
            ]
        )
    assert capsys.readouterr().out == ""


def test_supplemental_cli_condition_manifest_inventory_loader_is_strict() -> None:
    condition = NetworkConditionId.C6
    manifest = ConditionInputManifestV1(
        run_id="attempt-0001",
        condition_id=condition,
        network_trace_sha256="1" * 64,
        condition_plan_sha256=condition_plan_v1(condition).content_sha256,
        run_config_sha256="2" * 64,
        detection_cache_sha256="3" * 64,
        evidence_sha256s=("4" * 64,),
    )
    canonical = canonical_json_bytes({manifest.content_sha256: manifest.to_primitive()})
    assert verify_supplemental_registry._decode_condition_input_manifest_inventory(
        canonical
    ) == {manifest.content_sha256: manifest}

    wrong_key = canonical_json_bytes({"f" * 64: manifest.to_primitive()})
    with pytest.raises(ValueError, match="key does not match content"):
        verify_supplemental_registry._decode_condition_input_manifest_inventory(
            wrong_key
        )
    duplicate = canonical.replace(
        b'{"' + manifest.content_sha256.encode() + b'":',
        b'{"'
        + manifest.content_sha256.encode()
        + b'":{},"'
        + manifest.content_sha256.encode()
        + b'":',
        1,
    )
    with pytest.raises(ValueError, match="duplicate JSON key"):
        verify_supplemental_registry._decode_condition_input_manifest_inventory(
            duplicate
        )
