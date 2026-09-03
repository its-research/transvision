from __future__ import annotations

import json

import pytest

from tools.event_track_v2x import verify_ablation_registry


class _Plan:
    content_sha256 = "a" * 64


class _Registry:
    cells = (object(), object())

    def digest(self) -> str:
        return "b" * 64


class _Receipt:
    content_sha256 = "c" * 64


def _paths(tmp_path):
    plan = tmp_path / "plan.json"
    registry = tmp_path / "ablation.json"
    receipt = tmp_path / "measured-receipt.json"
    manifests = tmp_path / "condition-manifests.json"
    plan.write_bytes(b"canonical-plan")
    registry.write_bytes(b"canonical-ablation")
    receipt.write_bytes(b"canonical-receipt")
    manifests.write_bytes(b"canonical-manifests")
    return plan, registry, receipt, manifests


def test_ablation_cli_loads_all_external_inputs_and_calls_formal_validator(
    monkeypatch, tmp_path, capsys
) -> None:
    paths = _paths(tmp_path)
    plan = _Plan()
    registry = _Registry()
    receipt = _Receipt()
    manifests = {"d" * 64: object()}
    validated = []
    monkeypatch.setattr(
        verify_ablation_registry,
        "decode_plan",
        lambda data: plan if data == b"canonical-plan" else None,
    )
    monkeypatch.setattr(
        verify_ablation_registry,
        "decode_ablation_result_registry",
        lambda data: registry if data == b"canonical-ablation" else None,
    )
    monkeypatch.setattr(
        verify_ablation_registry,
        "decode_measured_trace_receipt",
        lambda data: receipt if data == b"canonical-receipt" else None,
    )
    monkeypatch.setattr(
        verify_ablation_registry,
        "decode_condition_input_manifest_inventory_v1",
        lambda data: manifests if data == b"canonical-manifests" else None,
    )
    monkeypatch.setattr(
        verify_ablation_registry,
        "validate_ablation_result_registry_v1",
        lambda value, value_plan, **kwargs: validated.append(
            (value, value_plan, kwargs)
        ),
    )

    assert verify_ablation_registry.main(
        [
            "--plan",
            str(paths[0]),
            "--registry",
            str(paths[1]),
            "--measured-trace-receipt",
            str(paths[2]),
            "--condition-input-manifest-inventory",
            str(paths[3]),
        ]
    ) == 0
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
    output = capsys.readouterr().out
    document = json.loads(output)
    assert output == json.dumps(
        document, ensure_ascii=False, separators=(",", ":"), sort_keys=True
    ) + "\n"
    assert document == {
        "cell_count": 2,
        "condition_input_manifest_count": 1,
        "experiment_plan_sha256": "a" * 64,
        "kind": "ablation_result_registry_verification_v1",
        "measured_trace_receipt_sha256": "c" * 64,
        "primary_sota_gate_eligible": False,
        "registry_sha256": "b" * 64,
        "schema_version": 1,
    }


def test_ablation_cli_does_not_emit_success_when_formal_validation_fails(
    monkeypatch, tmp_path, capsys
) -> None:
    paths = _paths(tmp_path)
    monkeypatch.setattr(verify_ablation_registry, "decode_plan", lambda _: _Plan())
    monkeypatch.setattr(
        verify_ablation_registry,
        "decode_ablation_result_registry",
        lambda _: _Registry(),
    )
    monkeypatch.setattr(
        verify_ablation_registry,
        "decode_measured_trace_receipt",
        lambda _: _Receipt(),
    )
    monkeypatch.setattr(
        verify_ablation_registry,
        "decode_condition_input_manifest_inventory_v1",
        lambda _: {"d" * 64: object()},
    )
    monkeypatch.setattr(
        verify_ablation_registry,
        "validate_ablation_result_registry_v1",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            ValueError("formal ablation registry invalid")
        ),
    )

    with pytest.raises(ValueError, match="formal ablation registry invalid"):
        verify_ablation_registry.main(
            [
                "--plan",
                str(paths[0]),
                "--registry",
                str(paths[1]),
                "--measured-trace-receipt",
                str(paths[2]),
                "--condition-input-manifest-inventory",
                str(paths[3]),
            ]
        )
    assert capsys.readouterr().out == ""
