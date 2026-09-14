from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from transvision.models.event_track_v2x.learning_contracts import (
    ASSOCIATION_CHECKPOINT_NAME_V1,
    ASSOCIATION_MANIFEST_NAME_V1,
    AssociationLearningContractError,
    AssociationTrainingArtifactManifestV1,
    AssociationTrainingConfigV1,
    decode_association_training_manifest_v1,
    verify_association_training_artifact,
)


def _manifest(
    checkpoint: bytes = b"pytorch-checkpoint",
) -> AssociationTrainingArtifactManifestV1:
    return AssociationTrainingArtifactManifestV1(
        fold_id=2,
        training_seed=1337,
        development_manifest_sha256="a" * 64,
        official_split_sha256="b" * 64,
        cohort_sha256="c" * 64,
        config_sha256=AssociationTrainingConfigV1().content_sha256,
        checkpoint_sha256=hashlib.sha256(checkpoint).hexdigest(),
        appearance_source="frozen_rgb_projection_128_canary_v1",
        covariance_source="spd_gt_diagonal_proxy_v1",
        train_sequence_count=37,
        validation_sequence_count=9,
        train_frame_count=74,
        validation_frame_count=18,
        train_object_count=350,
        validation_object_count=82,
        metrics={
            "train_loss": 0.4,
            "validation_assignment_accuracy": 0.7,
            "validation_loss": 0.5,
            "validation_pair_accuracy": 0.8,
        },
    )


def test_manifest_is_canonical_and_verifies_checkpoint(tmp_path: Path) -> None:
    checkpoint = b"pytorch-checkpoint"
    manifest = _manifest(checkpoint)
    root = tmp_path / "artifact"
    root.mkdir()
    (root / ASSOCIATION_CHECKPOINT_NAME_V1).write_bytes(checkpoint)
    (root / ASSOCIATION_MANIFEST_NAME_V1).write_bytes(manifest.canonical_bytes)

    decoded = decode_association_training_manifest_v1(manifest.canonical_bytes)
    verified = verify_association_training_artifact(root)

    assert decoded.content_sha256 == manifest.content_sha256
    assert verified.checkpoint_sha256 == hashlib.sha256(checkpoint).hexdigest()
    assert verified.gt_supervised_development_only is True
    assert verified.ranking_eligible is False
    assert verified.formal_evidence is False
    assert verified.paper_registry_writable is False


@pytest.mark.parametrize("sealed_split", ["val", "test", "test_A"])
def test_manifest_fails_closed_for_every_sealed_spd_split(sealed_split: str) -> None:
    values = _manifest().payload()
    values["split_name"] = sealed_split
    values.pop("kind")
    values.pop("schema_version")

    with pytest.raises(AssociationLearningContractError, match="fail-closed"):
        AssociationTrainingArtifactManifestV1(**values)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("gt_supervised_development_only", False),
        ("ranking_eligible", True),
        ("formal_evidence", True),
        ("paper_registry_writable", True),
    ],
)
def test_manifest_cannot_cross_development_evidence_boundary(
    field: str, value: bool
) -> None:
    document = json.loads(_manifest().canonical_bytes)
    document[field] = value
    payload = {key: item for key, item in document.items() if key != "content_sha256"}
    document["content_sha256"] = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()

    with pytest.raises(
        AssociationLearningContractError, match="outside paper registries"
    ):
        decode_association_training_manifest_v1(
            json.dumps(document, sort_keys=True, separators=(",", ":")).encode()
        )


def test_artifact_verifier_rejects_tampering_and_extra_registry_file(
    tmp_path: Path,
) -> None:
    checkpoint = b"pytorch-checkpoint"
    root = tmp_path / "artifact"
    root.mkdir()
    (root / ASSOCIATION_CHECKPOINT_NAME_V1).write_bytes(checkpoint + b"tampered")
    (root / ASSOCIATION_MANIFEST_NAME_V1).write_bytes(
        _manifest(checkpoint).canonical_bytes
    )
    with pytest.raises(AssociationLearningContractError, match="checkpoint SHA"):
        verify_association_training_artifact(root)

    (root / ASSOCIATION_CHECKPOINT_NAME_V1).write_bytes(checkpoint)
    (root / "paper-result-registry.json").write_text("{}", encoding="utf-8")
    with pytest.raises(AssociationLearningContractError, match="unexpected files"):
        verify_association_training_artifact(root)
