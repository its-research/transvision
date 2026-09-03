from dataclasses import replace

import pytest

from transvision.models.event_track_v2x.dataset_release import (
    DatasetReleaseError,
    DatasetReleaseReceiptV1,
    OFFICIAL_RELEASE_STATUS_V1,
    decode_dataset_release_receipt,
    require_scientific_clearml_verification,
)


def _receipt() -> DatasetReleaseReceiptV1:
    return DatasetReleaseReceiptV1(
        receipt_id="spd-val-release-v1",
        dataset_id="v2x-seq-spd",
        split_name="val",
        dataset_manifest_sha256="1" * 64,
        official_inventory_sha256="2" * 64,
        license_evidence_sha256="7" * 64,
        split_sha256="3" * 64,
        cohort_sha256="4" * 64,
        frame_contract_sha256="5" * 64,
        sequence_ids=tuple(f"sequence-{index:02d}" for index in range(21)),
        clearml_dataset_id="clearml-dataset-id",
        clearml_project_id="clearml-project-id",
        clearml_version="formal-v1",
        cold_cache_verification_sha256="6" * 64,
        release_identity_status=OFFICIAL_RELEASE_STATUS_V1,
        license_use_authorized=True,
        source_bytes_verified=True,
        cold_cache_verified=True,
        scientific_claims_allowed=True,
    )


def test_dataset_release_receipt_round_trip_and_hash() -> None:
    receipt = _receipt()
    assert decode_dataset_release_receipt(receipt.canonical_bytes) == receipt
    assert receipt.content_sha256 == _receipt().content_sha256


def test_dataset_release_receipt_rejects_unverified_or_wrong_cohort() -> None:
    with pytest.raises(DatasetReleaseError, match="scientific claim"):
        replace(_receipt(), scientific_claims_allowed=False)
    with pytest.raises(DatasetReleaseError, match="requires verified"):
        replace(_receipt(), license_use_authorized=False)
    with pytest.raises(DatasetReleaseError, match="21 sequences"):
        replace(_receipt(), sequence_ids=_receipt().sequence_ids[:-1])
    with pytest.raises(DatasetReleaseError, match="excludes"):
        replace(_receipt(), split_name="test")
    with pytest.raises(DatasetReleaseError, match="independently authenticated"):
        replace(_receipt(), release_identity_status="unverified-source-bytes")


def test_spd_train_release_is_allowed_for_oof_development() -> None:
    receipt = replace(
        _receipt(),
        receipt_id="spd-train-release-v1",
        split_name="train",
        sequence_ids=tuple(f"sequence-{index:02d}" for index in range(46)),
    )
    assert receipt.split_name == "train"


def test_current_clearml_smoke_output_cannot_become_scientific_receipt() -> None:
    with pytest.raises(DatasetReleaseError, match="does not authorize"):
        require_scientific_clearml_verification(
            {
                "byte_readback_verified": True,
                "cold_cache_verified": True,
                "scientific_claims_allowed": False,
            }
        )
    digest = require_scientific_clearml_verification(
        {
            "byte_readback_verified": True,
            "cold_cache_verified": True,
            "scientific_claims_allowed": True,
        }
    )
    assert len(digest) == 64
