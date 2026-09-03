from collections import Counter
from dataclasses import replace
import hashlib

import pytest

from transvision.models.event_track_v2x.development_split import (
    DEVELOPMENT_FOLD_COUNT_V1,
    DEVELOPMENT_FOLD_SALT_V1,
    DEVELOPMENT_FOLD_SIZES_V1,
    DevelopmentFoldV1,
    DevelopmentSplitError,
    DevelopmentSplitManifestV1,
    build_development_split_manifest_v1,
    decode_development_split_manifest_v1,
)
from transvision.models.event_track_v2x.wire import canonical_json_bytes


SEQUENCES = tuple(f"sequence-{index:02d}" for index in range(46))


def _manifest() -> DevelopmentSplitManifestV1:
    return build_development_split_manifest_v1(
        reversed(SEQUENCES),
        split_sha256="a" * 64,
    )


def _independent_rank(sequence_id: str, split_sha256: str) -> tuple[str, str]:
    payload = {
        "salt": DEVELOPMENT_FOLD_SALT_V1,
        "sequence_id": sequence_id,
        "split_sha256": split_sha256,
    }
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest(), sequence_id


def test_hash_sort_round_robin_is_deterministic_and_order_invariant() -> None:
    input_ids = list(reversed(SEQUENCES))
    original_input = input_ids.copy()
    first = build_development_split_manifest_v1(
        input_ids,
        split_sha256="a" * 64,
    )
    second = build_development_split_manifest_v1(
        SEQUENCES,
        split_sha256="a" * 64,
    )

    assert input_ids == original_input
    assert first == second
    ranked = tuple(
        sorted(SEQUENCES, key=lambda item: _independent_rank(item, "a" * 64))
    )
    expected_held_out = tuple(
        tuple(sorted(ranked[fold_id::DEVELOPMENT_FOLD_COUNT_V1]))
        for fold_id in range(DEVELOPMENT_FOLD_COUNT_V1)
    )
    assert tuple(fold.held_out_sequence_ids for fold in first.folds) == (
        expected_held_out
    )
    assert tuple(len(values) for values in expected_held_out) == (
        DEVELOPMENT_FOLD_SIZES_V1
    )


def test_each_sequence_is_held_out_once_and_every_fold_is_a_partition() -> None:
    expected = _manifest()
    held_out_counts = Counter(
        sequence_id
        for fold in expected.folds
        for sequence_id in fold.held_out_sequence_ids
    )

    assert held_out_counts == Counter({sequence_id: 1 for sequence_id in SEQUENCES})
    assert all(
        set(fold.fit_sequence_ids).isdisjoint(fold.held_out_sequence_ids)
        and set(fold.fit_sequence_ids) | set(fold.held_out_sequence_ids)
        == set(SEQUENCES)
        for fold in expected.folds
    )
    assert all(
        expected.fold_for_sequence(sequence_id) == fold.fold_id
        for fold in expected.folds
        for sequence_id in fold.held_out_sequence_ids
    )
    assert expected.payload()["fold_salt"] == DEVELOPMENT_FOLD_SALT_V1


def test_split_sha_is_bound_into_assignments_and_manifest_digest() -> None:
    first = _manifest()
    changed = build_development_split_manifest_v1(
        SEQUENCES,
        split_sha256="b" * 64,
    )

    assert changed.split_sha256 != first.split_sha256
    assert changed.folds != first.folds
    assert changed.content_sha256 != first.content_sha256


def test_development_split_round_trips_canonical_bytes() -> None:
    expected = _manifest()
    assert expected.canonical_bytes == canonical_json_bytes(
        expected.sealed_document()
    )
    assert decode_development_split_manifest_v1(expected.canonical_bytes) == expected


def test_decoder_rejects_tampering_and_duplicate_json_keys() -> None:
    expected = _manifest()
    with pytest.raises(DevelopmentSplitError, match="content SHA-256 mismatch"):
        decode_development_split_manifest_v1(
            expected.canonical_bytes.replace(b"sequence-00", b"sequence-99", 1)
        )

    duplicate_key = expected.canonical_bytes.replace(
        b'{"content_sha256":',
        b'{"content_sha256":"'
        + b"0" * 64
        + b'","content_sha256":',
        1,
    )
    with pytest.raises(DevelopmentSplitError, match="duplicate JSON key"):
        decode_development_split_manifest_v1(duplicate_key)


def test_manifest_rejects_missing_duplicate_and_wrong_fold_data() -> None:
    expected = _manifest()

    missing_field = expected.sealed_document()
    del missing_field["split_name"]
    with pytest.raises(DevelopmentSplitError, match="missing or unknown fields"):
        DevelopmentSplitManifestV1.from_mapping(missing_field)

    with pytest.raises(DevelopmentSplitError, match="exactly 46"):
        build_development_split_manifest_v1(
            SEQUENCES[:-1],
            split_sha256="a" * 64,
        )
    with pytest.raises(DevelopmentSplitError, match="unique and sorted"):
        build_development_split_manifest_v1(
            (*SEQUENCES[:-1], SEQUENCES[-2]),
            split_sha256="a" * 64,
        )

    wrong_fold_id = replace(expected.folds[0], fold_id=1)
    with pytest.raises(DevelopmentSplitError, match="canonical fold IDs"):
        replace(expected, folds=(wrong_fold_id, *expected.folds[1:]))

    bad_fold = replace(
        expected.folds[0],
        fit_sequence_ids=expected.folds[0].fit_sequence_ids[:-1],
    )
    with pytest.raises(DevelopmentSplitError, match="fixed hash-sort"):
        replace(expected, folds=(bad_fold, *expected.folds[1:]))

    first_held_out = expected.folds[0].held_out_sequence_ids
    second_held_out = expected.folds[1].held_out_sequence_ids
    moved = first_held_out[0]
    reassigned_first = tuple(sorted(first_held_out[1:]))
    reassigned_second = tuple(sorted((*second_held_out, moved)))
    wrong_assignment = (
        DevelopmentFoldV1(
            fold_id=0,
            fit_sequence_ids=tuple(sorted(set(SEQUENCES) - set(reassigned_first))),
            held_out_sequence_ids=reassigned_first,
        ),
        DevelopmentFoldV1(
            fold_id=1,
            fit_sequence_ids=tuple(sorted(set(SEQUENCES) - set(reassigned_second))),
            held_out_sequence_ids=reassigned_second,
        ),
        *expected.folds[2:],
    )
    with pytest.raises(DevelopmentSplitError, match="fixed hash-sort"):
        replace(
            expected,
            folds=wrong_assignment,
        )
