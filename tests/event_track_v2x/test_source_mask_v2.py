"""Source-ablation controls preserve sealed cache and tracker semantics."""
from __future__ import annotations

from dataclasses import FrozenInstanceError, fields

import numpy as np
import pytest

from transvision.models.event_track_v2x.detection_cache_v2 import DetectionCacheV2, canonical
from transvision.models.event_track_v2x.source_mask_v2 import SourceMaskedCacheV2
from transvision.models.event_track_v2x import tracking_v2 as tracking

# Reuse the existing prediction-only fixtures and actual tracker; no GT/data
# fixtures or frozen checkpoint files are needed for these bounded regressions.
from test_tracking_v2 import FixedModel, frame, tracker


def masked(pair, mask):
    return tuple(SourceMaskedCacheV2.from_frame(source, mask) for source in pair)


def short_sequence():
    """Include observations, a late image, missing detections, and expiry."""
    for number, timestamp, offset, empty in [
        (1, 1_000_000, 0., False),
        (2, 1_100_000, .1, False),
        (3, 1_200_000, .2, True),
        (4, 3_300_001, 2.3, True),
        (5, 3_400_001, 2.4, False),
    ]:
        states = [] if empty else [[offset, 0., 1., 2., 4., 2., -np.pi / 2, .2, 0.]]
        vehicle = frame(number=number, time_us=timestamp, states=states,
                        image_us=timestamp + (100_001 if number == 2 else 50_000))
        infrastructure = frame('infrastructure-side', number=number, time_us=timestamp - 20_000,
                               states=states, image_us=timestamp + 40_000)
        yield vehicle, infrastructure


@pytest.mark.parametrize('agent_mask', [None, False, True, 0, 4, -1, 1., '1', np.int64(1), [], {}])
def test_invalid_masks_fail_closed(agent_mask):
    with pytest.raises(ValueError, match='integer in'):
        SourceMaskedCacheV2.from_frame(frame(), agent_mask)


def test_only_original_v2_frame_can_be_wrapped():
    with pytest.raises(TypeError, match='original DetectionCacheV2'):
        SourceMaskedCacheV2.from_frame(object(), 1)
    view = SourceMaskedCacheV2.from_frame(frame(), 1)
    with pytest.raises(TypeError, match='original DetectionCacheV2'):
        SourceMaskedCacheV2.from_frame(view, 3)


@pytest.mark.parametrize('side,source_bit', [('vehicle-side', 1), ('infrastructure-side', 2)])
@pytest.mark.parametrize('agent_mask', [1, 2, 3])
def test_view_preserves_source_payload_identity_and_availability(side, source_bit, agent_mask):
    original = frame(side, image_us=1_100_000)
    digest = original.digest()
    view = SourceMaskedCacheV2.from_frame(original, agent_mask)
    assert isinstance(view, DetectionCacheV2)
    assert view.active_agent_mask == agent_mask
    assert view.metadata == original.metadata
    assert view.metadata['agent_mask'] == source_bit
    assert view.information_timestamp_us == original.information_timestamp_us
    assert view.count == original.count
    for field in fields(DetectionCacheV2):
        value, source_value = getattr(view, field.name), getattr(original, field.name)
        if isinstance(value, np.ndarray):
            assert value.dtype == source_value.dtype and value.shape == source_value.shape
            assert value.tobytes() == source_value.tobytes()
            assert not value.flags.writeable and not source_value.flags.writeable
        else:
            assert value == source_value
    assert view.digest() == original.digest() == digest
    for timestamp in [1_000_000, 1_099_999, 1_100_000, 2_000_000]:
        expected = original.available_at(timestamp) and bool(source_bit & agent_mask)
        assert view.available_at(timestamp) is expected
    # Reading the source view cannot suppress or otherwise mutate the original.
    assert original.available_at(1_100_000) is True
    assert not hasattr(original, 'active_agent_mask')


def test_view_mask_fields_metadata_and_arrays_are_immutable():
    original = frame()
    digest = original.digest()
    view = SourceMaskedCacheV2.from_frame(original, 1)
    for name, replacement in [('active_agent_mask', 3), ('metadata_json', b'{}'), ('states', np.zeros((0, 9)))]:
        with pytest.raises(FrozenInstanceError):
            setattr(view, name, replacement)
    assert not hasattr(view, '__dict__')
    for field in fields(DetectionCacheV2):
        value = getattr(view, field.name)
        if isinstance(value, np.ndarray):
            with pytest.raises(ValueError):
                value.flat[0] = 0
            with pytest.raises(ValueError):
                value.setflags(write=True)
    metadata_copy = view.metadata
    metadata_copy['agent_mask'] = 3
    assert view.metadata['agent_mask'] == original.metadata['agent_mask'] == 1
    assert view.digest() == original.digest() == digest


@pytest.mark.parametrize('mask', [1, 2])
def test_disabled_source_never_reaches_feature_encoder_or_model(monkeypatch, mask):
    calls = []
    real_features = tracking.frame_features

    def checked_features(source, *args, **kwargs):
        calls.append(source.metadata['side'])
        assert source.metadata['agent_mask'] & mask
        return real_features(source, *args, **kwargs)

    monkeypatch.setattr(tracking, 'frame_features', checked_features)
    model = FixedModel()
    result, audit = tracker(model).step(*masked((frame(), frame('infrastructure-side')), mask))
    active = [bool(mask & 1), bool(mask & 2)]
    assert result['source_available'] == active
    assert result['selected_detections'] == [int(x) for x in active]
    assert calls == (['vehicle-side'] if mask == 1 else ['infrastructure-side'])
    assert [tuple(value.shape) for value in model.inputs[0]] == [(1, int(active[0]), 203), (1, int(active[1]), 203)]
    assert audit['hypotheses'] == [{'pairs': [], 'unmatched_left': list(range(int(active[0]))),
                                    'unmatched_right': list(range(int(active[1]))), 'weight': 1., 'energy': 0.}]


@pytest.mark.parametrize('mask', [1, 2])
def test_single_source_tracking_is_independent_of_finite_model_logits(mask):
    models = [FixedModel(pair=100., dustbin=-100.), FixedModel(pair=-100., dustbin=100.),
              FixedModel(pair=.27, dustbin=-.91)]
    trackers = [tracker(model) for model in models]
    for pair in short_sequence():
        outputs = [tuple(canonical(item) for item in item_tracker.step(*masked(pair, mask)))
                   for item_tracker in trackers]
        assert outputs[0] == outputs[1] == outputs[2]
    assert len({item_tracker.commit_hash for item_tracker in trackers}) == 1


def test_all_sources_mask_preserves_full_short_sequence_canonical_bytes():
    original_tracker, masked_tracker = tracker(), tracker()
    for pair in short_sequence():
        original = original_tracker.step(*pair)
        wrapped = masked_tracker.step(*masked(pair, 3))
        assert tuple(canonical(item) for item in original) == tuple(canonical(item) for item in wrapped)
        assert original_tracker.commit_hash == masked_tracker.commit_hash
    for expected, actual in zip(original_tracker.model.inputs, masked_tracker.model.inputs):
        for source_expected, source_actual in zip(expected, actual):
            np.testing.assert_array_equal(source_expected.numpy(), source_actual.numpy())
