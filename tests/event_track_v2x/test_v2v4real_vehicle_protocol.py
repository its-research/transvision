"""Explicit native vehicle semantics without rewriting historical Car evidence."""
import copy
import json
import os
from dataclasses import asdict

import numpy as np
import pytest

from transvision.models.event_track_v2x.paper_protocol import PaperProtocol, require_same_protocol
from transvision.models.event_track_v2x.paper_evaluation_policy import (
    VEHICLE_PROTOCOL, NATIVE_VEHICLE_SELECTION, vehicle_binding, require_vehicle_binding,
)
from transvision.models.event_track_v2x.v2v4real_ground_truth import (
    prepare_frame, prepare_vehicle_frame, load_train_ground_truth, load_vehicle_ground_truth,
)
from tools.event_track_v2x import prepare_v2v4real_ground_truth as gt_tool
from tools.event_track_v2x.extract_v2v4real_archive import digest_file
from tools.event_track_v2x.v2v4real_gt_oracle import native_oracle
from test_v2v4real_ground_truth import pair, obj, make_volume


def test_historical_defaults_have_exact_original_hashes_and_spd_stays_car():
    assert PaperProtocol('spd', 'val').sha256 == '86b2ebfeba64e6d5e17b65fad03c415190147567794b99e123c023c30541aecb'
    old = PaperProtocol('v2v4real', 'official_test')
    assert old.sha256 == '06e5ce65c1bd93b06ab5582d31ff1ed588b52b90daeaacc218c19bce29dd1574'
    new = PaperProtocol('v2v4real', 'official_test', evaluation_class='vehicle')
    assert new.sha256 != old.sha256
    np.testing.assert_array_equal(new.select([.8, .7, .6], np.array([0, 1, 2])), [0, 1, 2])
    with pytest.raises(ValueError):
        PaperProtocol('spd', 'val', evaluation_class='vehicle')
    with pytest.raises(ValueError):
        PaperProtocol('v2v4real', 'train', candidates='rbf-car-first-top64-v1', evaluation_class='vehicle')


def test_native_vehicle_is_not_the_spd_three_class_map_and_preserves_raw_types():
    frames = pair()
    frames['0']['vehicles'].update({3: obj(kind='Truck'), 4: obj(kind='ConcreteTruck'),
                                  5: obj(kind='Van'), 6: obj(kind='Pedestrian')})
    before = copy.deepcopy(frames)
    assert {x['raw_class'] for x in prepare_frame(frames, ego_cav='0')['objects']} == {'Car'}
    rows = prepare_vehicle_frame(frames, ego_cav='0')['objects']
    assert {x['raw_class'] for x in rows} == {'Car', 'Truck', 'ConcreteTruck', 'Van'}
    assert {x['evaluation_class'] for x in rows} == {'vehicle'}
    assert {x['track_id'] for x in rows} == {1, 3, 4, 5, 102}
    assert frames['0']['vehicles'] == before['0']['vehicles']
    del frames['0']['vehicles'][3]['obj_type']
    with pytest.raises(ValueError):
        prepare_vehicle_frame(frames, ego_cav='0')


@pytest.mark.parametrize('split', ['train', 'test'])
def test_separate_vehicle_gt_manifest_never_passes_historical_loader(tmp_path, split):
    args = make_volume(tmp_path, split=split)
    output = tmp_path / 'vehicle-gt'
    m = gt_tool.prepare(*args, output, evaluation_protocol=VEHICLE_PROTOCOL)
    digest = digest_file(output/'manifest.json')[1]
    accepted, frames = load_vehicle_ground_truth(output, expected_manifest_sha256=digest)
    assert accepted == m and len(frames) == 2
    assert m['native_label_source'] == NATIVE_VEHICLE_SELECTION
    assert m['split'] == ('official_test' if split == 'test' else 'train')
    assert m['test_payloads_read'] is (split == 'test')
    assert not m['tracking_evaluation_performed'] and not m['paper_eligible']
    with pytest.raises(ValueError):
        load_train_ground_truth(output, expected_manifest_sha256=digest)
    if split == 'test':
        with pytest.raises(ValueError, match='official train only'):
            gt_tool.prepare(*args, tmp_path / 'historical')


@pytest.mark.parametrize('field', list(vehicle_binding()))
def test_old_or_incompletely_relabelled_cache_calibration_cannot_opt_in(field):
    binding = vehicle_binding()
    require_vehicle_binding(binding)
    del binding[field]
    with pytest.raises(ValueError, match='vehicle calibration binding'):
        require_vehicle_binding(binding)


def test_vehicle_and_historical_car_statistics_cannot_be_pooled():
    base = {k: 'same' for k in ('detector_sha256', 'embedding_sha256', 'motion', 'roi', 'frequency_hz',
                               'label_version', 'evaluator_sha256', 'amotp_definition')}
    car = dict(base, protocol=asdict(PaperProtocol('v2v4real', 'official_test')))
    vehicle = dict(base, protocol=asdict(PaperProtocol('v2v4real', 'official_test', evaluation_class='vehicle')))
    with pytest.raises(ValueError, match='incomparable'):
        require_same_protocol([car, vehicle])


def test_native_vehicle_numeric_oracle_retains_noncar_and_excludes_pedestrian():
    root = os.environ.get('V2V4REAL_GT_ORACLE_ROOT')
    if not root:
        pytest.skip('separately acquired pinned research reference required')
    frames = pair()
    frames['0']['vehicles'].update({3: obj(kind='Truck'), 4: obj(kind='ConcreteTruck'), 5: obj(kind='Pedestrian')})
    actual = prepare_vehicle_frame(frames, ego_cav='0')
    with native_oracle(root, vehicle=True) as oracle:
        expected = oracle(frames, '0')
    assert {x['track_id'] for x in actual['objects']} == set(expected) == {1, 3, 4, 102}
    for row in actual['objects']:
        np.testing.assert_allclose(row['corners_ego'], expected[row['track_id']], atol=1e-4, rtol=0)

