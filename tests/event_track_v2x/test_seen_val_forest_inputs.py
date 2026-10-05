import copy
import importlib.util
from pathlib import Path
import sys

import pytest

TOOLS = Path(__file__).resolve().parents[2]/'tools/event_track_v2x'
sys.path.insert(0, str(TOOLS))
spec = importlib.util.spec_from_file_location('seen_val_bridge', TOOLS/'prepare_rbf_seen_val_forest_inputs.py')
bridge = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bridge)


def fixture():
    row = dict(sequence_id='0003', vehicle_frame='v1', infrastructure_frame='i1',
               box_reference_timestamp_us=1000000)
    schedule = dict(contains_ground_truth=False, contains_system_error_offset=False, frames=[row])
    metadata = {}
    for side, frame in [('vehicle-side', 'v1'), ('infrastructure-side', 'i1')]:
        meta = dict(sequence_id='0003', side=side, frame_id=frame,
                    box_reference_timestamp_us=1000000, source_image_timestamp_us=1000000)
        metadata[('0003', side, frame)] = (side+'-hash', meta)
    event = dict(original_schedule_row=copy.deepcopy(row), decision_us=1100000, origin_us=1000000,
        new_count=0, old_count=0, unavailable_sources=[], source_receipts=[dict(side=side,
        frame_id=frame, frame_sha256=side+'-hash', declared_arrival_us=1100000,
        selected_detection_indices=[]) for side, frame in [('infrastructure-side','i1'),('vehicle-side','v1')]])
    return schedule, metadata, {'0003':[event]}


def test_zero_query_event_retains_real_deliveries():
    result = bridge.convert_events(*fixture())
    assert len(result['events']) == 1 and len(result['events'][0]['deliveries']) == 2
    assert result['zero_selected_query_events'] == 1 and result['zero_delivery_events'] == 0


def test_late_source_is_excluded_and_empty_event_preserved():
    schedule, metadata, recorded = fixture()
    for _, meta in metadata.values():
        meta['source_image_timestamp_us'] = 1100001
    event = recorded['0003'][0]
    event['source_receipts'] = []
    event['unavailable_sources'] = [dict(side=side, frame_id=frame, frame_sha256=side+'-hash')
        for side, frame in [('vehicle-side','v1'),('infrastructure-side','i1')]]
    result = bridge.convert_events(schedule, metadata, recorded)
    assert result['events'][0]['deliveries'] == [] and result['zero_delivery_events'] == 1


def test_deadline_equality_is_available():
    schedule, metadata, recorded = fixture()
    for _, meta in metadata.values():
        meta['source_image_timestamp_us'] = 1100000
    assert len(bridge.convert_events(schedule, metadata, recorded)['events'][0]['deliveries']) == 2


@pytest.mark.parametrize('mutation', [
    lambda s,m,r: s.update(contains_ground_truth=True),
    lambda s,m,r: s['frames'][0].update(GT_ID=17),
    lambda s,m,r: r['0003'].clear(),
    lambda s,m,r: r['0003'][0].update(origin_us=999999),
    lambda s,m,r: r['0003'][0].update(decision_us=1100001),
    lambda s,m,r: r['0003'][0].update(new_count=1),
    lambda s,m,r: r['0003'][0]['source_receipts'][0].update(frame_sha256='changed'),
    lambda s,m,r: r['0003'][0]['source_receipts'][0].update(declared_arrival_us=1099999),
    lambda s,m,r: m[('0003','vehicle-side','v1')][1].update(source_image_timestamp_us=1100001),
    lambda s,m,r: m[('0003','vehicle-side','v1')][1].update(frame_id='wrong'),
    lambda s,m,r: r['0003'][0]['source_receipts'][0].update(selected_detection_indices=[1,1]),
])
def test_protocol_tampering_rejected(mutation):
    values = fixture()
    mutation(*values)
    with pytest.raises(ValueError):
        bridge.convert_events(*values)


def test_reused_frame_rejected_even_at_later_reference():
    schedule, metadata, recorded = fixture()
    second = copy.deepcopy(recorded['0003'][0])
    second['original_schedule_row']['box_reference_timestamp_us'] += 100000
    second['decision_us'] += 100000
    recorded['0003'].append(second)
    schedule['frames'].append(copy.deepcopy(second['original_schedule_row']))
    with pytest.raises(ValueError, match='source frame repeated'):
        bridge.convert_events(schedule, metadata, recorded)


def test_manifest_path_escape_and_symlinks_rejected(tmp_path):
    (tmp_path/'ok').write_text('value')
    assert bridge.safe_file(tmp_path, 'ok') == tmp_path/'ok'
    (tmp_path/'link').symlink_to(tmp_path/'ok')
    for path in ('../ok', '/etc/hosts', 'link'):
        with pytest.raises(ValueError):
            bridge.safe_file(tmp_path, path)
