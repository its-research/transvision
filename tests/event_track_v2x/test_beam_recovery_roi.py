import copy

import pytest

from tools.event_track_v2x.audit_beam_recovery_roi import deltas
from tools.event_track_v2x.evaluate_train_inference_diagnostic import evaluator


def inputs():
    ground=[dict(sequence_id='s',frame_id='f',box_reference_timestamp_us=1,ego_translation_world=[0.,0.,0.])]
    box=lambda name,x:dict(track_id=name,class_label='car',mean=[x,0.,100.,4.,2.,1.,0.,0.,0.],score=.5)
    left=[dict(sequence_id='s',frame_id='f',box_reference_timestamp_us=1,predictions=[box('inside',49.),box('outside',50.)])]
    right=copy.deepcopy(left);right[0]['predictions'][1]['mean'][0]=60.
    return ground,left,right


def test_strict_native_xy_roi_does_not_turn_outside_change_into_identity_gain():
    _,adapter=evaluator();ground,left,right=inputs()
    result=deltas(adapter,ground,left,right)
    assert result['changed_output_frames']==1 and result['roi_changed_frames']==0
    assert result['all_roi_prediction_records_identical']
    assert result['counts']['off_roi_boxes']==result['counts']['on_roi_boxes']==1
    assert result['changed_frames'][0]['modified_fields']=={'mean':1}


def test_in_roi_score_or_identity_change_is_not_hidden():
    _,adapter=evaluator();ground,left,right=inputs()
    right[0]['predictions'][0]['score']=.6
    result=deltas(adapter,ground,left,right)
    assert result['roi_changed_frames']==1 and not result['all_roi_prediction_records_identical']


@pytest.mark.parametrize('bad',['missing','clock','duplicate'])
def test_partial_or_misaligned_outputs_rejected(bad):
    _,adapter=evaluator();ground,left,right=inputs()
    if bad=='missing':right=[]
    elif bad=='clock':right[0]['box_reference_timestamp_us']=2
    else:right[0]['predictions'].append(right[0]['predictions'][0])
    with pytest.raises(ValueError):deltas(adapter,ground,left,right)
