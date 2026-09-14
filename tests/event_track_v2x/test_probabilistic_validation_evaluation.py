"""Protocol and evidence guards independent of expensive full-val inference."""
import copy
import json
from pathlib import Path

import pytest

from tools.event_track_v2x import evaluate_probabilistic_validation as tool
from tools.event_track_v2x.run_probabilistic_tracking_v2 import validation_sources


def test_native_only_source_inventory_equals_actual_inference_closure():
    assert tool.inference_sources()==validation_sources()


def test_car_first_protocol_changes_description_not_metric_engines_or_roi():
    adapter=tool.native.load_adapter()
    before=copy.deepcopy(adapter.PROTOCOL)
    legacy=tool.native.protocol(adapter);current=tool.protocol(adapter)
    assert current['input_candidate_selection']==tool.INPUT_SELECTION
    assert current['input_candidate_selection']!=legacy['input_candidate_selection']
    assert 'source_ablation_reference' not in current
    assert current['evaluated_classes']==['car']
    assert current['supplementary_classes']==[]
    for k in ('nuscenes','trackeval','roi','box_state','temporal_policy'):
        assert current[k]==legacy[k]
    assert adapter.PROTOCOL==before


@pytest.mark.parametrize('name',['../escape','/absolute','a/../../escape','a\\bad',''])
def test_evidence_path_escape_is_rejected(tmp_path,name):
    with pytest.raises(ValueError):tool.child(tmp_path,name)


def test_symlinked_evidence_is_rejected(tmp_path):
    p=tmp_path/'real';p.write_text('data');(tmp_path/'link').symlink_to(p)
    with pytest.raises(ValueError,match='non-symlink'):tool.child(tmp_path,'link')


def test_partial_run_is_rejected_before_gt_or_native_metric_read(tmp_path,monkeypatch):
    run=tmp_path/'run';run.mkdir();(run/'predictions.jsonl').write_text('partial')
    monkeypatch.setattr(tool.native,'load_adapter',lambda:pytest.fail('partial run reached metric engine'))
    with pytest.raises(ValueError,match='regular non-symlink'):
        tool.evaluate(run,'a'*64,tmp_path/'absent-audit','b'*64,tmp_path/'absent-gt',tmp_path/'output')
    assert not (tmp_path/'output').exists()


def test_changed_final_hash_is_rejected_before_parsing_audit(tmp_path):
    run=tmp_path/'run';run.mkdir();(run/'full-validation-receipt.json').write_text('{}')
    with pytest.raises(ValueError,match='input identity'):
        tool.inspect_run(run,'a'*64,tmp_path/'absent-audit','b'*64)


def test_changed_input_cannot_receive_evaluation_seal(tmp_path):
    p=tmp_path/'input.json';p.write_text('{}')
    before={str(p):tool.native.evidence(p)}
    tool.unchanged(before);p.write_text('{"changed":true}')
    with pytest.raises(ValueError,match='evidence changed'):tool.unchanged(before)


@pytest.fixture
def binding_fixture(tmp_path):
    # Fabricated counts test file binding only, never run native metrics.
    run=tmp_path/'run';run.mkdir();audits=tmp_path/'audit';audits.mkdir()
    config=dict(association_algorithm='lbp',update_rule='pkf',anchor_decoder='joint-map')
    plan=dict(kind='probabilistic_single_history_spd_val_adaptation_plan_v1',class_scope=['car'],
        checkpoint_seed=1337,checkpoint_sha256='checkpoint',geometry_baseline=False,configuration=config,
        runtime={'fixture':True},source_sha256=tool.inference_sources())
    def write(path,value):tool.native.write_json(path,value);return tool.native.sha(path)
    plan_sha=write(run/'plan.json',plan)
    receipt=dict(status='complete',cache_split='val',completed_frames=3316,scheduled_frames=3316,
        allocation_teacher=False,learned_identity_enabled=True,plan_sha256=plan_sha,sequence_heads={},
        probabilistic_association_algorithm='lbp',probabilistic_update_rule='pkf',probabilistic_anchor_decoder='joint-map')
    for name,key in (('predictions.jsonl','predictions_sha256'),('tracking.jsonl','tracking_sha256'),
                     ('frame-timings.jsonl','frame_timings_sha256')):
        (run/name).write_bytes(b'fixture-only\n');receipt[key]=tool.native.sha(run/name)
    for i in range(21):
        name=f'sequence-{i:04d}.sqlite';(run/name).write_bytes(b'fixture-only-not-sqlite')
        receipt['sequence_heads'][f'{i:04d}']=dict(database=name,database_sha256=tool.native.sha(run/name))
    receipt_sha=write(run/'receipt.json',receipt)
    final=dict(receipt,full_official_validation_schedule_completed=True,checkpoint_sha256='checkpoint')
    final_sha=write(run/'full-validation-receipt.json',final)
    inventory={str(p):tool.native.sha(p) for p in run.iterdir()}
    for name in ('audit_probabilistic_validation.py','compare_train_probabilistic_controls.py'):
        p=tool.ROOT/'tools/event_track_v2x'/name;inventory[str(p)]=tool.native.sha(p)
    audit=dict(kind='probabilistic_full_spd_val_audit_v1',status='complete',inference_receipt_sha256=final_sha,
        source_directory=str(run),full_validation_coverage_verified=True,frames=3316,
        sequence_heads=receipt['sequence_heads'],configuration=config,update_rule='pkf',seed=1337,
        runtime=plan['runtime'],input_sha256=inventory,replay_receipt_sha256=receipt_sha)
    audit_path=audits/'audit.json';audit_sha=write(audit_path,audit)
    return run,final_sha,audit_path,audit_sha


def test_complete_binding_fixture_is_read_without_metric_computation(binding_fixture):
    value=tool.inspect_run(*binding_fixture)
    assert value['plan']['runtime']['fixture']
    assert value['receipt']['completed_frames']==3316
    assert len(value['evidence'])>100


@pytest.mark.parametrize('name',['predictions.jsonl','tracking.jsonl','frame-timings.jsonl',
                               'sequence-0000.sqlite','plan.json'])
def test_bound_payload_change_is_rejected(binding_fixture,name):
    (binding_fixture[0]/name).write_text('{}')
    with pytest.raises(ValueError):tool.inspect_run(*binding_fixture)


def test_copied_audit_for_another_directory_is_rejected(binding_fixture):
    run,final_sha,p,_=binding_fixture;value=tool.native.read_json(p)
    value['source_directory']=str(run.parent/'other')
    p.write_text(json.dumps(value))
    with pytest.raises(ValueError,match='independent audit'):
        tool.inspect_run(run,final_sha,p,tool.native.sha(p))
