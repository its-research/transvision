"""Full MHT binding/protocol guards; fabricated files never reach native metrics."""
import copy
from dataclasses import asdict
import json
from pathlib import Path
import sqlite3
import subprocess
import sys

import pytest

from tools.event_track_v2x import audit_full_mht_validation as audit
from tools.event_track_v2x import evaluate_mht_validation as evaluation
from tools.event_track_v2x.persistent_mht_tracking import PersistentMHTConfig


def test_configuration_pin_is_actual_frozen_width_four():
    assert audit.ledger.digest(asdict(PersistentMHTConfig())) == audit.CONFIG_SHA


def test_metric_protocol_keeps_engines_roi_and_class_scope():
    adapter = evaluation.native.load_adapter(); before = copy.deepcopy(adapter.PROTOCOL)
    old = evaluation.common.protocol(adapter); current = evaluation.protocol(adapter)
    assert old.pop('kind') != current.pop('kind')
    assert old == current and current['evaluated_classes'] == ['car']
    assert current['input_candidate_selection'] == evaluation.common.INPUT_SELECTION
    assert adapter.PROTOCOL == before


def test_partial_run_rejected_before_gt_or_metric_engine(tmp_path,monkeypatch):
    run = tmp_path/'partial';run.mkdir();(run/'full-validation-receipt.json').write_text('{}')
    monkeypatch.setattr(evaluation.native,'load_adapter',lambda:pytest.fail('partial replay reached metric engine'))
    with pytest.raises(ValueError,match='evidence identity'):
        evaluation.evaluate(run,'a'*64,tmp_path/'missing-audit','b'*64,tmp_path/'missing-gt',tmp_path/'output')
    assert not (tmp_path/'output').exists()


def test_native_modules_do_not_import_torch_or_live_tracker():
    p = subprocess.run([sys.executable,'-c',
        'import sys; from tools.event_track_v2x import evaluate_mht_validation, audit_full_mht_validation; '
        'assert "torch" not in sys.modules; '
        'assert "tools.event_track_v2x.persistent_mht_tracking" not in sys.modules'],
        cwd=Path(__file__).resolve().parents[2],capture_output=True,text=True)
    assert p.returncode == 0,p.stderr


@pytest.fixture
def binding(tmp_path,monkeypatch):
    # 3316 fabricated schedule rows and 21 fake DB files exercise binding only.
    # They are not valid tracker outputs, never audited as SQL or evaluated.
    root = tmp_path/'inputs';root.mkdir();run = tmp_path/'run';run.mkdir()
    def write(path,value): audit.native.write_json(path,value);return audit.native.sha(path)
    rows = [dict(sequence_id=f'{i%21:04d}',vehicle_frame=str(i),infrastructure_frame=str(i),
                 box_reference_timestamp_us=1000000+i*100000) for i in range(3316)]
    rows.sort(key=lambda r:(r['sequence_id'],r['box_reference_timestamp_us']))
    schedule = root/'schedule.json'
    schedule_sha = write(schedule,dict(kind='spd_official_validation_prediction_schedule_v1',
        contains_ground_truth=False,contains_system_error_offset=False,split_sha256=audit.SPLIT_SHA,frames=rows))
    cache = root/'manifest.json';cache_sha = write(cache,dict(split='val',frame_count=7189,
                                                            sequences=sorted({r['sequence_id'] for r in rows})))
    weights = root/'weights.pt';weights.write_bytes(b'fixture-not-a-trained-model')
    checkpoint = root/'checkpoint.json'
    cp_sha = write(checkpoint,dict(full_official_train=True,data_split='train',seed=1337,labels_in_model_inputs=False,
                                  weights=dict(path='weights.pt',sha256=audit.native.sha(weights))))
    monkeypatch.setattr(audit,'CACHE_SHA',cache_sha);monkeypatch.setattr(audit,'SCHEDULE_SHA',schedule_sha)
    monkeypatch.setattr(audit,'CHECKPOINTS',{1337:cp_sha})
    plan = dict(kind='scan_mht_replay_plan_v1',class_scope=['car'],cache_split='val',cache_sha256=cache_sha,
        split_sha256=audit.SPLIT_SHA,schedule_sha256=schedule_sha,schedule_rows_sha256=audit.ledger.digest(rows),
        checkpoint_seed=1337,checkpoint_sha256=cp_sha,configuration=asdict(PersistentMHTConfig()),
        full_official_schedule_verified=True,scheduled_frames=3316,geometry_baseline=False,gt_model_inputs=False,
        parameter_training=False,validation_parameter_search=False,validation_checkpoint_selection=False,
        paper_eligible=False,same_latency_or_memory_verified=False,reproduced_public_method=False,
        val_seen_during_research=True,runtime=dict(audit.RUNTIME,thread_environment=audit.THREAD_ENV),
        source_sha256=audit.ledger.inference_sources(),
        input_file_sha256={str(p):audit.native.sha(p) for p in (schedule,cache,checkpoint,weights)})
    plan_sha = write(run/'plan.json',plan)
    receipt = dict(kind='scan_mht_scheduled_replay_v1',status='complete',scheduled_frames=3316,completed_frames=3316,
        cache_split='val',cache_sha256=cache_sha,plan_sha256=plan_sha,sequence_heads={},
        geometry_development_baseline=False,gt_model_inputs=False,parameter_training=False,test_payloads_read=False,
        reproduced_public_method=False,paper_eligible=False,same_latency_or_memory_verified=False,exclusive_host=False,
        learned_identity_enabled=True,global_scan_mht=True,source_arrival_policy='scheduled_pair_snapshot_at_reference_plus_100ms')
    for name,key in (('predictions.jsonl','predictions_sha256'),('tracking.jsonl','tracking_sha256'),
                     ('frame-timings.jsonl','frame_timings_sha256')):
        # Parseable placeholders let the full auditor actually reach and reject
        # the fake SQLite file, rather than failing earlier on JSON decoding.
        (run/name).write_bytes(b'{}\n');receipt[key] = audit.native.sha(run/name)
    for scene in sorted({r['sequence_id'] for r in rows}):
        name='sequence-'+scene+'.sqlite';(run/name).write_bytes(b'fixture-not-SQLite')
        receipt['sequence_heads'][scene] = dict(database=name,database_sha256=audit.native.sha(run/name),
            frames=sum(r['sequence_id']==scene for r in rows),prediction_sha256='f'*64)
    write(run/'receipt.json',receipt)
    final = dict(receipt,full_official_validation_schedule_completed=True,checkpoint_sha256=cp_sha)
    final_sha = write(run/'full-validation-receipt.json',final)
    bound = audit.bind_run(run,final_sha)
    result = dict(kind=audit.KIND,status='complete',inference_receipt_sha256=final_sha,
        replay_receipt_sha256=audit.native.sha(run/'receipt.json'),source_directory=str(run),
        configuration=plan['configuration'],seed=1337,runtime=plan['runtime'],full_validation_coverage_verified=True,
        frames=3316,observations=0,sequence_heads=receipt['sequence_heads'],input_evidence=bound['evidence'],
        sequences={s:dict(frames=h['frames'],prediction_sha256=h['prediction_sha256'],observations=0)
                   for s,h in receipt['sequence_heads'].items()})
    for k in ('raw_factor_ledger_reconstructed','retained_alias_weights_reconstructed',
              'one_to_one_and_irreversible_predecessors_verified','published_ID_lifecycle_reconstructed'): result[k]=True
    for k in ('ground_truth_read','tracking_metrics_computed','full_top_k_optimality_recomputed',
              'conditional_mass_bounds_recomputed','gaussian_state_numerics_recomputed','fair_resources_verified',
              'paper_eligible','test_payloads_read','parameter_training'): result[k]=False
    ap=tmp_path/'audit.json';audit_sha=write(ap,result)
    return run,final_sha,ap,audit_sha,plan,receipt,final


def test_complete_metadata_binding_fixture_only(binding):
    bound = audit.inspect_audited_run(*binding[:4])
    assert bound['audit']['frames'] == 3316 and len(bound['evidence']) > 140


@pytest.mark.parametrize('field,value',[
    ('checkpoint_seed',42),('checkpoint_sha256','a'*64),('full_official_schedule_verified',False),
    ('class_scope',['car','pedestrian']),('scheduled_frames',8),('geometry_baseline',True),
    ('validation_parameter_search',True),('validation_checkpoint_selection',True),
    ('val_seen_during_research',False),('cache_split','test'),('schedule_sha256','a'*64),
])
def test_unfrozen_or_leaky_headers_are_rejected(binding,field,value):
    plan=copy.deepcopy(binding[4]);plan[field]=value
    with pytest.raises(ValueError):audit.check_headers(binding[6],binding[5],plan)


@pytest.mark.parametrize('mutation',['width','binary','threads','full-flag','count','teacher','source-inventory'])
def test_algorithm_runtime_and_final_contract_rejected(binding,mutation):
    run,sha,_,_,plan,receipt,final = binding;plan=copy.deepcopy(plan);final=copy.deepcopy(final);receipt=copy.deepcopy(receipt)
    if mutation=='width': plan['configuration']['state']['active_limit']=16
    elif mutation=='binary': plan['runtime']['assignment_binary_sha256']='a'*64
    elif mutation=='threads': plan['runtime']['thread_environment']['OMP_NUM_THREADS']='8'
    elif mutation=='full-flag': final['full_official_validation_schedule_completed']=False
    elif mutation=='count': receipt['completed_frames']=8;final['completed_frames']=8
    elif mutation=='teacher': receipt['parameter_training']=True;final['parameter_training']=True
    else:
        path=run/'plan.json';plan['source_sha256'].pop(next(iter(plan['source_sha256'])))
        path.write_bytes(audit.native.canonical(plan))
        with pytest.raises(ValueError):audit.bind_run(run,sha)
        return
    with pytest.raises(ValueError):audit.check_headers(final,receipt,plan)


@pytest.mark.parametrize('name',['plan.json','predictions.jsonl','tracking.jsonl','frame-timings.jsonl',
                               'sequence-0000.sqlite','receipt.json'])
def test_changed_replay_payload_rejected_before_metric_engine(binding,name):
    (binding[0]/name).write_bytes(b'changed')
    with pytest.raises((ValueError,KeyError)):audit.inspect_audited_run(*binding[:4])


@pytest.mark.parametrize('mutation',['subset','identity-check','risk-claim','missing-evidence','wrong-sequence-count'])
def test_subset_or_incomplete_audit_never_authorizes_gt(binding,mutation):
    run,sha,path,_,_,_,_=binding;value=audit.native.read_json(path)
    if mutation=='subset': value['kind']='scan_mht_replay_identity_audit_v1';value['full_validation_coverage_verified']=False
    elif mutation=='identity-check': value['retained_alias_weights_reconstructed']=False
    elif mutation=='risk-claim': value['full_top_k_optimality_recomputed']=True
    elif mutation=='missing-evidence': value['input_evidence'].pop(next(iter(value['input_evidence'])))
    else: value['sequences']['0000']['frames']-=1
    path.write_bytes(audit.native.canonical(value))
    with pytest.raises(ValueError):audit.inspect_audited_run(run,sha,path,audit.native.sha(path))


def test_full_audit_does_not_accept_fabricated_database(binding,tmp_path):
    with pytest.raises(sqlite3.DatabaseError,match='not a database'):
        audit.audit(binding[0],binding[1],tmp_path/'full-audit')
    assert not (tmp_path/'full-audit').exists()
