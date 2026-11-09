import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
from tools.event_track_v2x.v2v4real_motion_fit_inputs import sha
from tools.event_track_v2x.v2v4real_raw_motion_inputs import load_raw
from tools.event_track_v2x.v2v4real_causal_detector_motion_oracle import scalar_estimate
from tools.event_track_v2x.verify_v2v4real_full_causal_motion import verify
from tests.event_track_v2x.test_v2v4real_raw_motion_inputs import fixture


def save(p,d):p.write_text(json.dumps(d));return sha(p)


def candidate(tmp_path,*,corrupt=None):
    raw=tmp_path/'raw';raw.mkdir();ms,ars=fixture(raw)
    m,index,frames=load_raw(raw,ms,ars,{})
    covariance=np.eye(9);covariance[0,7]=covariance[7,0]=.2
    model=dict(seed=1337,gates=dict(max_speed_mps=40.,max_gap_ordinals=1),sides={s:dict(
        initial_prior=dict(velocity_source=[3.,-2.]),statuses={k:dict(covariance=(covariance*v).tolist())
        for k,v in [('matched',1),('unavailable',2)]}) for s in ('vehicle-side','infrastructure-side')})
    fit=tmp_path/'motion-train-fit-independent-fixture';fit.mkdir();original=fit/'model.json';modelsha=save(original,model)
    accept=fit/'acceptance-receipt.json';ash=save(accept,dict(kind='v2v4real_motion_train_candidate_independent_numeric_acceptance_v1',
        official_test_used=False,all_full9_residual_moments_and_payload_covariances_verified=True,
        input_sha256={str(original):modelsha},gate_selection=model['gates']))
    root=tmp_path/'candidate';root.mkdir();(root/'payloads').mkdir();(root/'model.json').write_bytes(original.read_bytes())
    history={};evidence=[];measured=0
    for i,f in enumerate(frames):
        key=f['sequence_id'],f['side'];pred=scalar_estimate(f,history.get(key),**model['gates']);history[key]=f
        valid=np.asarray(pred['velocity_valid'],dtype=bool);v=np.asarray(pred['velocity_source']).reshape(-1,2);v[~valid]=[3,-2]
        c=np.empty((len(v),9,9));c[valid]=covariance;c[~valid]=covariance*2
        if corrupt=='velocity' and i==0:v[0,0]+=1
        if corrupt=='covariance' and i==0:c[0,0,7]=c[0,7,0]=0
        name=f'payloads/{i:05d}.npz';np.savez_compressed(root/name,velocity_source=v,covariances=c)
        measured+=int(valid.sum());dependencies=pred['dependencies']
        if corrupt=='dependencies' and i==0:dependencies=['f'*64]
        evidence.append(dict(raw_row_sha256=f['row_sha256'],dependencies=dependencies,reasons=pred['reasons'],
            predecessor_candidates=pred['predecessor_candidates'],velocity_measured_valid=pred['velocity_valid'],
            initial_prior_applied=(~valid).tolist(),arrays=dict(path=name,sha256=sha(root/name))))
    e=root/'motion-frame-evidence.jsonl';e.write_text(''.join(json.dumps(x)+'\n' for x in evidence))
    rec=dict(kind='v2v4real_full_split_causal_motion_candidate_v1',split='train',seed=1337,online_GT_read=False,
        test_used_for_fitting_or_selection=False,source_sha256={},input_sha256={str(accept):ash},
        fit_acceptance_sha256=ash,model_sha256=modelsha,raw_root=str(raw),raw_manifest_sha256=ms,
        raw_archive_sha256=ars,checkpoint_sha256=m['checkpoint_sha256'],raw_index_sha256=m['rows_sha256'],
        evidence_sha256=sha(e),measured_valid=measured,candidates=3)
    digest=save(root/'candidate-receipt.json',rec)
    return SimpleNamespace(candidate=root,candidate_sha256=digest,output=tmp_path/'independent')


def test_actual_complete_fixture_and_empty_covariance_pass(tmp_path):
    args=candidate(tmp_path);verify(args)
    d=json.loads((args.output/'acceptance-receipt.json').read_text())
    assert d['source_rows']==4 and d['candidates']==3 and d['measured_valid']==1
    assert d['independent_acceptance_verified'] and not d['nine_state_cache_admitted']


@pytest.mark.parametrize('corrupt',['velocity','covariance','dependencies'])
def test_rehashed_bad_outputs_rejected_by_independent_math(tmp_path,corrupt):
    args=candidate(tmp_path,corrupt=corrupt)
    with pytest.raises((AssertionError,ValueError)):verify(args)
    assert not args.output.exists()
