#!/usr/bin/env python3
"""Full independent NumPy network and branch-and-bound assignment readback."""
import argparse,hashlib,heapq,itertools,json,tarfile,time
from pathlib import Path
import numpy as np
from scipy.special import logsumexp
from scipy.stats import chi2
from scipy.optimize import linear_sum_assignment
from read_spd_association_checkpoint_numpy import read_checkpoint,forward


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def optimum_energies(cost,count):
    n=cost.shape[0]
    if not n:return [0.]
    queue=[];serial=itertools.count()
    def admit(prefix,blocked):
        a=cost.copy()
        for i,j in blocked:a[i,j]=np.inf
        for i,j in enumerate(prefix):
            v=a[i,j];a[i,:]=np.inf;a[:,j]=np.inf;a[i,j]=v
        try:i,j=linear_sum_assignment(a)
        except ValueError:return
        if len(i)==n and np.isfinite(a[i,j]).all():heapq.heappush(queue,(float(cost[i,j].sum()),next(serial),tuple(j),prefix,blocked))
    admit((),frozenset());values=[];seen=set()
    while queue and len(values)<count:
        energy,_,choice,prefix,blocked=heapq.heappop(queue)
        if choice not in seen:values.append(energy);seen.add(choice)
        for i in range(len(prefix),n):admit(choice[:i],blocked|{(i,choice[i])})
    return values


def verify(fold):
    R=Path('/Volumes/Data/test/recover-before-fuse');B=R/f'artifacts/spd-canonical-oof-fold{fold}-heldout-car-association-inference-v1-independent-readback-20261001';D=R/f'artifacts/spd-canonical-oof-fold{fold}-full-heldout-car-association-input-v1-20261001';receipt=B/'independent-full-numeric-and-hypothesis-acceptance.json';failure=B/'independent-full-numeric-and-hypothesis-failure.json';assert not receipt.exists() and not failure.exists();byte=json.loads((B/'byte-readback-receipt.json').read_text());assert byte['all_registered_artifact_bytes_verified'];manifest=json.loads((B/'inference-manifest.json').read_text());input_proof=json.loads((D/'independent-full-input-acceptance.json').read_text());assert input_proof['all_feature_bytes_encoder_values_gate_distances_and_deadlines_verified'];assert manifest['source_event_inventory_sha256']==sha(D/'source-events.json')==sha(B/'source-events.json') and manifest['runtime_sha256']==sha(B/'runtime.json') and manifest['hypotheses_sha256']==sha(B/'all-hypotheses.jsonl') and manifest['logit_archive_sha256']==sha(B/'all-logits.tar.gz');checkpoint_path=R/f'artifacts/spd-canonical-oof-fold{fold}-car-only-association-byte-freeze-20261001/checkpoint.pt';assert sha(checkpoint_path)==manifest['checkpoint_sha256'];checkpoint=read_checkpoint(checkpoint_path);assert checkpoint['configuration']['class_scope']=='car' and checkpoint['fold_id']==fold;key=lambda r:(r['sequence_id'],r['vehicle_frame_id']);inputs={key(r):r for r in json.loads((D/'examples.json').read_text())};records={key(r):r for r in manifest['records']};hypotheses={}
    for line in (B/'all-hypotheses.jsonl').read_text().splitlines():
        r=json.loads(line);assert key(r) not in hypotheses;hypotheses[key(r)]=r
    assert set(records)==set(inputs)==set(hypotheses) and len(records)==manifest['input_vehicle_references'];runtime=json.loads((B/'runtime.json').read_text());assert len(runtime)==4 and len({r['GPU_uuid'] for r in runtime})==4;expected={r['logits']['path']:r for r in records.values()};seen=set();errors=[];numeric_failures=0;max_error=0.;start=time.monotonic();cases=0
    with tarfile.open(B/'all-logits.tar.gz','r:gz') as tar:
        for member in tar:
            assert member.isfile() and member.name in expected and member.name not in seen;row=expected[member.name];payload=tar.extractfile(member).read();assert len(payload)==row['logits']['bytes'] and hashlib.sha256(payload).hexdigest()==row['logits']['sha256'];seen.add(member.name)
            import io
            with np.load(io.BytesIO(payload),allow_pickle=False) as z:v={k:z[k] for k in z.files}
            assert set(v)=={'pair_logits','left_dustbin_logits','right_dustbin_logits','left_query_indices','right_query_indices','left_classes','right_classes','innovation_distance_squared'};inp=inputs[key(row)];hrow=hypotheses[key(row)];assert row['input_features_sha256']==inp['features']['sha256']==hrow['input_features_sha256']
            for k in ['infrastructure_frame_id','decision_time_us','origin_us','available']:assert row[k]==inp[k]==hrow[k]
            fp=D/inp['features']['path'];assert sha(fp)==inp['features']['sha256']
            with np.load(fp,allow_pickle=False) as z:x={k:z[k] for k in z.files}
            for k in ['left_query_indices','right_query_indices','left_classes','right_classes','innovation_distance_squared']:assert np.array_equal(x[k],v[k])
            pair,ld,rd=[v[k] for k in ['pair_logits','left_dustbin_logits','right_dustbin_logits']];n,m=len(x['left']),len(x['right']);assert pair.shape==(n,m) and ld.shape==(n,) and rd.shape==(m,) and all(a.dtype==np.float32 and np.isfinite(a).all() for a in [pair,ld,rd]);reference=forward(checkpoint,x['left'],x['right']);differences=[float(np.max(abs(actual-ref))) if actual.size else 0. for actual,ref in zip([pair,ld,rd],reference)];max_error=max(max_error,max(differences));passed=all(np.allclose(actual,ref,atol=1e-4,rtol=1e-4) for actual,ref in zip([pair,ld,rd],reference))
            if not passed:
                numeric_failures+=1
                if len(errors)<8:errors.append(dict(sequence_id=row['sequence_id'],vehicle_frame_id=row['vehicle_frame_id'],max_absolute_errors=differences))
            assert set(hrow['gate_and_TopH_retained_hypotheses'])=={'0.9','0.95','0.99'}
            for confidence in [.90,.95,.99]:
                gate=v['innovation_distance_squared']<=chi2.ppf(confidence,3);groups=hrow['gate_and_TopH_retained_hypotheses'][str(confidence)];assert set(groups)=={'1','3','5'}
                if n and m:
                    logits=np.where(gate,pair.astype(float),-np.inf);rl=np.c_[logits,ld];rl-=logsumexp(rl,axis=1)[:,None];cl=np.c_[logits.T,rd];cl-=logsumexp(cl,axis=1)[:,None];cost=np.full((n,m+n),np.inf);cost[:,:m]=-rl[:,:m]-cl[:,:n].T+cl[:,-1][None,:];cost[np.arange(n),m+np.arange(n)]=-rl[:,-1];best=optimum_energies(cost,5);baseline=-cl[:,-1].sum();best=[x+baseline for x in best]
                else:best=[0.]
                for h in [1,3,5]:
                    alternatives=groups[str(h)];columns=[];energies=[];assert any(not a['pairs'] for a in alternatives)
                    for a in alternatives:
                        links=dict(a['pairs']);assert len(links)==len(a['pairs']) and len(set(links.values()))==len(links) and all(0<=i<n and 0<=j<m and gate[i,j] for i,j in links.items());assert a['unmatched_left']==sorted(set(range(n))-set(links)) and a['unmatched_right']==sorted(set(range(m))-set(links.values()));col=tuple(links.get(i,m+i) for i in range(n));columns.append(col);energy=float(cost[np.arange(n),col].sum()+baseline) if n and m else 0.;assert abs(energy-a['energy'])<=1e-9;energies.append(energy)
                    assert len(columns)==len(set(columns));values=np.asarray(energies);weights=np.exp(-values-logsumexp(-values));assert np.allclose([a['weight'] for a in alternatives],weights,atol=1e-12,rtol=1e-10);expected_best=best[:h];assert np.allclose(sorted(energies)[:len(expected_best)],expected_best,atol=1e-9,rtol=1e-10);assert len(alternatives) in [len(expected_best),len(expected_best)+1];cases+=1
            if len(seen)%100==0 or len(seen)==len(expected):print(json.dumps(dict(stage='independent_full_NumPy_association_inference_readback',fold_id=fold,completed=len(seen),total=len(expected),eta_seconds=(time.monotonic()-start)/len(seen)*(len(expected)-len(seen)),numeric_failure_frames=numeric_failures)),flush=True)
    assert seen==set(expected);proof=dict(fold_id=fold,task_id=byte['task_id'],inference_manifest_sha256=sha(B/'inference-manifest.json'),input_independent_acceptance_sha256=sha(D/'independent-full-input-acceptance.json'),checkpoint_sha256=sha(checkpoint_path),full_vehicle_reference_frames=len(inputs),source_frames=manifest['source_frames'],all_registered_bytes_identity_coverage_gate_TopH_unmatched_and_energy_verified=True,numeric_logits_recomputed_without_Torch=True,all_logits_match_independent_NumPy=numeric_failures==0,numeric_failure_frames=numeric_failures,first_numeric_failures=errors,max_absolute_error=max_error,absolute_tolerance=1e-4,relative_tolerance=1e-4,hypothesis_cells=cases,production_network_or_assignment_helpers_called=False,held_out_GT_read=False,full_channel_tracking_replay=False,paper_eligible=False,verifier_sha256=sha(Path(__file__)))
    out=failure if numeric_failures else receipt;out.write_text(json.dumps(proof,indent=2)+'\n');print(json.dumps(dict(fold_id=fold,accepted=numeric_failures==0,numeric_failure_frames=numeric_failures,receipt_sha256=sha(out))),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--fold',type=int,choices=range(5),required=True);a=p.parse_args();verify(a.fold)
