"""Reconstruct full temporal state/discrete decisions from sealed pair outputs.

Only GT-free detector caches, public frozen frame identities/times and sealed
pair alternatives supply inputs. Recorded outputs, nodes and lifecycle are
comparison targets, never recurrence inputs. No original arithmetic executes.
"""
import argparse
import datetime
import gzip
import hashlib
import json
import math
from pathlib import Path
import platform
import numpy as np
import stage1_discrete_replay_oracle as replay
import stage1_fusion_numeric_oracle as numeric


def canonical(value):return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()


def ordinary(path):
    p=Path(path)
    if not p.is_file() or any(x.is_symlink() for x in [p,*p.parents]):raise ValueError('ordinary contained original input required')
    return p


def scalar(x,y):
    if not math.isfinite(float(x)) or not math.isfinite(float(y)) or not math.isclose(x,y,abs_tol=1e-8,rel_tol=1e-10):raise ValueError('complete original scalar differs')
    return abs(x-y)


class Sources:
    def __init__(self,plan):
        self.root=Path(plan['cache_root']);self.pins={};self.bind(self.root/'manifest.json',plan['cache_manifest_sha256'])
        manifest=json.loads((self.root/'manifest.json').read_bytes())
        if manifest['gt_in_cache'] or manifest['test_payloads_read'] or manifest['split']!='val':raise ValueError('original GT-free seen-val detector cache required')
        self.entries={}
        for entry in manifest['frames']:
            p=self.path(entry['metadata']['path']);self.bind(p,entry['metadata']['sha256'],entry['metadata']['bytes']);meta=json.loads(p.read_bytes())
            key=(meta['side'],meta['sequence_id'],meta['frame_id'])
            if key in self.entries:raise ValueError('duplicate full original source frame')
            self.entries[key]=(entry,meta)

    def path(self,relative):
        p=Path(relative)
        if p.is_absolute() or '..' in p.parts:raise ValueError('original contained relative path required')
        return self.root/p

    def bind(self,path,digest,size=None):
        p=ordinary(path)
        if sha(p)!=digest or (size is not None and p.stat().st_size!=size):raise ValueError('original full source bytes differ')
        self.pins[str(p)]=digest

    def frame(self,side,sid,fid,target,deadline):
        entry,meta=self.entries[side,sid,fid];path=self.path(entry['arrays']['path']);self.bind(path,entry['arrays']['sha256'],entry['arrays']['bytes'])
        if meta['arrays_sha256']!=entry['arrays']['sha256']:raise ValueError('original raw array binding differs')
        with np.load(path,allow_pickle=False) as z:arrays={k:z[k].copy() for k in z.files}
        if set(arrays)!={'states','covariances','raw_scores','scores','class_indices','appearance','appearance_valid'}:raise ValueError('full original detector schema required')
        h=hashlib.sha256(canonical(meta))
        for key in sorted(arrays):
            a=arrays[key];h.update(key.encode());h.update(a.dtype.str.encode());h.update(canonical(list(a.shape)));h.update(a.tobytes())
        if h.hexdigest()!=entry['frame_sha256']:raise ValueError('raw detector semantic digest differs')
        info=max(meta['box_reference_timestamp_us'],meta['source_image_timestamp_us']);available=info<=deadline
        order=np.argsort(-arrays['raw_scores'],kind='stable');order=order[arrays['raw_scores'][order]>=.05][:64] if available else []
        source=[]
        for i in order:
            mean,cov=numeric.physical_source(arrays['states'][i],arrays['covariances'][i],meta,target)
            source.append((mean,cov,float(arrays['scores'][i]),int(arrays['class_indices'][i])))
        return source,h.hexdigest(),available,info


def event_identity(values):
    return [{k:x[k] for k in ['event','track_id','node_index'] if k in x} for x in values]


def evaluate(plan_path):
    plan_path=ordinary(plan_path);plan_sha=sha(plan_path);plan=json.loads(plan_path.read_bytes());source=Sources(plan)
    pins={str(plan_path):plan_sha,str(Path(__file__).absolute()):sha(__file__),str(Path(numeric.__file__).absolute()):sha(numeric.__file__),str(Path(replay.__file__).absolute()):sha(replay.__file__)}
    for key in ['predictions','association','diagnostics']:
        p=ordinary(plan[key]);pins[str(p)]=plan[key+'_sha256']
        if sha(p)!=pins[str(p)]:raise ValueError('original sealed full input byte binding differs')
    trackers={};prior_commits={};prior_diagnostics={};counts=dict(frames=0,source_states=0,nodes=0,full_assignment_certificates=0,temporal_assignments=0,outputs=0,events=0)
    maximum_mean=maximum_covariance=maximum_scalar=maximum_duality_gap=0.;trace=hashlib.sha256();current_identity=None
    try:
        with gzip.open(plan['diagnostics'],'rt') as diagnostics,open(plan['predictions']) as predictions,open(plan['association']) as associations:
            for line in diagnostics:
                d=json.loads(line);pl=predictions.readline();al=associations.readline()
                if not pl or not al:raise ValueError('full original streams truncated')
                p=json.loads(pl);a=json.loads(al);sid,fid=d['sequence_id'],d['frame_id'];target=d['box_reference_timestamp_us'];current_identity=dict(sequence_id=sid,frame_id=fid,completed_frames=counts['frames'])
                if any((x['sequence_id'],x['frame_id'])!=(sid,fid) for x in [p,a]) or d['mode']!=plan['mode']:raise ValueError('full original mode/frame alignment differs')
                if p['box_reference_timestamp_us']!=target or p['decision_timestamp_us']!=target+100000 or d['decision_timestamp_us']!=target+100000:raise ValueError('original causal timestamp/deadline differs')
                if d['source_frame_ids'][0]!=fid:raise ValueError('vehicle reference/source identity differs')
                for value,key,prior in [(p,'commit_sha256',prior_commits),(d,'diagnostic_commit_sha256',prior_diagnostics)]:
                    body=dict(value);actual=body.pop(key);previous_key='previous_'+key
                    if body[previous_key]!=prior.get(sid,'0'*64) or hashlib.sha256(canonical(body)).hexdigest()!=actual:raise ValueError('original complete append-only commit chain differs')
                    prior[sid]=actual
                if hashlib.sha256(canonical(a)).hexdigest()!=d['association_frame_sha256'] or d['prediction_commit_sha256']!=p['commit_sha256']:raise ValueError('original sealed cross-stream commit binding differs')
                sides=[]
                for i,side in enumerate(['vehicle-side','infrastructure-side']):
                    x=source.frame(side,sid,d['source_frame_ids'][i],target,target+100000)
                    if x[1]!=d['source_cache_sha256'][i] or x[1]!=p['source_cache_sha256'][i] or x[2]!=d['source_available'][i] or x[2]!=p['source_available'][i] or x[3]!=p['source_information_timestamp_us'][i] or len(x[0])!=d['selected_detections'][i] or len(x[0])!=p['selected_detections'][i]:raise ValueError('full selected original causal sources differ')
                    sides.append(x[0]);counts['source_states']+=len(x[0])
                values=replay.nodes(sides[0],sides[1],a['hypotheses'],plan['mode']);counts['nodes']+=len(values)
                if len(values)!=len(d['nodes']):raise ValueError('complete independent node inventory differs')
                for x,y in zip(values,d['nodes']):
                    maximum_mean=max(maximum_mean,numeric.compare_mean(x['mean'],y['mean']));maximum_scalar=max(maximum_scalar,scalar(x['score'],y['score']))
                if sid not in trackers:trackers[sid]=replay.Replay(sid)
                previous_count=len(trackers[sid].tracks)
                out,matches,events,certificate=trackers[sid].step(target,values)
                counts['full_assignment_certificates']+=1;maximum_duality_gap=max(maximum_duality_gap,certificate['duality_gap'])
                actual_matches=[{k:x[k] for k in ['track_id','node_index']} for x in d['temporal_assignments']]
                if matches!=actual_matches:
                    print('RBF_DISCRETE_REPLAY_DECISION_DIFFERENCE_JSON '+json.dumps(dict(**current_identity,independent_matches=matches,original_matches=actual_matches,assignment_certificate=certificate,full_native_admission=False)),flush=True)
                    raise ValueError('independent global temporal decisions differ; no original-choice fallback')
                if event_identity(events)!=event_identity(d['events']):raise ValueError('full independent lifecycle decisions differ')
                actual={x['track_id']:x for x in p['predictions']}
                if len(actual)!=len(p['predictions']) or set(out)!=set(actual) or previous_count!=d['tracks_before'] or len(out)!=d['tracks_after']:raise ValueError('independent entire temporal state inventory differs')
                serialized=[]
                for tid,x in sorted(out.items()):
                    y=actual[tid]
                    if ['car','bicycle','pedestrian'][x['class_index']]!=y['class_label']:raise ValueError('original output class differs')
                    maximum_mean=max(maximum_mean,numeric.compare_mean(x['mean'],y['mean']));maximum_covariance=max(maximum_covariance,numeric.compare_covariance(x['covariance'],y['covariance']));maximum_scalar=max(maximum_scalar,scalar(x['score'],y['score']))
                    if [h['track_id'] for h in x['identity_hypotheses']]!=[h['track_id'] for h in y['identity_hypotheses']]:raise ValueError('original identity alternative order differs')
                    for h,j in zip(x['identity_hypotheses'],y['identity_hypotheses']):maximum_scalar=max(maximum_scalar,scalar(h['probability'],j['probability']))
                    maximum_scalar=max(maximum_scalar,scalar(x['other_identity_probability'],y['other_identity_probability']))
                    serialized.append(dict(track_id=tid,mean=x['mean'].tolist(),covariance=x['covariance'].tolist(),score=x['score']))
                trace.update(canonical(dict(sequence_id=sid,frame_id=fid,independent_predictions=serialized,independent_matches=matches,independent_events=events)))
                counts['outputs']+=len(out);counts['events']+=len(events);counts['temporal_assignments']+=len(matches);counts['frames']+=1
                if counts['frames']%100==0:print(json.dumps(dict(stage='full_sealed_association_independent_discrete_and_continuous_replay',run_id=plan['run_id'],**counts)),flush=True)
            if predictions.readline() or associations.readline():raise ValueError('unaccounted complete sealed frames')
        if counts['frames']!=3316 or len(trackers)!=21:raise ValueError('full21 sequence original3316 frames required')
        for path,digest in {**source.pins,**pins}.items():
            if sha(ordinary(path))!=digest:raise ValueError('original frozen source or input changed during full replay')
    except BaseException as error:
        print('RBF_DISCRETE_REPLAY_FAILURE_JSON '+json.dumps(dict(kind='rbf_stage1_independent_full_discrete_replay_failure_preserved_v1',run_id=plan['run_id'],current_frame=current_identity,completed_counts=counts,error_type=type(error).__name__,reason=str(error),source_sha256=sha(__file__),oracle_sha256=sha(replay.__file__),plan_sha256=plan_sha,full_native_admission=False,original_strict_replay_pass=plan['strict_replay_pass'],paper_metrics_complete=False)),flush=True)
        raise
    return dict(kind='rbf_stage1_full_native_independent_discrete_and_continuous_replay_under_sealed_association_v1',checked_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),run_id=plan['run_id'],counts=counts,all21_native_sequences=True,full_independent_global_assignment_and_continuous_state_replay_passed=True,previous_actual_outputs_used_as_recurrence_inputs=False,original_lifecycle_or_assignment_decisions_used_as_inputs=False,production_arithmetic_or_SciPy_assignment_executed=False,all_primal_dual_assignment_certificates_checked=True,maximum_duality_gap=maximum_duality_gap,maximum_mean_difference=maximum_mean,maximum_covariance_difference=maximum_covariance,maximum_scalar_difference=maximum_scalar,mean_atol=numeric.MEAN_ATOL,covariance_atol=numeric.COVARIANCE_ATOL,rtol=numeric.RTOL,certificate_atol=replay.CERTIFICATE_ATOL,independent_all_output_trace_sha256=trace.hexdigest(),original_input_files=len(source.pins),plan_sha256=plan_sha,source_sha256=sha(__file__),oracle_sha256=sha(replay.__file__),numeric_oracle_sha256=sha(numeric.__file__),numpy_version=np.__version__,python_version=platform.python_version(),sealed_association_outputs_are_inputs=True,original_source_frame_identity_and_timestamps_are_public_schedule_inputs=True,new_neural_model_inference_executed=False,causal_error_counterfactual_proven=False,original_strict_replay_pass=plan['strict_replay_pass'],historical_M3_2027_strict_failure_unmodified=True,stage1_qualified=False,paper_metrics_complete=False,goal_status='active')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--plan',required=True);a=p.parse_args()
    print('RBF_DISCRETE_REPLAY_JSON '+json.dumps(evaluate(a.plan),sort_keys=True),flush=True)
