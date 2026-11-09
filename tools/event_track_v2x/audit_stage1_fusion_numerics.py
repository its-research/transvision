"""Full sealed-frame numerical audit, conditional on sealed discrete decisions.

Only original pure arithmetic ASTs rehydrate the opaque covariance preimages.
They never supply the independently checked equations. Every such reference
must match the recorded SHA before use. This is not a fresh model replay,
independent global-assignment proof, or causal error counterfactual.
"""
import argparse
import ast
import __future__
import dataclasses
import gzip
import hashlib
import json
import math
import platform
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import stage1_fusion_numeric_oracle as oracle


def canonical(x):
    return json.dumps(x, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024*1024), b''): h.update(chunk)
    return h.hexdigest()


def ordinary(path):
    p = Path(path)
    if not p.is_file() or any(x.is_symlink() for x in [p, *p.parents]):
        raise ValueError('original ordinary nonlinked input required')
    return p


def scalar(actual, expected):
    if not math.isfinite(float(actual)) or not math.isfinite(float(expected)) or not math.isclose(actual, expected, rel_tol=1e-10, abs_tol=1e-8):
        raise ValueError('independent original scalar differs')
    return abs(float(actual)-float(expected))


def reference_math(root, hashes):
    """Whitelist only unchanged arithmetic; no imports or tracker/model code."""
    selected = {
        'arrays.py': {'immutable_float64'},
        'fusion.py': {'CIFusion', '_inputs', 'covariance_intersection'},
        'prediction_features.py': {'wrap_angle', 'validate_pose', 'transform_state_covariance'},
        'tracking_v2.py': {'propagate', 'physical_world', 'ci', 'moment_match'},
    }
    namespace = {'np': np, 'dataclass': dataclasses.dataclass, '__name__': __name__}
    for name, names in selected.items():
        p = ordinary(Path(root)/name)
        if sha(p) != hashes[name]: raise ValueError('original source binding differs')
        parsed = ast.parse(p.read_text())
        body = [n for n in parsed.body if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in names]
        if {n.name for n in body} != names: raise ValueError('original reference arithmetic missing')
        exec(compile(ast.Module(body=body, type_ignores=[]), str(p), 'exec', flags=__future__.annotations.compiler_flag), namespace)
    return namespace


def transport(mean, covariance, delta):
    m, p = oracle.gaussian(mean, covariance)
    f = np.eye(9); f[0,7] = f[1,8] = delta
    m[0] += delta*m[7]; m[1] += delta*m[8]
    return m, f@p@f.T + np.eye(9)*.1*abs(delta)


class Audit:
    def __init__(self, plan):
        self.plan = plan
        self.ref = reference_math(plan['original_source_root'], plan['original_source_sha256'])
        self.counts = dict(frames=0, source_states=0, components=0, nodes=0, temporal_assignments=0, output_states=0, covariance_preimages=0)
        self.maximum_mean = self.maximum_covariance = self.maximum_scalar = 0.
        self.cache = Path(plan['cache_root']); self.touched = {}
        self.bind(self.cache/'manifest.json', plan['cache_manifest_sha256'])
        manifest = json.loads((self.cache/'manifest.json').read_bytes())
        if manifest['gt_in_cache'] is not False or manifest['test_payloads_read'] is not False or manifest['split'] != 'val':
            raise ValueError('original GT-free seen-validation input required')
        self.entries = {}
        for entry in manifest['frames']:
            meta_path = self.cache/entry['metadata']['path']
            self.bind(meta_path, entry['metadata']['sha256'], entry['metadata']['bytes'])
            meta = json.loads(meta_path.read_bytes())
            key = (meta['side'], meta['sequence_id'], meta['frame_id'])
            if key in self.entries: raise ValueError('duplicate raw source frame')
            self.entries[key] = (entry, meta)
        self.previous = {}; self.last_target = {}; self.last_update = {}; self.next_id = {}

    def bind(self, path, digest, size=None):
        p = ordinary(path)
        if sha(p) != digest or (size is not None and p.stat().st_size != size):
            raise ValueError('original full input byte binding differs')
        self.touched[str(p)] = digest

    def mean(self, x, y):
        self.maximum_mean = max(self.maximum_mean, oracle.compare_mean(x,y))

    def covariance(self, x, y):
        self.maximum_covariance = max(self.maximum_covariance, oracle.compare_covariance(x,y))

    def number(self, x, y):
        self.maximum_scalar = max(self.maximum_scalar, scalar(x,y))

    def record(self, record, independent, reference_covariance):
        if hashlib.sha256(canonical(np.asarray(reference_covariance).tolist())).hexdigest() != record['covariance_sha256']:
            raise ValueError('original complete covariance SHA preimage not recovered')
        self.counts['covariance_preimages'] += 1
        self.mean(independent[0], record['mean']); self.covariance(independent[1], reference_covariance)
        self.number(np.trace(independent[1]), record['covariance_trace'])
        upper = independent[1][:3,:3][np.triu_indices(3)]
        for x,y in zip(upper, record['position_covariance_upper']): self.number(x,y)

    def source(self, side, sid, fid, target, deadline, recorded_digest, available):
        entry, meta = self.entries[side,sid,fid]
        arrays_path = self.cache/entry['arrays']['path']
        self.bind(arrays_path, entry['arrays']['sha256'], entry['arrays']['bytes'])
        if meta['arrays_sha256'] != entry['arrays']['sha256']: raise ValueError('metadata array binding differs')
        with np.load(arrays_path, allow_pickle=False) as z:
            arrays = {k:z[k].copy() for k in z.files}
        if set(arrays) != {'states','covariances','scores','raw_scores','class_indices','appearance','appearance_valid'}:
            raise ValueError('unknown original raw cache keys')
        digest = hashlib.sha256(canonical(meta))
        for key in sorted(arrays):
            a=arrays[key];digest.update(key.encode());digest.update(a.dtype.str.encode());digest.update(canonical(list(a.shape)));digest.update(a.tobytes())
        if digest.hexdigest() != entry['frame_sha256'] or recorded_digest != digest.hexdigest(): raise ValueError('original raw source semantic digest differs')
        info = max(meta['box_reference_timestamp_us'], meta['source_image_timestamp_us'])
        if available != (info <= deadline): raise ValueError('source causal deadline differs')
        order = np.argsort(-arrays['raw_scores'],kind='stable')
        order = order[arrays['raw_scores'][order] >= .05][:64] if available else np.zeros(0,dtype=np.int64)
        ref_mean, ref_cov = self.ref['physical_world'](SimpleNamespace(metadata=meta, **arrays),order,target)
        independent = [oracle.physical_source(arrays['states'][i], arrays['covariances'][i], meta, target) for i in order]
        return arrays, order, ref_mean, ref_cov, independent, info

    def frame(self, diagnostic, prediction, association):
        d,p,a = diagnostic,prediction,association
        sid,fid = d['sequence_id'],d['frame_id'];target=d['box_reference_timestamp_us']
        if any((x['sequence_id'],x['frame_id']) != (sid,fid) for x in [p,a]): raise ValueError('sealed full stream frame alignment differs')
        if p['box_reference_timestamp_us'] != target or d['decision_timestamp_us'] != target+100000: raise ValueError('original reference/deadline differs')
        if d['mode'] != self.plan['mode']: raise ValueError('original numerical mode differs')
        hypotheses=a['hypotheses'];weights=[h['weight'] for h in hypotheses]
        if abs(math.fsum(weights)-1)>1e-12: raise ValueError('sealed posterior alternatives not normalized')
        source=[];side_names=['vehicle-side','infrastructure-side'];records={}
        for rec in d['source_records']:
            key=(rec['side'],rec['selected_index'])
            if key in records:raise ValueError('duplicate source state record')
            records[key]=rec
        for index,side in enumerate(side_names):
            s=self.source(side,sid,d['source_frame_ids'][index],target,d['decision_timestamp_us'],d['source_cache_sha256'][index],d['source_available'][index])
            if len(s[1]) != d['selected_detections'][index] or s[5] != p['source_information_timestamp_us'][index]:raise ValueError('native source selection/info differs')
            for local,raw in enumerate(s[1]):
                rec=records.pop((side,local))
                if rec['raw_index']!=int(raw) or rec['frame_id']!=d['source_frame_ids'][index] or rec['class_index']!=int(s[0]['class_indices'][raw]):raise ValueError('source state identity differs')
                self.number(rec['raw_score'],s[0]['raw_scores'][raw]);self.number(rec['score'],s[0]['scores'][raw])
                self.record(rec,s[4][local],s[3][local]);self.counts['source_states']+=1
            source.append(s)
        if records:raise ValueError('unaccounted source record')
        vehicle,road=source;vi,ii=vehicle[1],road[1];nodes=[]
        residuals=[j for j in range(len(ii)) if sum(h['weight'] for h in hypotheses if j in h['unmatched_right'])>0]
        if len(d['nodes']) != len(vi)+len(residuals):raise ValueError('full mixture/residual node inventory differs')
        for i,raw in enumerate(vi):
            rec=d['nodes'][i];independent=[];ref_means=[];ref_covariances=[];scores=[]
            if len(rec['components']) != len(hypotheses) or rec['source_raw_index']!=int(raw) or rec['source_selected_index']!=i or rec['class_index']!=int(vehicle[0]['class_indices'][raw]):raise ValueError('complete vehicle alternatives differ')
            for k,(h,component) in enumerate(zip(hypotheses,rec['components'])):
                pairs=dict(h['pairs']);partner=pairs.get(i)
                if component['hypothesis_index']!=k or component['partner_selected_index']!=partner or component['weight']!=h['weight']:raise ValueError('sealed original pair/weight changed')
                if partner is None:
                    ind=vehicle[4][i];rc=vehicle[3][i];score=vehicle[0]['scores'][raw];kind='vehicle_unmatched'
                else:
                    if component['partner_raw_index']!=int(ii[partner]):raise ValueError('partner source reference differs')
                    score=max(vehicle[0]['scores'][raw],road[0]['scores'][ii[partner]])
                    if d['mode']=='M2':ind=road[4][partner];rc=road[3][partner];kind='road_state_replacement'
                    else:
                        ind=oracle.ci(*vehicle[4][i],*road[4][partner]);kind='cross_source_ci'
                        # Original recorded means remove historical libm-vector
                        # variability only from opaque reference rehydration.
                        rm=d['source_records'][i]['mean']
                        other=next(x['mean'] for x in d['source_records'] if x['side']=='infrastructure-side' and x['selected_index']==partner)
                        rc=self.ref['ci'](np.asarray(rm),vehicle[3][i],np.asarray(other),road[3][partner])[1]
                if component['kind']!=kind:raise ValueError('original component mode differs')
                self.record(component,ind,rc);self.number(component['score'],score)
                independent.append(ind);ref_means.append(component['mean']);ref_covariances.append(rc);scores.append(score);self.counts['components']+=1
            node=oracle.mixture([x[0] for x in independent],[x[1] for x in independent],weights)
            rc=self.ref['moment_match'](ref_means,ref_covariances,weights)[1]
            self.record(rec,node,rc);score=math.fsum(float(w)*float(s) for w,s in zip(weights,scores));self.number(rec['score'],score)
            self.number(rec['matched_mass'],sum(h['weight'] for h in hypotheses if i not in h['unmatched_left']))
            self.number(rec['unmatched_mass'],sum(h['weight'] for h in hypotheses if i in h['unmatched_left']))
            nodes.append((node,rc,rec['mean'],score,rec['class_index']))
        for j in residuals:
            rec=d['nodes'][len(nodes)];mass=sum(h['weight'] for h in hypotheses if j in h['unmatched_right']);raw=ii[j]
            score=float(road[0]['scores'][raw])*(1 if d['mode']=='M3' else mass)
            if rec['kind']!='road_residual' or rec['components'] or rec['source_selected_index']!=j or rec['source_raw_index']!=int(raw):raise ValueError('original residual inventory differs')
            self.record(rec,road[4][j],road[3][j]);self.number(rec['score'],score);self.number(rec['unmatched_mass'],mass)
            self.number(rec['matched_mass'],sum(h['weight'] for h in hypotheses if j not in h['unmatched_right']))
            nodes.append((road[4][j],road[3][j],rec['mean'],score,rec['class_index']))
        self.counts['nodes']+=len(nodes)
        previous=self.previous.get(sid,{});dt=0 if sid not in self.last_target else (target-self.last_target[sid])/1e6
        if sid in self.last_target and dt<=0:raise ValueError('causal reference order differs')
        priors={tid:transport(x['mean'],x['covariance'],dt) for tid,x in previous.items()}
        prior_scores={tid:x['score']*.95**dt for tid,x in previous.items()}
        outputs={x['track_id']:x for x in p['predictions']};expected={};touched=set();matched_nodes=set()
        if len(outputs)!=len(p['predictions']) or d['tracks_before']!=len(previous):raise ValueError('full output identity inventory differs')
        for match in d['temporal_assignments']:
            tid=match['track_id'];j=match['node_index']
            if tid not in previous or tid in touched or j in matched_nodes:raise ValueError('sealed temporal identity/node repeated')
            touched.add(tid);matched_nodes.add(j);node,rc,rm,score,cls=nodes[j]
            prior=priors[tid];reference_prior=self.ref['propagate'](np.asarray(previous[tid]['mean']),np.asarray(previous[tid]['covariance']),dt)[1]
            self.record(match['prior_state'],prior,reference_prior)
            self.number(match['prior_score'],prior_scores[tid]);self.number(match['node_score'],score)
            innovation=node[0][:3]-prior[0][:3];uncertainty=2*(node[1][:3,:3]+prior[1][:3,:3]);distance=float(innovation@np.linalg.solve(uncertainty,innovation))
            self.number(match['mahalanobis_squared'],distance);self.number(match['assigned_cost'],.5*distance-math.log(max(score,1e-12)))
            if distance>d['gate_mahalanobis_squared']+1e-8:raise ValueError('selected temporal association outside frozen gate')
            for x,y in zip(innovation,match['innovation_xyz']):self.number(x,y)
            expected[tid]=(oracle.ci(*prior,*node),max(prior_scores[tid],score),cls)
            self.last_update[sid,tid]=target;self.counts['temporal_assignments']+=1
        for event in d['events']:
            kind=event['event'];tid=event.get('track_id')
            if kind in ['miss','kill_before_assignment']:
                if tid not in previous or tid in touched:raise ValueError('sealed lifecycle repeats or invents prior identity')
                touched.add(tid);self.number(event['prior_score'] if kind=='miss' else event['score'],prior_scores[tid])
                age=(target-self.last_update[sid,tid])/1e6
                if kind=='kill_before_assignment':
                    if age<=2 and prior_scores[tid]>=.05:raise ValueError('prior killed inside frozen survival support')
                else:
                    if age>2 or prior_scores[tid]<.05:raise ValueError('ineligible track assigned miss')
                    score=prior_scores[tid]*.75
                    if score>=.05:expected[tid]=(priors[tid],score,['car','bicycle','pedestrian'].index(previous[tid]['class_label']))
            elif kind in ['birth','birth_rejected_score']:
                j=event['node_index'];node,rc,rm,score,cls=nodes[j]
                if j in matched_nodes:raise ValueError('matched node also enters birth decision')
                effective=float(d['nodes'][j]['original_score']) if d['mode']=='M4' and d['nodes'][j]['kind']=='road_residual' else score
                self.number(event['score'],effective)
                if kind=='birth':
                    if effective<.3:raise ValueError('birth outside frozen support')
                    number=self.next_id.get(sid,1);self.next_id[sid]=number+1
                    if tid!=f'{sid}:{number:06d}':raise ValueError('original deterministic birth ID differs')
                    expected[tid]=(node,effective,cls);self.last_update[sid,tid]=target
                elif effective>=.3:raise ValueError('eligible birth rejected')
            elif kind=='kill_after_miss':
                if tid not in touched or prior_scores[tid]*.75>=.05:raise ValueError('incorrect post-miss kill')
                self.number(event['score'],prior_scores[tid]*.75)
            else:raise ValueError('unknown sealed lifecycle event')
        if touched!=set(previous) or set(expected)!=set(outputs) or d['tracks_after']!=len(outputs):raise ValueError('full prior/current numerical inventory not covered')
        for tid,(state,score,cls) in expected.items():
            actual=outputs[tid]
            self.mean(state[0],actual['mean']);self.covariance(state[1],actual['covariance']);self.number(score,actual['score'])
            if actual['class_label']!=['car','bicycle','pedestrian'][cls]:raise ValueError('output class changed')
            self.counts['output_states']+=1
        self.previous[sid]=outputs;self.last_target[sid]=target;self.counts['frames']+=1


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--plan',required=True);args=parser.parse_args()
    plan_path=ordinary(args.plan);plan_hash=sha(plan_path);plan=json.loads(plan_path.read_bytes())
    for k in ['predictions','association','diagnostics']: 
        if sha(ordinary(plan[k]))!=plan[k+'_sha256']:raise ValueError('original sealed stream byte binding differs')
    audit=Audit(plan)
    with gzip.open(plan['diagnostics'],'rt') as ds, open(plan['predictions']) as ps, open(plan['association']) as ass:
        for line in ds:
            pl,al=ps.readline(),ass.readline()
            if not pl or not al:raise ValueError('full frame streams truncated')
            audit.frame(json.loads(line),json.loads(pl),json.loads(al))
            if audit.counts['frames']%100==0:print(json.dumps(dict(stage='full_sealed_fusion_numerics',run_id=plan['run_id'],**audit.counts)),flush=True)
        if ps.readline() or ass.readline():raise ValueError('unaccounted full stream frames')
    if audit.counts['frames']!=3316 or len(audit.previous)!=21:raise ValueError('full original native cohort required')
    for path,digest in audit.touched.items():
        if sha(ordinary(path))!=digest:raise ValueError('original raw input changed during full audit')
    for name,digest in plan['original_source_sha256'].items():
        if sha(ordinary(Path(plan['original_source_root'])/name))!=digest:raise ValueError('original source changed during full audit')
    for k in ['predictions','association','diagnostics']:
        if sha(plan[k])!=plan[k+'_sha256']:raise ValueError('sealed stream changed during full audit')
    if sha(plan_path)!=plan_hash:raise ValueError('frozen numerical plan changed')
    receipt=dict(kind='rbf_stage1_full_native_sealed_decision_conditional_independent_fusion_numerical_audit_v1',run_id=plan['run_id'],counts=audit.counts,full_sequences=21,all_full_native_source_component_node_and_conditional_temporal_numerics_passed=True,maximum_mean_difference=audit.maximum_mean,maximum_covariance_difference=audit.maximum_covariance,maximum_scalar_difference=audit.maximum_scalar,mean_atol=oracle.MEAN_ATOL,covariance_atol=oracle.COVARIANCE_ATOL,rtol=oracle.RTOL,original_covariance_preimages_exact=True,original_source_sha256=plan['original_source_sha256'],original_input_files=len(audit.touched),source_sha256=sha(__file__),oracle_sha256=sha(oracle.__file__),plan_sha256=plan_hash,numpy_version=np.__version__,python_version=platform.python_version(),conditioning='sealed original association alternatives and lifecycle decisions; previous actual full output for each one-step temporal update',independent_global_assignment_proof=False,independent_fresh_model_or_full_scratch_numeric_replay=False,causal_error_counterfactual_proven=False,original_strict_replay_pass=plan['strict_replay_pass'],stage1_qualified=False,paper_metrics_complete=False)
    print('RBF_FUSION_NUMERIC_JSON '+json.dumps(receipt,sort_keys=True),flush=True)


if __name__=='__main__':main()
