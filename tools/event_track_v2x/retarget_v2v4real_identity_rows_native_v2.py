#!/usr/bin/env python3
"""Target-only correction of admitted native rows; predictions/support immutable.

Full raw score-ordered matching is repeated before selecting cached row indices.
Only positive/known/reason fields change. Old artifacts and live tools are kept.
"""
import argparse
from collections import Counter
import copy
import io
import json
from pathlib import Path
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from tools.event_track_v2x.v2v4real_motion_fit_inputs import read,sha
from tools.event_track_v2x.v2v4real_identity_supervision_native_v2 import source_supervision
from transvision.models.event_track_v2x.prediction_features import match_predictions
from transvision.models.event_track_v2x.forest_supervision import DetectionIdentityTarget,relation

TARGETS={'positive','known','reason'}


def labels_for_sequence(scene,a,gt,entries,cache,index,frames,pins):
    labels=[];offset=0
    rows=sorted((r for r in gt if r['sequence_id']==scene),key=lambda r:(r['frame_ordinal'],r['side'],r['frame_key']))
    for row in rows:
        entry,meta=entries[scene,row['side'],row['frame_key']]
        with np.load(io.BytesIO(read(cache/entry['arrays']['path'],entry['arrays']['sha256'],pins)),allow_pickle=False) as z:
            raw={k:z[k].copy() for k in z.files}
        if np.any(raw['class_indices']!=0):raise ValueError('merged vehicle source class differs')
        boxes=np.asarray(row['states_source_lwh'],dtype=np.float64).reshape(-1,7)
        matches=match_predictions(raw['states'],raw['raw_scores'],raw['class_indices'],boxes,np.zeros(len(boxes),dtype=int),distance=2.)
        selected=sorted((i for i,v in enumerate(raw['raw_scores']) if v>=.05),key=lambda i:(-float(raw['raw_scores'][i]),i))[:64]
        source=int(row['cav_id']);ann=frames[scene,row['side'],row['frame_key']].annotations
        for detection in selected:
            if (offset>=len(a['source']) or int(a['source'][offset])!=source or str(a['frame_id'][offset])!=row['frame_key']
                    or int(a['detection_index'][offset])!=detection or str(a['cache_sha256'][offset])!=entry['frame_sha256']):
                raise ValueError('admitted selected raw detection binding differs')
            j=int(matches[detection])
            labels.append(DetectionIdentityTarget(scene,source,None,status='unmatched_prediction') if j<0 else index.targets[ann[j].reference])
            offset+=1
    if offset!=len(a['source']):raise ValueError('selected node coverage differs')
    return labels


def corrected_targets(a,labels):
    n=len(labels);width=a['contexts'].shape[1]
    positive=np.zeros((n,width),dtype=bool);known=positive.copy();reasons=[]
    # Sufficient statistics of ALL earlier labels, not just retained parents.
    seen=set();identities=set();unknown=False;opposite=[False,False];unlinked=[False,False]
    for i,q in enumerate(labels):
        length=int(a['lengths'][i]);context=list(map(int,a['contexts'][i,:length]))
        if context[-1]!=i or any(j>=i for j in context[:-1]):raise ValueError('causal context required')
        if q.status!='matched':reason=q.status
        else:
            values=[relation(q,labels[j]) for j in context[:-1]]
            if any(v is True for v in values):
                positive[i,1:length]=[v is True for v in values]
                known[i,:length]=[True]+[v is not None for v in values];reason='parent'
            elif q.identity in identities:reason='out_of_support'
            elif (unknown or (unlinked[1-q.source_id] if q.cross_linked else opposite[1-q.source_id])
                    or any(v is None for v in values)):reason='uncertain_birth'
            else:positive[i,0]=True;known[i,:length]=True;reason='birth'
        reasons.append(reason)
        if q not in seen:
            seen.add(q)
            if q.status=='matched':
                identities.add(q.identity);opposite[q.source_id]=True
                if not q.cross_linked:unlinked[q.source_id]=True
            elif q.status!='unmatched_prediction':unknown=True
    return positive,known,np.asarray(reasons,dtype='U24')


def run(args):
    pins={};old=Path(args.rows).absolute();out=Path(args.output).absolute();cache=Path(args.cache).absolute()
    if out.exists() or any(p.is_symlink() for p in (out,*out.parents)):raise ValueError('fresh regular target version required')
    accepted=json.loads(read(args.row_admission,args.row_admission_sha256,pins))
    if not all(accepted.get(k) is True for k in ('all_original_cache_nodes_and_prediction_only_features_verified','all_causal_parent_contexts_independently_verified','all_native_temporal_cross_source_partial_targets_independently_verified')):
        raise ValueError('existing complete partial-row numeric/support admission required')
    candidate=json.loads(read(old/'candidate-receipt.json',accepted['candidate_receipt_sha256'],pins))
    m=json.loads(read(old/'manifest.json',candidate['manifest_sha256'],pins))
    if len(m['shards'])!=32 or accepted['nodes']!=candidate['nodes'] or m['provenance']['official_test_labels_read'] is not False:raise ValueError('full native train only required')
    labelroot=Path(args.labels).absolute()
    lp=json.loads(read(args.label_admission,m['provenance']['full_train_labels_independent_admission_sha256'],pins))
    if lp['all_source_geometry_and_native_ids_verified'] is not True:raise ValueError('native label admission required')
    lc=json.loads(read(labelroot/'candidate-receipt.json',lp['candidate_receipt_sha256'],pins))
    gt=[json.loads(x) for x in read(labelroot/'source-labels.jsonl',lc['source_labels_sha256'],pins).splitlines()]
    frames,index,_=source_supervision(gt)
    cm=json.loads(read(cache/'manifest.json',m['provenance']['train_cache_sha256'],pins));entries={}
    if cm['gt_in_cache'] is not False or cm['split']!='train':raise ValueError('prediction-only train cache required')
    for entry in cm['frames']:
        meta=json.loads(read(cache/entry['metadata']['path'],entry['metadata']['sha256'],pins))
        key=meta['sequence_id'],meta['side'],meta['frame_id']
        if key in entries or meta['arrays_sha256']!=entry['arrays']['sha256']:raise ValueError('unique raw metadata binding required')
        entries[key]=entry,meta
    if set(entries)!=set(frames):raise ValueError('complete source coverage required')
    sources={str(p):sha(p) for p in [Path(__file__),ROOT/'tools/event_track_v2x/v2v4real_identity_supervision_native_v2.py',ROOT/'transvision/models/event_track_v2x/prediction_features.py',ROOT/'transvision/models/event_track_v2x/forest_supervision.py']}
    out.mkdir(parents=True);records=[];total=Counter();cross=temporal=0
    try:
        for record in m['shards']:
            with np.load(io.BytesIO(read(old/record['path'],record['sha256'],pins)),allow_pickle=False) as z:a={k:z[k].copy() for k in z.files}
            labels=labels_for_sequence(record['sequence_id'],a,gt,entries,cache,index,frames,pins)
            a['positive'],a['known'],a['reason']=corrected_targets(a,labels)
            mask=np.arange(a['contexts'].shape[1])[None,:]<a['lengths'][:,None]-1
            parent=a['contexts'].copy();parent[~mask]=0
            p=a['positive'][:,1:];valid=mask[:,:-1];src=a['source'][parent[:,:-1]]
            ce=int(np.count_nonzero(p&valid&(src!=a['source'][:,None])));te=int(np.count_nonzero(p&valid&(src==a['source'][:,None])))
            cross+=ce;temporal+=te;counts=dict(Counter(a['reason'].tolist()));total.update(counts)
            path=out/record['path']
            with path.open('xb') as f:np.savez_compressed(f,**a)
            records.append(dict(record,sha256=sha(path),reason_counts=counts,supervised_rows=counts.get('birth',0)+counts.get('parent',0),cross_source_positive_parent_edges=ce,temporal_positive_parent_edges=te))
            print(json.dumps(dict(stage='native_target_only_v2',sequence_id=record['sequence_id'],completed_sequences=len(records),cross_source_positive_parent_edges=ce)),flush=True)
        if cross<=0 or temporal<=0:raise ValueError('both actual supervision types required')
        for p,h in {**pins,**sources}.items():
            if sha(p)!=h:raise ValueError('target-only inputs changed')
        new=copy.deepcopy(m);new['shards']=records;new['annotation_audit']=index.audit
        new['provenance'].update(native_target_contract='author_native_mapped_id_one_explicit_association_v2',reused_partial_row_admission_sha256=args.row_admission_sha256,prediction_and_context_fields_reused_without_change=True,producer_source_sha256=sources)
        (out/'manifest.json').open('x').write(json.dumps(new,indent=2,sort_keys=True)+'\n')
        receipt=dict(kind='v2v4real_full_train_identity_rows_native_target_v2_candidate',seed=candidate['seed'],nodes=candidate['nodes'],sequences=32,source_rows=14210,
            manifest_sha256=sha(out/'manifest.json'),annotation_audit=index.audit,reason_counts=dict(total),cross_source_positive_parent_edges=cross,temporal_positive_parent_edges=temporal,
            reused_rows_root=str(old),reused_row_admission_path=str(Path(args.row_admission).absolute()),reused_row_admission_sha256=args.row_admission_sha256,
            changed_array_fields=sorted(TARGETS),official_test_labels_read=False,labels_in_model_inputs=False,source_sha256=sources,input_sha256=pins,
            identity_row_supervision_independently_verified=False,checkpoint_qualified=False,paper_metrics_complete=False)
        p=out/'candidate-receipt.json';p.open('x').write(json.dumps(receipt,indent=2,sort_keys=True)+'\n')
        print(json.dumps(dict(receipt=str(p),sha256=sha(p),cross_positive=cross,temporal_positive=temporal)),flush=True)
    except BaseException as error:
        (out/'failure.json').open('x').write(json.dumps(dict(error_type=type(error).__name__,error=str(error),completed_shards=len(records),partial_outputs_not_training_ready=True),indent=2)+'\n');raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('rows','row-admission','row-admission-sha256','cache','labels','label-admission','output'):p.add_argument('--'+name,required=True)
    run(p.parse_args())
