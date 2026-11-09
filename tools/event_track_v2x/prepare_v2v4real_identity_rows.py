#!/usr/bin/env python3
"""Dataset-specific full V2V train rows from independently admitted real inputs.

The common 203-feature architecture has three class slots; native merged
vehicle occupies slot zero. The original train name-group fit/selection roles
are retained explicitly. No checkpoint fitting or official-test label access.
"""
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from tools.event_track_v2x.v2v4real_motion_fit_inputs import read,sha
from tools.event_track_v2x.v2v4real_identity_supervision import source_supervision
from transvision.models.event_track_v2x.paper_native_cache import NativePaperCache
from transvision.models.event_track_v2x.forest_tracking import PaperForestTrackingConfig
from transvision.models.event_track_v2x.forest_training_data import prepare_training_rows


def cache_admission(value,seed):
    if (value.get('kind')!='v2v4real_native_9d_cache_independent_byte_numeric_readback_v1'
            or value.get('split')!='train' or value.get('seed')!=seed
            or value.get('native_nine_state_cache_independently_read_back') is not True
            or value.get('complete_source_frame_coverage_verified') is not True
            or value.get('frames')!=14210 or value.get('online_GT_read') is not False):
        raise ValueError('independently read-back full train 9D cache required before labels')


def prepare(args):
    pins={};out=Path(args.output).absolute()
    ca=json.loads(read(args.cache_admission,args.cache_admission_sha256,pins));cache_admission(ca,args.seed)
    part=json.loads(read(args.partition,args.partition_sha256,pins))
    if part['official_split']!='train' or part['official_test_used_for_partition'] is not False:
        raise ValueError('original train-only identity fit/selection partition required')
    accepted=json.loads(read(args.label_admission,args.label_admission_sha256,pins))
    if (accepted.get('kind')!='v2v4real_full_train_identity_label_independent_geometry_v1'
            or accepted.get('all_source_geometry_and_native_ids_verified') is not True
            or accepted.get('source_rows')!=14210 or accepted.get('official_test_read') is not False):
        raise ValueError('independently verified full train source geometry required')
    root=Path(args.labels).absolute();candidate_path=root/'candidate-receipt.json'
    c=json.loads(read(candidate_path,accepted['candidate_receipt_sha256'],pins))
    if c['split']!='train' or c['official_test_read'] is not False:raise ValueError('train-only native labels required')
    if c['input_sha256'].get(str(Path(args.partition).absolute()))!=args.partition_sha256:
        raise ValueError('original native label partition differs')
    rows=[json.loads(x) for x in read(root/'source-labels.jsonl',c['source_labels_sha256'],pins).splitlines()]
    frames,index,schedule=source_supervision(rows)
    if len(frames)!=14210 or len(schedule)!=7105 or {k[0] for k in frames}!=set(part['sequence_to_name_group']):
        raise ValueError('all 32 train sequences and full paired schedule required')
    cache=NativePaperCache(args.cache,ca['cache_manifest_sha256'])
    if (json.loads(cache.manifest_json)['dataset']!='v2v4real' or len(cache.index)!=14210 or set(cache.index)!=set(frames)):
        raise ValueError('exact native cache/supervision source support required')
    roles={}
    for sid,g in part['sequence_to_name_group'].items():
        role='identity_fit' if g in part['identity_fit'] else 'identity_selection' if g in part['identity_selection'] else None
        if role is None:raise ValueError('every source sequence needs original frozen identity role')
        roles[sid]=role
    source_paths=[Path(__file__),ROOT/'tools/event_track_v2x/v2v4real_identity_supervision.py']+sorted((ROOT/'transvision/models/event_track_v2x').glob('*.py'))
    sources={str(p):sha(p) for p in source_paths}
    provenance=dict(dataset='v2v4real',seed=args.seed,full_official_train_verified=True,native_class_scope=['vehicle'],
        network_class_slot_mapping={'car':'vehicle'},train_cache_sha256=ca['cache_manifest_sha256'],
        train_cache_independent_admission_sha256=args.cache_admission_sha256,
        full_train_labels_independent_admission_sha256=args.label_admission_sha256,
        original_nested_partition_sha256=args.partition_sha256,sequence_to_name_group=part['sequence_to_name_group'],
        sequence_identity_roles=roles,nominal_rate_hz=10,official_test_labels_read=False,
        upstream_sequence_isolated=part['upstream_sequence_isolated'],train_test_overlap_disclosed=True,
        physical_identity_continuity_verified=False,identity_model_fitted=False,producer_source_sha256=sources)
    def final_check():
        for p,h in {**pins,**sources}.items():
            if sha(p)!=h:raise ValueError('frozen supervision inputs or sources changed')
    result=prepare_training_rows(cache,schedule,frames,index,out,config=PaperForestTrackingConfig(),
        provenance=provenance,finalize_check=final_check)
    receipt=dict(kind='v2v4real_full_train_identity_rows_candidate_v1',seed=args.seed,
        manifest_sha256=sha(out/'manifest.json'),source_rows=len(frames),paired_frames=len(schedule),sequences=32,
        nodes=sum(r['nodes'] for r in result['shards']),supervised_rows=sum(r['supervised_rows'] for r in result['shards']),
        native_class_scope=['vehicle'],network_class_slot_mapping={'car':'vehicle'},
        annotation_audit=index.audit,original_nested_partition_sha256=args.partition_sha256,
        input_sha256=pins,source_sha256=sources,official_test_labels_read=False,identity_row_supervision_independently_verified=False,
        checkpoint_qualified=False,paper_metrics_complete=False)
    p=out/'candidate-receipt.json';p.open('x').write(json.dumps(receipt,indent=2,sort_keys=True)+'\n')
    print(json.dumps(dict(receipt=str(p),sha256=sha(p),nodes=receipt['nodes'],supervised_rows=receipt['supervised_rows'])),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--seed',type=int,choices=[1337,2027,3407],required=True)
    for n in ('cache','cache-admission','cache-admission-sha256','labels','label-admission','label-admission-sha256','partition','partition-sha256','output'):
        p.add_argument('--'+n,required=True)
    prepare(p.parse_args())
