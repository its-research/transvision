#!/usr/bin/env python3
"""Create a training-only GPU bundle after full independent native row admission.

No upload, enqueue, raw YAML/GT boxes/IDs, test arrays, or priority/CUDA branch
state packages. Physical GPU admission and cloud-byte readback are separate.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import tarfile

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from tools.event_track_v2x.v2v4real_identity_training_contract_native_v2 import FIT_CONFIG,admitted_roles


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024**2),b''):h.update(block)
    return h.hexdigest()


def ordinary(path):
    p=Path(path).absolute()
    if not p.is_file() or any(v.is_symlink() for v in (p,*p.parents)):
        raise ValueError('ordinary original input/source required')
    return p


def contained(root,name):
    p=Path(name)
    if not name or p.is_absolute() or '..' in p.parts or p.as_posix()!=name:
        raise ValueError('canonical contained input required')
    return ordinary(Path(root)/p)


def read(path,digest):
    p=ordinary(path)
    if sha(p)!=digest:raise ValueError('full admitted metadata byte identity differs')
    return json.loads(p.read_bytes())


def new(path,value):
    with Path(path).open('x') as stream:stream.write(json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+'\n')


def verify_archive(path,inventory):
    """Reopen every archive member and verify exact intended byte identities."""
    with tarfile.open(path,'r:gz') as archive:
        members=archive.getmembers()
        if len(members)!=len(inventory) or len({m.name for m in members})!=len(members) or set(m.name for m in members)!=set(inventory):
            raise ValueError('training bundle inventory differs')
        for member in members:
            p=Path(member.name);record=inventory[member.name]
            if not member.isfile() or p.is_absolute() or '..' in p.parts or p.as_posix()!=member.name or member.size!=record['bytes']:
                raise ValueError('training bundle contains unexpected/unsafe member')
            stream=archive.extractfile(member);h=hashlib.sha256()
            for block in iter(lambda:stream.read(8*1024**2),b''):h.update(block)
            if h.hexdigest()!=record['sha256']:raise ValueError('training bundle member bytes differ')


def prepare(args):
    data=Path(args.data).absolute();out=Path(args.output).absolute()
    manifest=read(data/'manifest.json',args.manifest_sha256)
    admitted=read(args.row_admission,args.row_admission_sha256)
    partition=read(args.partition,args.partition_sha256)
    roles=admitted_roles(manifest,admitted,partition,args.seed,args.manifest_sha256)
    if manifest['provenance']['original_nested_partition_sha256']!=args.partition_sha256:
        raise ValueError('original train partition binding differs')
    if out.exists() or any(p.is_symlink() for p in (out,*out.parents)):raise ValueError('fresh ordinary training input output required')
    files={'rows/manifest.json':ordinary(data/'manifest.json'),'row-admission.json':ordinary(args.row_admission),
        'partition.json':ordinary(args.partition)}
    for record in manifest['shards']:
        p=contained(data,record['path'])
        if sha(p)!=record['sha256']:raise ValueError('all original training shard bytes required')
        files['rows/'+record['path']]=p
    closure=read(args.source_closure,args.source_closure_sha256)
    if (closure.get('kind')!='native_v2_actual_training_top_level_import_source_closure'
            or closure.get('actual_top_level_imports_succeeded') is not True
            or closure.get('CUDA_branch_state_or_priority_modules_present') is not False
            or len(closure['files'])!=27 or set(closure['files'])!=set(closure['source_sha256'])
            or any('batched_branch_states' in n or 'priority' in n or 'persistent' in n for n in closure['files'])):
        raise ValueError('bound training-only source closure required')
    sources=[contained(ROOT,name) for name in closure['files']]
    if any(sha(ROOT/name)!=h for name,h in closure['source_sha256'].items()):
        raise ValueError('actual imported training closure source changed')
    files['training-source-closure.json']=ordinary(args.source_closure)
    source_inventory={p.relative_to(ROOT).as_posix():sha(ordinary(p)) for p in sources}
    files.update({'source/'+name:ordinary(ROOT/name) for name in source_inventory})
    semantic=dict(dataset='v2v4real',seed=args.seed,manifest_sha256=args.manifest_sha256,
        row_admission_sha256=args.row_admission_sha256,partition_sha256=args.partition_sha256,
        fit_config=FIT_CONFIG,source_sha256=source_inventory,source_closure_sha256=args.source_closure_sha256,native_target_contract='author_native_mapped_id_one_explicit_association_v2',
        selection='original_train_name_group_holdout_then_full_train_refit',all_rows_final_weight_forward=True)
    semantic_sha=hashlib.sha256(json.dumps(semantic,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
    plan=dict(kind='v2v4real_native_joint_identity_admitted_GPU_input_bundle_v2',semantic_training_identity=semantic_sha,
        semantic=semantic,fit_sequences=[r['sequence_id'] for r in roles['fit']],holdout_sequences=[r['sequence_id'] for r in roles['holdout']],
        native_class_scope=['vehicle'],network_class_slot_mapping={'car':'vehicle'},train_test_overlap_disclosed=True,
        upstream_sequence_isolated=partition['upstream_sequence_isolated'],physical_session_independence_claimed=False,
        raw_YAML_or_GT_boxes_or_IDs_or_official_test_arrays_in_bundle=False,training_targets_in_shards=True,
        GPU_cloud_publication_complete=False,physical_GPU_admission_complete=False,actual_training_started=False,
        blocked_priority_or_CUDA_branch_state_package_included=False,paper_metrics_complete=False)
    out.mkdir(parents=True);new(out/'bundle-plan.json',plan);files['bundle-plan.json']=out/'bundle-plan.json'
    inventory={name:dict(sha256=sha(p),bytes=p.stat().st_size) for name,p in files.items()}
    package=out/'input-bundle.tar.gz'
    with tarfile.open(package,'x:gz') as archive:
        for name,p in sorted(files.items()):archive.add(p,arcname=name,recursive=False)
    verify_archive(package,inventory)
    for name,p in files.items():
        if sha(p)!=inventory[name]['sha256']:raise ValueError('original admitted input/source changed while packaging')
    proof=dict(kind='v2v4real_native_identity_GPU_input_bundle_local_byte_readback_v2',seed=args.seed,
        semantic_training_identity=semantic_sha,package=dict(path=package.name,sha256=sha(package),bytes=package.stat().st_size),
        plan_sha256=sha(out/'bundle-plan.json'),inventory=inventory,complete_archive_inventory_and_member_bytes_read_back=True,
        full_native_row_admission_sha256=args.row_admission_sha256,actual_cloud_registered_bytes_independently_read=False,
        physical_GPU_admission_complete=False,actual_training_started=False,paper_metrics_complete=False)
    new(out/'local-byte-readback.json',proof)
    print(json.dumps(dict(receipt=str(out/'local-byte-readback.json'),sha256=sha(out/'local-byte-readback.json'),
        seed=args.seed,semantic_training_identity=semantic_sha,actual_training_started=False)),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--seed',type=int,choices=[1337,2027,3407],required=True)
    for name in ('data','manifest-sha256','row-admission','row-admission-sha256','partition','partition-sha256','source-closure','source-closure-sha256','output'):
        p.add_argument('--'+name,required=True)
    prepare(p.parse_args())
