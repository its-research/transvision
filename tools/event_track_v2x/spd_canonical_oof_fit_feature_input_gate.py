"""Admit fit-only feature inputs; never admit them for held-out selection scoring."""
import hashlib
import json
from pathlib import Path
from run_cooptrack_official_oof_gpu4_filehost_v4 import validate_cohort


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def validate_inputs(root, package, expected_manifest_sha256, *, package_manifest_sha256):
    root = Path(root)
    validate_cohort(package)
    if digest(root / 'input-manifest.json') != expected_manifest_sha256:
        raise ValueError('fit-feature input manifest byte identity differs')
    m = json.loads((root / 'input-manifest.json').read_bytes())
    if (m.get('kind') != 'spd_canonical_oof_fit_feature_inputs_v1'
            or m.get('cohort') != 'canonical-oof-fit-feature-inputs'
            or m.get('fold_id') != package['fold_id']
            or m.get('fit_sequence_ids') != package['fit_sequence_ids']
            or m.get('excluded_held_out_sequence_ids') != package['held_out_sequence_ids']
            or m.get('training_package_manifest_sha256') != package_manifest_sha256
            or m.get('held_out_selection_scoring_eligible') is not False
            or m.get('predictions_generated') is not False
            or m.get('paper_eligible') is not False
            or m.get('source_all_train_input_admission_sha256') != '29214bb602a4637637a479636109e6bec6591282d7e2647c2d628bc8ebd967df'
            or m.get('canonical_fivefold_manifest_sha256') != package['canonical_fivefold_manifest_sha256']
            or m.get('official_split_sha256') != package['official_split_sha256']
            or m.get('infos_encoding') != 'pickle-protocol2-numpy-public-array-v1'
            or any(m.get(k) is not False for k in ('gt_payloads_in_package', 'gt_payloads_read', 'test_payloads_read', 'val_payloads_read'))):
        raise ValueError('inputs are not corresponding label-free fit complement')
    if set(m['fit_sequence_ids']) & set(m['excluded_held_out_sequence_ids']):
        raise ValueError('fit and held-out memberships overlap')
    sources = m.get('source_input_manifests', [])
    if len(sources) != 4 or sorted(r['fold_id'] for r in sources) != [f for f in range(5) if f != package['fold_id']]:
        raise ValueError('fit input must bind the four complementary input folds')
    paths = set()
    for r in m['payload_inventory']:
        relative = Path(r['path'])
        if relative.is_absolute() or '..' in relative.parts or r['path'] in paths:
            raise ValueError('unsafe or duplicate inference payload')
        paths.add(r['path'])
        p = root / relative
        if p.is_symlink() or not p.is_file() or p.stat().st_size != r['bytes'] or digest(p) != r['sha256']:
            raise ValueError('inference payload byte identity differs: ' + r['path'])
    expected = paths | {'input-manifest.json'}
    for side in ('vehicle-side', 'infrastructure-side'):
        for filename,key in [('image-pose-infos.pkl','infos_sha256'),('frame-index.json','frame_index_sha256'),('data_info.json','metadata_sha256')]:
            p=root/side/filename
            if p.is_symlink() or digest(p)!=m[side][key]:raise ValueError('bound metadata bytes differ')
            expected.add(side+'/'+filename)
        rows=json.loads((root/side/'frame-index.json').read_bytes())
        identities={(r['sequence_id'],r['frame_id']) for r in rows}
        if len(rows)!=m['frames'][side] or len(identities)!=len(rows) or {r['sequence_id'] for r in rows}!=set(m['fit_sequence_ids']):
            raise ValueError('fit-feature frame coverage differs')
        payloads={r['path']:r for r in m['payload_inventory']}
        for row in rows:
            if payloads[side+'/image/'+row['frame_id']+'.jpg']['sha256']!=row['image_sha256']:
                raise ValueError('frame image provenance differs')
    actual={p.relative_to(root).as_posix() for p in root.rglob('*') if p.is_file() or p.is_symlink()}
    if actual!=expected or any(p.is_symlink() for p in root.rglob('*')):
        raise ValueError('extra, missing or symlinked inference payload')
    split=json.loads((root/'train-split.json').read_bytes())
    if split!={'batch_split':{'train':m['fit_sequence_ids'],'val':[],'test':[],'test_A':[]}}:
        raise ValueError('validation/test or wrong train split')
    return m
