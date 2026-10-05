"""Read exact canonical fit supervision; never open held-out/val/test PKLs.

Only geometry, coarse class and source-local IDs are returned. Velocity targets
must be reconstructed from past fit GT by the separate example collector.
This loader does not fit calibration or certify prediction-cache acceptance.
"""
import hashlib
import io
import json
from pathlib import Path
import pickle
import sys

import numpy as np

from run_cooptrack_official_oof_gpu4_filehost_v4 import validate_cohort

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
from transvision.models.event_track_v2x.detection_cache_v2 import CLASSES

SIDES = ('vehicle-side', 'infrastructure-side')


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()


def need(ok, message):
    if not ok:
        raise ValueError(message)


def legacy_bytes(value, encoding='latin1', errors='strict'):
    need(type(value) is str and encoding in ('latin1', 'latin-1') and errors == 'strict',
         'unsupported legacy bytes')
    return value.encode('latin1')


class RestrictedSupervision(pickle.Unpickler):
    def find_class(self, module, name):
        allowed = {('numpy', 'dtype'): np.dtype, ('numpy', 'ndarray'): np.ndarray,
                   ('numpy.core.multiarray', '_reconstruct'): np.core.multiarray._reconstruct,
                   ('__builtin__', 'bytes'): bytes, ('_codecs', 'encode'): legacy_bytes}
        need((module, name) in allowed, 'forbidden supervision pickle global')
        return allowed[(module, name)]


def load_fit_supervision(converted, package_path, fit_inputs):
    converted, package_path, fit_inputs = map(Path, (converted, package_path, fit_inputs))
    package = json.loads(package_path.read_bytes())
    validate_cohort(package)
    fit, held = package['fit_sequence_ids'], package['held_out_sequence_ids']
    inputs = json.loads((fit_inputs / 'input-manifest.json').read_bytes())
    need(inputs.get('kind') == 'spd_canonical_oof_fit_feature_inputs_v1'
         and inputs.get('fold_id') == package['fold_id']
         and inputs.get('fit_sequence_ids') == fit
         and inputs.get('excluded_held_out_sequence_ids') == held
         and inputs.get('training_package_manifest_sha256') == sha(package_path)
         and inputs.get('held_out_selection_scoring_eligible') is False,
         'supervision input is not corresponding canonical fit complement')
    path = converted / 'conversion-manifest.json'
    need(not path.is_symlink() and sha(path) == package['conversion_manifest_sha256'],
         'conversion differs from frozen training package')
    manifest = json.loads(path.read_bytes())
    body = dict(manifest)
    content = body.pop('content_sha256')
    need(hashlib.sha256(json.dumps(body, sort_keys=True, separators=(',', ':'),
                                  ensure_ascii=False, allow_nan=False).encode()).hexdigest() == content,
         'conversion content checksum differs')
    need(manifest['kind'] == 'cooptrack_fit_conversion_v1'
         and manifest['cohort'] == 'fit-fold' and manifest['fold_id'] == package['fold_id']
         and manifest['fit_sequence_ids'] == fit
         and manifest['input_manifest_sha256'] == package['input_manifest_sha256']
         and manifest['raw_labels_modified'] is False
         and manifest['forecasting_targets_generated'] is False
         and manifest['removed_frames'] == manifest['interpolated_annotations'] == 0,
         'supervision conversion boundary differs')
    inventory = {r['path']: r for r in manifest['inventory']}
    need(len(inventory) == len(manifest['inventory']), 'duplicate conversion inventory')
    rows, counts = {}, {}
    for side in SIDES:
        index_path = fit_inputs / side / 'frame-index.json'
        need(sha(index_path) == inputs[side]['frame_index_sha256'], 'fit frame index differs')
        index = {r['frame_id']: r for r in json.loads(index_path.read_bytes())}
        need(len(index) == inputs['frames'][side], 'duplicate or missing fit index frames')
        relative = side + '/spd_infos_temporal_train.pkl'
        record, path = inventory[relative], converted / relative
        need(not path.is_symlink() and path.stat().st_size == record['size_bytes']
             and sha(path) == record['sha256'], 'fit supervision pickle bytes differ')
        data = RestrictedSupervision(io.BytesIO(path.read_bytes())).load()
        count = annotations = 0
        for info in data['infos']:
            key = (side, info['scene_token'], info['token'])
            row = index.get(info['token'])
            need(key not in rows and key[1] in fit and key[1] not in held
                 and row is not None and row['sequence_id'] == key[1]
                 and info['timestamp'] == row['box_reference_timestamp_us'],
                 'supervision contains held-out, duplicate or mismatched frame')
            boxes, tracks = np.asarray(info['gt_boxes'], dtype=float), np.asarray(info['gt_inds'])
            names, tokens = list(info['gt_names']), list(info['anno_tokens'])
            need(boxes.ndim == 2 and boxes.shape[1] == 7
                 and tracks.shape == (len(boxes),) and len(names) == len(tokens) == len(boxes)
                 and len(set(tokens)) == len(tokens) and np.isfinite(boxes).all()
                 and np.all(boxes[:, 3:6] > 0) and (not len(tracks) or tracks.dtype.kind in 'iu'),
                 'invalid fit supervision geometry or IDs')
            rows[key] = {'state': np.concatenate([boxes, np.full((len(boxes), 2), np.nan)], axis=1),
                         'classes': np.asarray([CLASSES.index(str(n)) if str(n) in CLASSES else -1
                                                for n in names], dtype=np.int64),
                         'tracks': [str(int(t)).zfill(6) for t in tracks], 'tokens': tokens}
            count += 1
            annotations += len(boxes)
        need({k for k in rows if k[0] == side} ==
             {(side, r['sequence_id'], r['frame_id']) for r in index.values()},
             'supervision does not exactly cover fit prediction frames')
        counts[side] = {'frames': count, 'annotations': annotations, 'pickle_sha256': record['sha256']}
    proof = {'kind': 'canonical_oof_fit_calibration_supervision_readback',
             'fold_id': package['fold_id'], 'fit_sequence_ids': fit,
             'excluded_held_out_sequence_ids': held, 'counts': counts,
             'conversion_manifest_sha256': sha(converted / 'conversion-manifest.json'),
             'fit_input_manifest_sha256': sha(fit_inputs / 'input-manifest.json'),
             'package_manifest_sha256': sha(package_path), 'held_out_gt_read': False,
             'val_or_test_read': False, 'future_velocity_fields_used': False,
             'calibration_fitted': False, 'formal_v2_ready': False, 'paper_eligible': False}
    return rows, proof
