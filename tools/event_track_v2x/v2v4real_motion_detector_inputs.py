"""Read admitted native train detector payloads for new motion dependencies.

This is a consumer of existing original raw exports, not a detector rerun or
new raw acceptance. Caller pins manifest/index/prior admission and train frame
selection before iteration. No official-test labels or inferred velocities enter.
"""
import hashlib
import io
import json
from pathlib import PurePosixPath
import tarfile
import numpy as np

ARRAYS = {'states_source_legacy7', 'raw_scores', 'class_indices', 'appearance', 'appearance_valid'}


def selected_payloads(manifest, index, selected, archive):
    # This check runs before opening even an archive supplied as a callable.
    if (manifest.get('kind') != 'v2v4real_train_raw_native_features_v1'
            or manifest.get('split') != 'train' or manifest.get('official_test_read') is not False
            or manifest.get('gt_read') is not False
            or manifest.get('candidate_protocol') != 'rbf-all-class-top64-v1'
            or manifest.get('matching_candidate_rule') != 'raw_score_ge_0.05_stable_top64_before_nms'):
        raise ValueError('original train-only all-class raw envelope required')
    if not selected or len(index) != manifest['rows']:
        raise ValueError('nonempty train selection and full bound index required')
    rows, names = {}, {}
    for row in index:
        key = row['sequence_id'], row['frame_id'], row['side']
        path = row['payload'];p = PurePosixPath(path)
        if (p.is_absolute() or '..' in p.parts or p.as_posix() != path
                or len(p.parts) != 2 or p.parts[0] != 'payloads' or p.suffix != '.npz'
                or key in rows or path in names):
            raise ValueError('unique contained original row/payload required')
        rows[key], names[path] = row, key
    if not set(selected) <= set(rows):
        raise ValueError('train selection not covered by original raw index')
    seen, result = set(), {}
    stream = archive() if callable(archive) else archive
    with tarfile.open(fileobj=stream, mode='r:*') as tar:
        for member in tar:
            if not member.isfile() or member.name in seen:
                raise ValueError('ordinary unique archive members required')
            seen.add(member.name)
            if member.name == 'manifest.json':
                if json.loads(tar.extractfile(member).read()) != manifest:
                    raise ValueError('original archive manifest differs')
            elif member.name == 'predictions.jsonl':
                b = tar.extractfile(member).read()
                if (hashlib.sha256(b).hexdigest() != manifest['rows_sha256']
                        or [json.loads(line) for line in b.splitlines()] != index):
                    raise ValueError('original archive index bytes/rows differ')
            elif member.name == 'anchors.npy':
                if hashlib.sha256(tar.extractfile(member).read()).hexdigest() != manifest['anchor_file_sha256']:
                    raise ValueError('original archive anchors SHA differs')
            elif member.name not in names:
                raise ValueError('unexpected original archive member')
            elif names[member.name] in selected:
                key = names[member.name];row = rows[key]
                b = tar.extractfile(member).read()
                if hashlib.sha256(b).hexdigest() != row['payload_sha256']:
                    raise ValueError('selected original payload SHA differs')
                with np.load(io.BytesIO(b), allow_pickle=False) as z:
                    if set(z.files) != ARRAYS:
                        raise ValueError('exact GT-free native raw arrays required')
                    arrays = {k: z[k].copy() for k in ARRAYS}
                n = row['candidates'];s = arrays['states_source_legacy7']
                if (type(n) is not int or not 0 <= n <= 64 or s.shape != (n, 7)
                        or s.dtype not in (np.dtype('float32'), np.dtype('float64'))
                        or not np.isfinite(s).all() or np.any(s[:, 3:6] <= 0)
                        or arrays['class_indices'].shape != (n,) or arrays['class_indices'].dtype.kind not in 'iu'
                        or arrays['raw_scores'].shape != (n,) or not np.isfinite(arrays['raw_scores']).all()
                        or np.any(arrays['raw_scores'] < .05)):
                    raise ValueError('original all-class candidate count/state/score differs')
                row_sha = hashlib.sha256(json.dumps(row, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()
                result[key] = dict(raw_row=row, raw_row_sha256=row_sha, payload_bytes=b, arrays=arrays)
    if seen != set(names) | {'anchors.npy', 'manifest.json', 'predictions.jsonl'} or set(result) != set(selected):
        raise ValueError('original archive/selected-row coverage differs')
    return result
