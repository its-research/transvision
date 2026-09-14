#!/usr/bin/env python3
"""Prepare full SPD train identity rows, never val/test or legacy pair features.

Only two previously audited, locally converted PKLs can be deserialized. The
old target audit is used for byte identities, not as temporal identity labels.
No network access, external publication, detector training or GT export.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import pickle
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from transvision.models.event_track_v2x.detection_cache_v2 import contained_file, sha_file
from transvision.models.event_track_v2x.forest_cache_stream import VerifiedForestCache
from transvision.models.event_track_v2x.forest_supervision import AnnotationIdentityIndex
from transvision.models.event_track_v2x.forest_tracking import ForestTrackingConfig
from transvision.models.event_track_v2x.forest_training_data import (
    exact_cooperative_links, frame_from_converted, prepare_training_rows,
)


TRAIN_CACHE_SHA256 = '1137740ecdf2aca7372536998ac485585ca89f4792bf68e287fa75d2e07a6fa0'
CONVERTED_SHA256 = {
    'vehicle-side': 'd4b328e4e83d73b09d6961aec05747c13647ead92b576d6605da3a8f2d2577bb',
    'infrastructure-side': 'e0e78a0bd2ce8584353780fb5fae2f33ba1822e7a40b22ad9884b960d226213e',
}


def preparation_sources():
    module = ROOT/'transvision/models/event_track_v2x'
    source_paths = [Path(__file__)]+[module/name for name in ('forest_training_data.py', 'forest_supervision.py',
        'forest_row_context.py', 'forest_tracking.py', 'forest_cache_stream.py', 'detection_cache_v2.py',
        'prediction_features.py', 'tracking_v2.py', 'recoverable_states.py', 'identity_forest.py', 'fusion.py', 'arrays.py')]
    return {p: sha_file(p) for p in source_paths}


def prepare(args):
    source_sha256 = preparation_sources()
    if sha_file(args.target_audit) != args.target_audit_sha256:
        raise ValueError('target audit identity differs')
    audit = json.loads(args.target_audit.read_bytes())
    if audit['converted_train_sha256'] != CONVERTED_SHA256:
        raise ValueError('target audit was not built from the sealed full train conversion')
    cache = VerifiedForestCache(args.cache, TRAIN_CACHE_SHA256)
    manifest = json.loads(cache.manifest_json)
    if manifest['split'] != 'train' or len(manifest['sequences']) != 46 or len(cache.index) != 16338:
        raise ValueError('full official train V2 cache required')
    frames = {}
    for source, side in enumerate(('vehicle-side', 'infrastructure-side')):
        path = contained_file(args.converted, side+'/spd_infos_temporal_train.pkl')
        # Known trusted local conversion bytes only, NEVER arbitrary external PKL.
        payload = path.read_bytes()
        if hashlib.sha256(payload).hexdigest() != CONVERTED_SHA256[side]:
            raise ValueError('untrusted or changed converted supervision; refusing pickle')
        infos = pickle.loads(payload)['infos']
        del payload
        for info in infos:
            frame = frame_from_converted(info, source)
            key = side, frame.frame_id
            if key in frames or (frame.sequence_id, side, frame.frame_id) not in cache.index:
                raise ValueError('converted GT outside sealed train cache or duplicated')
            _, meta = cache.index[(frame.sequence_id, side, frame.frame_id)]
            if json.loads(meta)['box_reference_timestamp_us'] != frame.timestamp_us:
                raise ValueError('converted GT timestamp differs')
            frames[key] = frame
        del infos
    if len(frames) != 16338 or {f.sequence_id for f in frames.values()} != set(manifest['sequences']):
        raise ValueError('converted supervision does not cover full train')
    path = contained_file(args.projection, 'cooperative/data_info.json')
    if sha_file(path) != audit['cooperative_metadata_sha256']:
        raise ValueError('cooperative metadata identity differs')
    pairs = json.loads(path.read_bytes())
    if (len(pairs) != 7445 or any(len({p[field] for p in pairs}) != len(pairs)
                                for field in ('vehicle_frame', 'infrastructure_frame'))):
        raise ValueError('full train cooperative pair count/uniqueness differs')
    inventory = {r['path']: r for r in audit['cooperative_label_inventory']}
    expected = {'cooperative/label/'+p['vehicle_frame']+'.json' for p in pairs}
    if set(inventory) != expected or len(inventory) != len(audit['cooperative_label_inventory']):
        raise ValueError('cooperative train label inventory differs')
    links, schedule, checked = [], [], {path: audit['cooperative_metadata_sha256']}
    for pair in pairs:
        scene = pair['vehicle_sequence']
        left, right = frames[('vehicle-side', pair['vehicle_frame'])], frames[('infrastructure-side', pair['infrastructure_frame'])]
        if not scene == pair['infrastructure_sequence'] == left.sequence_id == right.sequence_id:
            raise ValueError('cooperative pair crosses train sequences')
        relative = 'cooperative/label/'+left.frame_id+'.json'
        path = contained_file(args.projection, relative)
        if sha_file(path) != inventory[relative]['sha256'] or path.stat().st_size != inventory[relative]['bytes']:
            raise ValueError('cooperative label identity differs')
        checked[path] = inventory[relative]['sha256']
        links.extend(exact_cooperative_links(json.loads(path.read_bytes()), left, right))
        schedule.append(dict(sequence_id=scene, vehicle_frame=left.frame_id, infrastructure_frame=right.frame_id,
                             box_reference_timestamp_us=left.timestamp_us))
    index = AnnotationIdentityIndex((a for f in frames.values() for a in f.annotations), links)
    schedule.sort(key=lambda r: (r['sequence_id'], r['box_reference_timestamp_us']))
    if any(sha_file(p) != sha for p, sha in checked.items()):
        raise ValueError('cooperative supervision changed during read')
    provenance = dict(full_official_train_verified=True, converted_train_sha256=CONVERTED_SHA256,
        target_audit_sha256=args.target_audit_sha256, train_cache_sha256=TRAIN_CACHE_SHA256,
        detector_and_calibration_in_sample=True, out_of_fold_predictions=False,
        train_selection_holdout_verified=False, validation_or_test_inputs_read=False,
        temporal_labels_reconstructed_from_source_ids_and_exact_cooperative_tokens=True,
        producer_source_sha256={p.relative_to(ROOT).as_posix(): sha for p, sha in source_sha256.items()})
    def final_check():
        if any(sha_file(p) != sha for p, sha in source_sha256.items()) or sha_file(args.target_audit) != args.target_audit_sha256:
            raise ValueError('preparation sources or target audit changed')
    result = prepare_training_rows(cache, schedule, frames, index, args.output, config=ForestTrackingConfig(),
                                   provenance=provenance, finalize_check=final_check)
    return dict(manifest_sha256=sha_file(args.output/'manifest.json'), sequences=len(result['shards']),
                nodes=sum(s['nodes'] for s in result['shards']),
                supervised_rows=sum(s['supervised_rows'] for s in result['shards']),
                paper_eligible=False, training_performed=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('cache', 'converted', 'projection', 'target-audit', 'output'):
        parser.add_argument('--'+key, type=Path, required=True)
    parser.add_argument('--target-audit-sha256', required=True)
    print(json.dumps(prepare(parser.parse_args()), sort_keys=True))


if __name__ == '__main__':
    main()
