#!/usr/bin/env python3
"""Recover fit-only cross-source IDs from retained ClearML native SPD labels.

No geometry, image features, held-out labels or val/test labels enter the output.
V2 validates numeric local ID plus unique annotation token; raw IDs remain intact.
Archive binding proves local asset identity, not official publication provenance.
"""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import zipfile


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda: f.read(8 * 1024**2), b''):
            h.update(b)
    return h.hexdigest()


def need(ok, message):
    if not ok:
        raise ValueError(message)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'),
                      allow_nan=False).encode()


def extract(archive, expected_sha, package_path, fit_root, output):
    need(not output.exists(), 'preserve previous output')
    need(archive.is_file() and not archive.is_symlink()
         and sha(archive) == expected_sha, 'native archive identity differs')
    package = json.loads(package_path.read_bytes())
    source = json.loads((fit_root / 'input-manifest.json').read_bytes())
    fit, held = package['fit_sequence_ids'], package['held_out_sequence_ids']
    need(len(set(fit) | set(held)) == 46 and not set(fit) & set(held)
         and source['cohort'] == 'fit-fold' and source['fold_id'] == package['fold_id']
         and source['fit_sequence_ids'] == fit
         and source['excluded_held_out_sequence_ids'] == held
         and sha(fit_root / 'input-manifest.json') == package['input_manifest_sha256']
         and source['official_val_read'] is False
         and source['test_or_test_A_read'] is False, 'canonical fit boundary differs')
    records = {r['path']: r for r in source['payload_inventory']}
    need(len(records) == len(source['payload_inventory']), 'duplicate fit inventory')
    indexes = {}
    for side in ('vehicle-side', 'infrastructure-side'):
        rows = json.loads((fit_root / side / 'data_info.json').read_bytes())
        indexes[side] = {r['frame_id']: r for r in rows}
        need(len(indexes[side]) == len(rows) and {r['sequence_id'] for r in rows} == set(fit),
             'native fit metadata coverage differs')
    output.mkdir()
    counts = dict(pairs=0, positive_links=0, unilateral_links=0, numeric_id_string_aliases=0)
    checked = {}
    with zipfile.ZipFile(archive) as z, (output / 'identity-mapping.jsonl').open('xb') as out:
        names = z.namelist()
        need(len(names) == len(set(names)), 'duplicate native ZIP members')
        meta_raw = z.read('V2X-Seq-SPD/cooperative/data_info.json')
        pairs = json.loads(meta_raw)
        seen = set()
        for pair in pairs:
            vs, ins = pair['vehicle_sequence'], pair['infrastructure_sequence']
            if vs not in fit or ins not in fit:
                continue
            need(vs == ins and vs not in held, 'cross-sequence cooperative pair')
            frames = (pair['vehicle_frame'], pair['infrastructure_frame'])
            need((vs, *frames) not in seen, 'duplicate cooperative pair')
            seen.add((vs, *frames))
            side_labels = []
            for side, frame in zip(('vehicle-side', 'infrastructure-side'), frames):
                row = indexes[side].get(frame)
                need(row is not None and row['sequence_id'] == vs, 'pair outside fit frame inventory')
                rel = side + '/' + row['label_lidar_std_path']
                need(not PurePosixPath(rel).is_absolute() and '..' not in PurePosixPath(rel).parts,
                     'unsafe label path')
                if rel not in checked:
                    raw = z.read('V2X-Seq-SPD/' + rel)
                    need(len(raw) == records[rel]['size_bytes']
                         and hashlib.sha256(raw).hexdigest() == records[rel]['sha256']
                         and sha(fit_root / rel) == records[rel]['sha256'],
                         'native single-side label does not bind fit conversion source')
                    labels = json.loads(raw)
                    by_token = {v['token']: str(v['track_id']) for v in labels}
                    need(len(by_token) == len(labels), 'duplicate native annotation token')
                    checked[rel] = by_token
                side_labels.append(checked[rel])
            member = 'V2X-Seq-SPD/cooperative/label/' + frames[0] + '.json'
            raw = z.read(member)
            labels = json.loads(raw)
            used = [set(), set()]
            links = []; bindings = []; identity_ids = set()
            for label in labels:
                need(label['veh_frame_id'] == frames[0] and label['inf_frame_id'] == frames[1],
                     'cooperative frame reference differs')
                identity_id = str(label['track_id'])
                need(identity_id.isascii() and identity_id.isdecimal() and int(identity_id) not in identity_ids, 'invalid or duplicate cooperative physical identity')
                identity_ids.add(int(identity_id))
                ids = (str(label['veh_track_id']), str(label['inf_track_id']))
                tokens = (label['veh_token'], label['inf_token'])
                for i, (identity, token) in enumerate(zip(ids, tokens)):
                    if identity == '-1':
                        need(token == '-1', 'missing source ID has nonmissing token')
                    else:
                        native_id = side_labels[i].get(token)
                        need(token not in used[i] and native_id is not None
                             and identity.isascii() and identity.isdecimal()
                             and native_id.isascii() and native_id.isdecimal()
                             and int(identity) == int(native_id),
                             'cooperative numeric ID / unique annotation token reference differs')
                        used[i].add(token)
                        counts['numeric_id_string_aliases'] += int(identity != native_id)
                        bindings.append(dict(side=('vehicle-side','infrastructure-side')[i], annotation_token=token, raw_cooperative_track_id=identity, raw_native_track_id=native_id, numeric_local_track_id=int(identity), cooperative_identity_id=identity_id))
                if '-1' not in ids:
                    links.append(dict(cooperative_identity_id=identity_id, vehicle_track_id=ids[0], infrastructure_track_id=ids[1],
                                      vehicle_token=tokens[0], infrastructure_token=tokens[1]))
                    counts['positive_links'] += 1
                else:
                    counts['unilateral_links'] += 1
            out.write(canonical(dict(sequence_id=vs, vehicle_frame_id=frames[0],
                                     infrastructure_frame_id=frames[1], positive_links=links, source_identity_bindings=bindings,
                                     cooperative_label_sha256=hashlib.sha256(raw).hexdigest())) + b'\n')
            counts['pairs'] += 1
    need(counts['pairs'] > 0 and counts['positive_links'] > 0, 'no real positive supervision')
    result = dict(kind='canonical_fit_only_native_cross_source_identity_annotation_tuple_mapping_v2',
                  fold_id=package['fold_id'], fit_sequence_ids=fit,
                  excluded_held_out_sequence_ids=held, native_archive_sha256=expected_sha,
                  cooperative_metadata_sha256=hashlib.sha256(meta_raw).hexdigest(),
                  package_sha256=sha(package_path), fit_input_manifest_sha256=package['input_manifest_sha256'],
                  mapping_sha256=sha(output / 'identity-mapping.jsonl'), counts=counts,
                  native_side_label_files_verified=len(checked), held_out_label_payloads_read=False,
                  val_test_label_payloads_read=False, geometry_exported=False,
                  raw_local_id_strings_preserved=True, numeric_id_and_unique_annotation_token_verified=True, unique_cooperative_identity_per_pair_verified=True, full_prediction_supervision_matched=False, paper_eligible=False, ETA='unknown')
    (output / 'manifest.json').write_bytes(canonical(result))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('archive', 'package', 'fit-root', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--archive-sha256', required=True)
    a = parser.parse_args()
    print(json.dumps(extract(a.archive, a.archive_sha256, a.package, a.fit_root, a.output)))
