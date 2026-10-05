#!/usr/bin/env python3
"""Materialize a separately frozen overlap-controlled V2V4Real train view."""
import argparse
import hashlib
import json
import os
from pathlib import Path

from tools.event_track_v2x.prepare_v2v4real_inputs import inventory_native_root
from transvision.models.event_track_v2x.experiment_progress import ExperimentProgress


def sha_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def canonical(data):
    return (json.dumps(data, sort_keys=True, separators=(',', ':')) + '\n').encode()


def plan(partition_path, partition_sha256, membership_path, membership_sha256):
    for path, expected in ((partition_path, partition_sha256),
                           (membership_path, membership_sha256)):
        path = Path(path)
        if not path.is_file() or path.is_symlink() or sha_file(path) != expected:
            raise ValueError('frozen input identity differs')
    partition = json.loads(Path(partition_path).read_bytes())
    membership = json.loads(Path(membership_path).read_bytes())
    if (partition.get('kind') != 'v2v4real_formal_nested_training_partition_v1'
            or membership.get('kind') != 'v2v4real_overlap_controlled_train_membership_v1'
            or membership.get('variant_id') != 'v2v4real-train-overlap-controlled-v1'
            or membership.get('protocol_id') != partition.get('protocol_id')
            or membership.get('source_partition_sha256') != partition_sha256
            or membership.get('source_catalog_sha256') != partition.get('catalog_sha256')
            or membership.get('official_archives_modified') is not False
            or membership.get('official_test_modified') is not False
            or membership.get('formal_independent_test_eligible') is not False
            or membership.get('physical_session_provenance_verified') is not False):
        raise ValueError('overlap-controlled source contract differs')
    original = partition['sequence_to_name_group']
    excluded = set(membership['excluded_train_sequences'])
    retained = membership['retained_train_sequence_to_name_group']
    if (len(original) != membership['source_official_train_sequence_count']
            or len(retained) != membership['retained_train_sequence_count']
            or set(retained) != set(original) - excluded
            or any(original[sequence] != group for sequence, group in retained.items())
            or len(excluded) != 3 or len(retained) != 29
            or membership.get('all_known_exact_overlap_excluded') is not True):
        raise ValueError('train membership subtraction differs')
    excluded_groups = {original[sequence] for sequence in excluded}
    if len(excluded_groups) != 1:
        raise ValueError('exact-overlap group differs')
    roles = {}
    for role in ('detector_fit', 'calibration_fit', 'identity_fit', 'identity_selection'):
        expected = set(partition[role]) - excluded_groups
        actual = set(membership['role_groups_after_exclusion'][role])
        if expected != actual:
            raise ValueError('role groups differ: ' + role)
        roles[role] = actual
    if not roles['detector_fit'] or not roles['calibration_fit'] or roles['detector_fit'] & roles['calibration_fit']:
        raise ValueError('invalid detector role partition')
    if (roles['identity_fit'] != roles['detector_fit'] | roles['calibration_fit']
            or roles['identity_selection'] & roles['identity_fit']):
        raise ValueError('identity fit/selection role boundary differs')
    selected = {role: sorted(sequence for sequence, group in retained.items() if group in roles[role])
                for role in ('detector_fit', 'calibration_fit', 'identity_selection')}
    if (set(selected['detector_fit']) & set(selected['calibration_fit'])
            or set(selected['identity_selection']) &
            (set(selected['detector_fit']) | set(selected['calibration_fit']))
            or sum(map(len, selected.values())) != len(retained)):
        raise ValueError('retained sequence role coverage differs')
    return original, excluded, selected


def materialize(source_roots, partition_path, partition_sha256,
                membership_path, membership_sha256, output):
    original, excluded, selected = plan(partition_path, partition_sha256,
                                        membership_path, membership_sha256)
    output = Path(output).absolute()
    if output.exists() or output.is_symlink() or output.parent.is_symlink():
        raise ValueError('fresh ordinary output required')
    sources = {}
    for root in map(lambda value: Path(value).absolute(), source_roots):
        if root.is_symlink() or not root.is_dir():
            raise ValueError('ordinary native source root required')
        inventory = inventory_native_root(root)
        for sequence in inventory['sequences']:
            if sequence in sources:
                raise ValueError('duplicate sequence across source roots')
            sources[sequence] = root / sequence
    if set(sources) != set(original):
        raise ValueError('full source train coverage differs from frozen partition')
    if any(sequence in selected['detector_fit'] + selected['calibration_fit'] for sequence in excluded):
        raise ValueError('excluded sequence selected')
    source_items = {}
    file_count = 0
    for sequence in (selected['detector_fit'] + selected['calibration_fit']
                     + selected['identity_selection']):
        items = sorted(sources[sequence].rglob('*'))
        if any(item.is_symlink() or (not item.is_file() and not item.is_dir()) for item in items):
            raise ValueError('unsupported source entry')
        source_items[sequence] = items
        file_count += sum(item.is_file() for item in items)
    output.mkdir()
    records = []
    progress = ExperimentProgress('v2v4real_overlap_controlled_materialization_files',
                                  file_count)
    role_directories = {'detector_fit': 'train', 'calibration_fit': 'validate',
                        'identity_selection': 'identity_selection'}
    for role, sequences in selected.items():
        role_dir = output / role_directories[role]
        role_dir.mkdir()
        for sequence in sequences:
            source = sources[sequence]
            target = role_dir / sequence
            target.mkdir()
            for item in source_items[sequence]:
                relative = item.relative_to(source)
                destination = target / relative
                if item.is_symlink() or (not item.is_file() and not item.is_dir()):
                    raise ValueError('unsupported source entry')
                if item.is_dir():
                    destination.mkdir()
                else:
                    before = item.stat()
                    digest = sha_file(item)
                    os.link(item, destination)
                    after = item.stat()
                    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
                        raise ValueError('source changed during materialization')
                    records.append({'role': role, 'sequence_id': sequence,
                                    'path': destination.relative_to(output).as_posix(),
                                    'bytes': after.st_size, 'sha256': digest})
                    progress.update(len(records))
    receipt = {'kind': 'v2v4real_overlap_controlled_materialization_v1',
               'variant_id': 'v2v4real-train-overlap-controlled-v1',
               'partition_sha256': partition_sha256,
               'membership_sha256': membership_sha256,
               'detector_fit_sequences': selected['detector_fit'],
               'calibration_fit_sequences': selected['calibration_fit'],
               'identity_selection_sequences': selected['identity_selection'],
               'excluded_train_sequences': sorted(excluded),
               'official_test_included': False,
               'physical_session_provenance_verified': False,
               'formal_independent_test_eligible': False,
               'training_not_run': True,
               'hard_links_used': True,
               'raw_yaml_may_contain_gt': True,
               'inventory': records}
    (output / 'materialization-receipt.json').write_bytes(canonical(receipt))
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-roots', nargs='+', required=True)
    parser.add_argument('--partition', required=True)
    parser.add_argument('--partition-sha256', required=True)
    parser.add_argument('--membership', required=True)
    parser.add_argument('--membership-sha256', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    receipt = materialize(args.source_roots, args.partition, args.partition_sha256,
                          args.membership, args.membership_sha256, args.output)
    print(json.dumps({'detector_fit_sequences': len(receipt['detector_fit_sequences']),
                      'calibration_fit_sequences': len(receipt['calibration_fit_sequences']),
                      'identity_selection_sequences': len(receipt['identity_selection_sequences']),
                      'membership_sha256': receipt['membership_sha256']}))


if __name__ == '__main__':
    main()
