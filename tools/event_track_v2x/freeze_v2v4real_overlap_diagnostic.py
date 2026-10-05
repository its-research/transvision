#!/usr/bin/env python3
"""Freeze completed v1 diagnostic detector artifacts with byte-level lineage checks.

The resulting freeze is diagnostic only. A separate GPU acceptance task must
verify the checkpoint tensor envelope and model forward pass.
"""
import argparse
import datetime
import hashlib
import json
from pathlib import Path
from urllib.parse import urlparse

SOURCE_BY_SEED = {
    1337: ('669833fb759742a4a857f99f87341e0b',
           '7f58517f4c62452cbef9b98c79ff1420',
           '93b3f63cb1058c725224e275c713566e2a11090812026060684c77867a3c3a69', 'A100'),
    2027: ('f945af3976e34a479e40b56ef2a7be24',
           '7f58517f4c62452cbef9b98c79ff1420',
           '93b3f63cb1058c725224e275c713566e2a11090812026060684c77867a3c3a69', 'A100'),
    3407: ('fd7a87ec5ffa43c3b28efc48bdc9b7e3',
           '79aeb14f8d4641f381150c3f912f623f',
           '122157453fa481ce27e9adb97528990bab439c8b9e30d5687229da6a57744f1a', '5090'),
}
MEMBERSHIP_SHA = '0f429069e56c1525484d32cb7c9ec4affd49bc5eee5a724d16f3cf44a6c9b4d9'
PARTITION_SHA = '02373d0f59ca4e88757b6b3c3c22d9afda0ff94fe47f0f7d10f80211876bbcb2'
CONFIG_SHA = '138c4ad3508fdd7061f5b290c83ad8f0772fea33f0af0e2be56330423a3c92d1'
ARTIFACT_FILES = {
    'best-checkpoint': 'best.pth',
    'training-receipt': 'training-receipt.json',
    'epoch-log': 'epochs.jsonl',
    'pip-report': 'pip-report.json',
    'publication-receipt': 'publication-receipt.json',
}
FILE_HOSTS = {'10.100.34.118', '10.100.35.118'}
AUTHENTICATED_FILE_HOST = '10.100.35.118'


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def copy_artifact_with_readback(artifact, target):
    from clearml.storage.helper import StorageHelper
    uri = urlparse(artifact.url)
    if (uri.scheme != 'http' or uri.hostname not in FILE_HOSTS
            or uri.port != 8081):
        raise ValueError('artifact is outside the designated private file service')
    url = artifact.url.replace(uri.hostname + ':8081',
                               AUTHENTICATED_FILE_HOST + ':8081', 1)
    digest = hashlib.sha256()
    size = 0
    with target.open('xb') as stream:
        for block in StorageHelper.get(url).download_as_stream(url):
            stream.write(block)
            digest.update(block)
            size += len(block)
    if size != artifact.size or digest.hexdigest() != artifact.hash:
        raise ValueError('ClearML artifact readback differs')


def parse_concatenated_json(raw):
    """The frozen trainer appends canonical JSON objects without newlines."""
    text = raw.decode()
    decoder = json.JSONDecoder()
    rows = []
    offset = 0
    while offset < len(text):
        while offset < len(text) and text[offset].isspace():
            offset += 1
        if offset == len(text):
            break
        row, offset = decoder.raw_decode(text, offset)
        if not isinstance(row, dict):
            raise ValueError('epoch log contains a non-object')
        rows.append(row)
    return rows


def verify_receipts(seed, task_id, source_id, source_sha, family, paths):
    training = json.loads(paths['training-receipt'].read_bytes())
    publication = json.loads(paths['publication-receipt'].read_bytes())
    if (training.get('kind') != 'v2v4real_overlap_controlled_detector_diagnostic_training_receipt_v1'
            or training.get('seed') != seed or training.get('protocol_id') != 'v2v4real-nominal-10hz-formal-v1'
            or training.get('variant_id') != 'v2v4real-train-overlap-controlled-v1'
            or training.get('paper_eligible') is not False
            or training.get('formal_independent_test_eligible') is not False
            or training.get('physical_session_provenance_verified') is not False
            or training.get('official_test_used') is not False
            or training.get('identity_selection_used') is not False
            or training.get('epochs') != 60 or training.get('batch_size_per_rank') != 8
            or training.get('world_size') != 4 or training.get('config_sha256') != CONFIG_SHA
            or training.get('partition_sha256') != PARTITION_SHA):
        raise ValueError('diagnostic training receipt differs')
    if (publication.get('kind') != 'v2v4real_overlap_controlled_detector_diagnostic_publication_v1'
            or publication.get('task_id') != task_id
            or publication.get('source_task_id') != source_id
            or publication.get('source_sha256') != source_sha
            or publication.get('membership_sha256') != MEMBERSHIP_SHA
            or publication.get('partition_sha256') != PARTITION_SHA
            or publication.get('paper_eligible') is not False
            or publication.get('formal_independent_test_eligible') is not False
            or publication.get('variant_id') != 'v2v4real-train-overlap-controlled-v1'
            or publication.get('world_size') != 4):
        raise ValueError('diagnostic publication receipt differs')
    if family == '5090' and publication.get('gpu_family') != '5090':
        raise ValueError('5090 family binding differs')
    if family == 'A100' and not publication.get('worker_id', '').startswith('10.100.34.18-A100:'):
        raise ValueError('A100 worker binding differs')
    for key in ('best-checkpoint', 'training-receipt', 'epoch-log', 'pip-report'):
        row = publication['artifacts'][key]
        if row['sha256'] != sha(paths[key]) or row['bytes'] != paths[key].stat().st_size:
            raise ValueError('published artifact digest differs: ' + key)
    epochs = parse_concatenated_json(paths['epoch-log'].read_bytes())
    if len(epochs) != 60 or [r.get('epoch') for r in epochs] != list(range(1, 61)):
        raise ValueError('epoch log is incomplete')
    if any(r.get('seed') != seed or r.get('world_size') != 4
           or r.get('batch_size_per_rank') != 8 for r in epochs):
        raise ValueError('epoch log topology differs')
    best = min(epochs, key=lambda r: (r['validation_loss'], r['epoch']))
    selected = training.get('selected')
    if selected != {'epoch': best['epoch'], 'validation_loss': best['validation_loss']}:
        raise ValueError('best checkpoint selection differs from full epoch log')
    if epochs[-1].get('best_epoch') != best['epoch']:
        raise ValueError('final epoch selection differs')
    return training, publication, best


def freeze(seed, output):
    from clearml import Task
    task_id, source_id, source_sha, family = SOURCE_BY_SEED[seed]
    task = Task.get_task(task_id=task_id)
    if str(task.status) != 'completed':
        raise ValueError(f'training task {task_id} is {task.status}; no checkpoint freeze')
    source = Task.get_task(task_id=source_id)
    if (str(source.status) != 'completed' or source.artifacts['source'].hash != source_sha):
        raise ValueError('source publication differs')
    if output.exists() or output.is_symlink():
        raise ValueError('fresh output directory required')
    output.mkdir(parents=True)
    paths = {}
    for key, name in ARTIFACT_FILES.items():
        artifact = task.artifacts[key]
        target = output / name
        copy_artifact_with_readback(artifact, target)
        paths[key] = target
    training, publication, best = verify_receipts(seed, task_id, source_id, source_sha,
                                                   family, paths)
    receipt = {'kind': 'v2v4real_overlap_controlled_v1_diagnostic_checkpoint_byte_freeze_v1',
        'checked_at_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'task_id': task_id, 'seed': seed, 'gpu_family': family,
        'source_task_id': source_id, 'source_sha256': source_sha,
        'membership_sha256': MEMBERSHIP_SHA, 'partition_sha256': PARTITION_SHA,
        'config_sha256': CONFIG_SHA, 'selected': best,
        'artifacts': {key: {'path': str(path), 'sha256': sha(path),
                            'bytes': path.stat().st_size} for key, path in paths.items()},
        'training_receipt_verified': True, 'publication_receipt_verified': True,
        'all_60_epochs_verified': True, 'checkpoint_bytes_frozen': True,
        'checkpoint_tensor_envelope_verified': False,
        'raw_head_forward_verified': False,
        'paper_eligible': False, 'formal_independent_test_eligible': False,
        'official_test_used': False}
    path = output / 'diagnostic-byte-freeze-receipt.json'
    path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + '\n')
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed', type=int, choices=SOURCE_BY_SEED, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(freeze(args.seed, args.output), sort_keys=True))


if __name__ == '__main__':
    main()
