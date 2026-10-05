"""Freeze an ETA-enabled priority source from the published v3 bytes.

This changes logging only. It does not modify the v3 source or dispatch work.
"""
import argparse
import ast
import gzip
import hashlib
import io
import json
from pathlib import Path
import tarfile


TRAINER = 'tools/event_track_v2x/train_allocation_policy_ddp.py'
PROGRESS = 'transvision/models/event_track_v2x/experiment_progress.py'
OLD = {
    'return dict(ddp_sources(),**allocation_sources(),\n'
    '                **{Path(__file__).relative_to(ROOT).as_posix():sha_file(__file__)})':
    'return dict(ddp_sources(),**allocation_sources(),\n'
    "                **{\'transvision/models/event_track_v2x/experiment_progress.py\':sha_file(ROOT/\'transvision/models/event_track_v2x/experiment_progress.py\'),\n"
    '                   Path(__file__).relative_to(ROOT).as_posix():sha_file(__file__)})',
    '    plan_sha=_rank_zero(create_output);started=time.monotonic();directory=output/str(seed)':
    '    from transvision.models.event_track_v2x.experiment_progress import ExperimentProgress\n'
    '    plan_sha=_rank_zero(create_output);started=time.monotonic();directory=output/str(seed)\n'
    '    eta = ExperimentProgress("priority_training_epochs", config.epochs) if rank == 0 else None',
    "                print(json.dumps(entry,sort_keys=True),flush=True)\n"
    '            _rank_zero(save_epoch)':
    "                print(json.dumps(entry,sort_keys=True),flush=True)\n"
    '                eta.update(epoch, force=True)\n'
    '            _rank_zero(save_epoch)',
}


def digest(data):
    return hashlib.sha256(data).hexdigest()


def freeze(source, helper, output):
    source, helper, output = Path(source), Path(helper), Path(output)
    if output.exists() or output.is_symlink():
        raise ValueError('new destination required')
    original_sha = digest(source.read_bytes())
    if original_sha != 'df5ae65ad69ba1c9e211270500e379107c62ba70f9c074b854a659dee3e72ae0':
        raise ValueError('priority v3 source differs')
    members = {}
    with tarfile.open(source, 'r:gz') as archive:
        for item in archive:
            if not item.isfile() or item.name in members:
                raise ValueError('unexpected source member')
            members[item.name] = archive.extractfile(item).read()
    if PROGRESS in members:
        raise ValueError('progress helper already present')
    text = members[TRAINER].decode()
    for old, new in OLD.items():
        if text.count(old) != 1:
            raise ValueError('trainer anchor differs')
        text = text.replace(old, new)
    members[TRAINER] = text.encode()
    members[PROGRESS] = helper.read_bytes()
    for name, data in members.items():
        if name.endswith('.py'):
            ast.parse(data, filename=name)
    output.mkdir(parents=True)
    blob = io.BytesIO()
    with tarfile.open(fileobj=blob, mode='w') as archive:
        for name in sorted(members):
            data = members[name]
            item = tarfile.TarInfo(name)
            item.size, item.mode, item.mtime = len(data), 0o644, 0
            archive.addfile(item, io.BytesIO(data))
    target = output / 'priority-authorized-gpu-source.tar.gz'
    target.write_bytes(gzip.compress(blob.getvalue(), mtime=0))
    with tarfile.open(target, 'r:gz') as archive:
        assert {item.name for item in archive} == set(members)
        for name, data in members.items():
            assert archive.extractfile(name).read() == data
    receipt = {
        'kind': 'priority_authorized_gpu_eta_source_freeze_v4',
        'base_v3_sha256': original_sha,
        'archive_sha256': digest(target.read_bytes()),
        'changed_members': [TRAINER, PROGRESS],
        'inventory': [{'path': name, 'bytes': len(data), 'sha256': digest(data)}
                      for name, data in sorted(members.items())],
        'trainer_log_stage': 'priority_training_epochs',
        'eta_after_each_completed_epoch': True,
        'eta_is_acceptance_evidence': False,
        'teacher_data_included': False,
        'source_published': False,
        'training_complete': False,
    }
    (output / 'source-freeze-receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--helper', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    receipt = freeze(args.source, args.helper, args.output)
    print('SOURCE_FROZEN', len(receipt['inventory']), receipt['archive_sha256'])


if __name__ == '__main__':
    main()
