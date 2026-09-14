#!/usr/bin/env python3
"""Run byte-exact migrated research tests from a transvision compatibility snapshot.

The original archive and paper repository are read-only. A create-once local
snapshot overlays archived research code with explicitly inventoried paper
contracts and four retained build dependencies. Nothing is restored to thesis;
no task submission, registry write or network publication is performed.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import subprocess
import sys
import tempfile

REPOSITORY = Path(__file__).resolve().parents[2]
if str(REPOSITORY) not in sys.path:
    sys.path.insert(0, str(REPOSITORY))
from tools.event_track_v2x.migrate_thesis_code import BUILD_DEPENDENCIES, sha, write_once

ARCHIVE = REPOSITORY / 'legacy/thesis-research-20260912'
GROUPS = (
    'tests', 'experiments/clearml/tests', 'experiments/rtp_v2x/tests',
    'experiments/adapters/centerpoint_v2xseq/tests',
    'experiments/adapters/transvision_spd/tests',
    'experiments/adapters/v2xgraph_tfd/tests',
)
CANARY = 'evidence/clearml/synth-causal-canary/b824e7114467492aa71f87c0dad0d555'


def regular_file(root, relative):
    relative = PurePosixPath(relative)
    if relative.is_absolute() or '..' in relative.parts:
        raise ValueError('snapshot path must be relative and contained')
    path = root / relative
    if path.is_symlink() or path.resolve() != path.absolute() or not path.is_file():
        raise ValueError(f'snapshot input must be a physical regular file: {path}')
    return path


def prepare(thesis, work_root):
    manifest_path = regular_file(ARCHIVE, 'migration-manifest.json')
    manifest = json.loads(manifest_path.read_text())
    copies = json.loads(regular_file(ARCHIVE, 'copies-verified.json').read_text())
    if copies['manifest_sha256'] != sha(manifest_path):
        raise ValueError('archive manifest changed')
    sources = {}
    for row in manifest['files']:
        path = regular_file(ARCHIVE, row['path'])
        if sha(path) != row['sha256'] or path.stat().st_size != row['bytes']:
            raise ValueError(f'archive bytes changed: {path}')
        sources[row['path']] = (path, 'migrated_research')
    if len(sources) != manifest['file_count']:
        raise ValueError('archive file count mismatch')
    support = {name for name in BUILD_DEPENDENCIES if '/tests/' not in name}
    support.update(path.relative_to(thesis).as_posix() for path in
                   (thesis / 'experiments/clearml').rglob('*.json'))
    support.update(f'{CANARY}/{name}' for name in
                   ('metrics.json', 'events.jsonl', 'run_manifest.json', 'receipt.json'))
    for name in sorted(support):
        if name in sources:
            raise ValueError('support cannot overwrite migrated research')
        sources[name] = (regular_file(thesis, name), 'retained_support_copy')
    # Restrict generated code to this canonical repository, never to thesis.
    if (work_root.resolve() != work_root.absolute() or work_root == REPOSITORY
            or REPOSITORY not in work_root.parents or work_root == ARCHIVE
            or ARCHIVE in work_root.parents):
        raise ValueError('work root must be a physical transvision subdirectory outside the archive')
    work_root.mkdir(parents=True, exist_ok=True)
    output = Path(tempfile.mkdtemp(prefix='run-', dir=work_root))
    context = output / 'context'
    rows = []
    for name, (source, role) in sorted(sources.items()):
        target = context / name
        target.parent.mkdir(parents=True, exist_ok=True)
        before = sha(source)
        with source.open('rb') as src, target.open('xb') as dst:
            shutil.copyfileobj(src, dst)
        os.chmod(target, source.stat().st_mode & 0o777)
        if sha(target) != before or sha(source) != before:
            raise ValueError(f'snapshot input changed during copy: {name}')
        rows.append({'path': name, 'role': role, 'sha256': before,
                     'bytes': target.stat().st_size, 'source': str(source)})
    write_once(output / 'inputs.json', {'kind': 'migrated_research_test_context_v1',
               'archive_manifest_sha256': sha(manifest_path), 'files': rows,
               'original_sources_modified': False})
    return output, context


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--thesis-root', type=Path, required=True)
    parser.add_argument('--work-root', type=Path, default=REPOSITORY / 'work_dirs/legacy-thesis-tests')
    parser.add_argument('--suite', choices=('all', 'tfd'), default='all')
    args = parser.parse_args()
    output, context = prepare(args.thesis_root.resolve(), args.work_root.absolute())
    environment = {**os.environ, 'PYTHONDONTWRITEBYTECODE': '1',
                   'OPENBLAS_NUM_THREADS': '1', 'OMP_NUM_THREADS': '1',
                   'PYTHONPATH': str(context)}
    groups = GROUPS if args.suite == 'all' else ('experiments/rtp_v2x/tests',)
    pattern = 'test_*.py' if args.suite == 'all' else 'test_v2xseq_tfd_canary.py'
    results = []
    for index, group in enumerate(groups):
        command = [sys.executable, '-m', 'unittest', 'discover', '-s', group, '-p', pattern, '-v']
        print(f'Running migrated research tests: {group}', flush=True)
        process = subprocess.run(command, cwd=context, env=environment,
                                 stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        with (output / f'suite-{index}.log').open('x') as stream:
            stream.write(process.stdout)
        match = re.search(r'^Ran (\d+) tests? in', process.stdout, re.MULTILINE)
        count = int(match.group(1)) if match else 0
        skipped = re.search(r'^OK \(skipped=(\d+)\)', process.stdout, re.MULTILINE)
        skip_count = int(skipped.group(1)) if skipped else 0
        passed = process.returncode == 0 and count > 0
        results.append({'suite': group, 'command': command, 'returncode': process.returncode,
                        'test_count': count, 'skipped': skip_count, 'passed': passed,
                        'log': f'suite-{index}.log', 'log_sha256': sha(output / f'suite-{index}.log')})
        print(f'{group}: {count} tests, {skip_count} skipped, passed={passed}', flush=True)
        if not passed:
            print(process.stdout[-6000:], flush=True)
    report = {'kind': 'migrated_research_test_receipt_v1', 'suites': results,
              'test_count': sum(row['test_count'] for row in results),
              'skipped': sum(row['skipped'] for row in results),
              'passed': all(row['passed'] for row in results),
              'inputs_sha256': sha(output / 'inputs.json'),
              'real_dataset_evaluation': False, 'clearml_publication': False}
    write_once(output / 'results.json', report)
    print(json.dumps({'passed': report['passed'], 'tests': report['test_count'],
                      'skipped': report['skipped'], 'receipt': str(output / 'results.json')}))
    return 0 if report['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
