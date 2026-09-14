#!/usr/bin/env python3
"""Move explicitly inventoried thesis code, retaining byte-exact recovery copies.

Only research files are moved. Build tools, their shared validation dependency
closure, paper text, typesetting, data, metadata and Git history stay in place.
No network, Git mutation or directory deletion is performed. Source files are
removed only after ALL copies verify.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import stat
import subprocess


EXTENSIONS = {
    '.py', '.pyc', '.pyo', '.sh', '.bash', '.zsh', '.js', '.mjs', '.cjs',
    '.ts', '.tsx', '.jsx', '.ipynb', '.c', '.cc', '.cpp', '.h', '.hpp',
    '.rs', '.go', '.java', '.class', '.r', '.jl', '.sql', '.lua', '.luc',
    '.pl', '.rb', '.ps1', '.bat', '.cmd', '.so', '.dylib', '.o', '.a',
}
NAMES = {'Makefile', 'GNUmakefile', 'Dockerfile', 'justfile', 'latexmkrc', '.latexmkrc'}
BUILD_DEPENDENCIES = {
    'experiments/rtp_v2x/full_release_identity_contract.py',
    'experiments/rtp_v2x/baseline_reproduction_contract.py',
    'experiments/rtp_v2x/provenance_seals.py',
    'experiments/clearml/plan_required_runs.py',
    'experiments/rtp_v2x/tests/test_full_release_identity_contract.py',
    'experiments/rtp_v2x/tests/test_baseline_reproduction_contract.py',
    'experiments/rtp_v2x/tests/test_provenance_seals.py',
    'experiments/clearml/tests/test_plan_required_runs.py',
}
RESEARCH_ROOT_TESTS = {
    'tests/test_checked_canary_evidence.py', 'tests/test_publish_task.py',
    'tests/test_download_canary_evidence.py',
}


def is_research(relative):
    name = relative.as_posix()
    if name in BUILD_DEPENDENCIES:
        return False
    # Deliberately narrow after the safety review: no evidence artifact,
    # bytecode, build input or generated cache is removed by this tool.
    return relative.suffix in {'.py', '.sh'} and (
        name.startswith('experiments/') or name in RESEARCH_ROOT_TESTS)


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def is_code(path):
    return (path.suffix.lower() in EXTENSIONS or path.name in NAMES
            or path.name.endswith(('.lua.gz', '.luc.gz'))
            or ('.github/workflows' in path.as_posix() and path.suffix in {'.yml', '.yaml'}))


def inventory(source):
    result = []
    for directory, subdirs, files in os.walk(source, followlinks=False):
        subdirs[:] = sorted(name for name in subdirs if name != '.git')
        for name in sorted(files):
            path = Path(directory) / name
            if not is_code(path) or not is_research(path.relative_to(source)):
                continue
            relative = path.relative_to(source)
            if path.is_symlink() or any(parent.is_symlink() for parent in path.parents if parent != source.parent):
                raise ValueError(f'symlink in code path: {relative}')
            info = path.stat()
            if not stat.S_ISREG(info.st_mode):
                raise ValueError(f'not a regular file: {relative}')
            result.append({'path': relative.as_posix(), 'bytes': info.st_size,
                           'sha256': sha(path), 'mode': stat.S_IMODE(info.st_mode)})
    return sorted(result, key=lambda item: item['path'])


def write_once(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as stream:
        json.dump(value, stream, ensure_ascii=False, sort_keys=True, indent=2)
        stream.write('\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--archive', type=Path, required=True)
    parser.add_argument('--receipt', type=Path, required=True)
    parser.add_argument('--apply', action='store_true')
    args = parser.parse_args()
    source, archive, receipt = (path.absolute() for path in (args.source, args.archive, args.receipt))
    if (source != source.resolve() or not (source / '.git').is_dir()
            or source in archive.parents or archive in source.parents
            or source == archive or not source.is_dir()):
        parser.error('distinct physical source Git repository and external archive required')
    if archive.exists() or archive.is_symlink() or receipt.exists():
        parser.error('archive and receipt are create-once; existing paths are never overwritten')
    rows = inventory(source)
    manifest = {'kind': 'thesis_code_migration_v1', 'source_root': str(source),
                'archive_root': str(archive), 'files': rows, 'file_count': len(rows),
                'total_bytes': sum(row['bytes'] for row in rows),
                'source_git_head': subprocess.check_output(
                    ['git', '-C', str(source), 'rev-parse', 'HEAD'], text=True).strip(),
                'history_rewritten': False, 'git_commit_or_push': False,
                'latex_typesetting_sources_preserved': True,
                'scope': 'experiments_source_and_three_research_tests_only',
                'evidence_and_all_bytecode_preserved': True,
                'retained_build_dependencies': sorted(BUILD_DEPENDENCIES)}
    if not args.apply:
        print(json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2))
        return
    if not rows:
        parser.error('no source code to migrate')
    archive.mkdir(parents=True, exist_ok=False)
    write_once(archive / 'migration-manifest.json', manifest)
    for row in rows:
        origin, target = source / row['path'], archive / row['path']
        target.parent.mkdir(parents=True, exist_ok=True)
        with origin.open('rb') as src, target.open('xb') as dst:
            shutil.copyfileobj(src, dst)
        os.chmod(target, row['mode'])
    for row in rows:
        if sha(source / row['path']) != row['sha256'] or sha(archive / row['path']) != row['sha256']:
            raise ValueError('source/copy mismatch; no source files removed')
    if inventory(source) != rows:
        raise ValueError('source inventory changed; no source files removed')
    write_once(archive / 'copies-verified.json', {'status': 'all_copies_verified',
               'manifest_sha256': sha(archive / 'migration-manifest.json')})
    for row in rows:
        origin = source / row['path']
        if sha(origin) != row['sha256']:
            raise ValueError('source changed during removal; archive is complete, stop and reconcile')
        origin.unlink()
    remaining = inventory(source)
    result = {**manifest, 'status': 'migrated' if not remaining else 'remaining_code_detected',
              'remaining_code_files': [row['path'] for row in remaining],
              'archive_manifest_sha256': sha(archive / 'migration-manifest.json'),
              'recoverable': True}
    write_once(receipt, result)
    print(json.dumps({key: result[key] for key in
          ('status', 'file_count', 'total_bytes', 'remaining_code_files', 'recoverable')}, sort_keys=True))


if __name__ == '__main__':
    main()
