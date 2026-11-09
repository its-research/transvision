#!/usr/bin/env python3
"""Prepare exact already-qualified public author assets on the CPU host."""
import hashlib
import json
import os
import shutil
import stat
import subprocess
import time
import zipfile
from pathlib import Path, PurePosixPath

import requests


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()


def new(path, value):
    path.open('x').write(json.dumps(value, sort_keys=True, indent=2) + '\n')


def prepare(plan, source_code):
    root = Path(plan['output'])
    assert root.is_dir() and not root.is_symlink()
    assert sha(root/'failure.json') == plan['prior_failure_sha256']
    assert sha(root/'plan.json') == plan['prior_plan_sha256']
    assert sha(root/'preparation-source.py') == plan['prior_source_sha256']
    assert sha(Path(plan['python'])) == plan['python_executable_sha256']
    needed = sum(a['bytes'] for a in plan['assets']) + sum(a['bytes'] for a in plan['val_files'].values())
    needed += 2 * plan['author_source_bytes'] + plan['minimum_free_reserve_bytes']
    needed -= sum(a['bytes'] for a in plan['assets'])
    assert shutil.disk_usage(root.parent).free > needed, 'Insufficient admitted storage for exact native inputs'
    (root / 'continuation-source-v2.py').open('x').write(source_code)
    new(root / 'continuation-plan-v2.json', plan)
    try:
        repository = root / 'author-source/DMSTrack'
        def git(*args):
            return subprocess.check_output(['git', '-C', str(repository), *args])
        assert git('rev-parse', 'HEAD').decode().strip() == plan['commit']
        assert not git('diff', 'HEAD', '--')
        source_files = {}
        for row in git('ls-tree', '-r', '-z', plan['commit']).split(b'\0'):
            if not row:
                continue
            metadata, encoded_name = row.split(b'\t', 1)
            mode, kind, oid = metadata.decode().split()
            name = encoded_name.decode()
            path = repository / name
            assert kind == 'blob' and mode in ('100644', '100755', '120000')
            if mode == '120000':
                assert path.is_symlink()
                raw = os.fsencode(os.readlink(path))
            else:
                assert path.is_file() and not path.is_symlink()
                raw = path.read_bytes()
            assert hashlib.sha1(b'blob ' + str(len(raw)).encode() + b'\0' + raw).hexdigest() == oid
            source_files[name] = dict(bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest(), mode=mode)
        assert len(source_files) == plan['author_source_files']
        for name, value in plan['critical_author_sources'].items():
            assert source_files[name]['sha256'] == value
        new(root / 'author-source-readback-v2.json', dict(commit=plan['commit'], files=source_files,
            author_symlink_targets_dereferenced=False, tracked_source_unchanged=True))
        assets = {}
        for spec in plan['assets']:
            path = root / spec['name']
            assert path.is_file() and not path.is_symlink()
            assert path.stat().st_size == spec['bytes'] and sha(path) == spec['sha256']
            transfer = json.loads((root / (spec['name'] + '.readback.json')).read_bytes())
            assert transfer['bytes'] == spec['bytes'] and transfer['sha256'] == spec['sha256']
            assert transfer['transport'] == 'SSH exact already-qualified public author bytes'
            assets[spec['name']] = dict(path=str(path), bytes=spec['bytes'], sha256=spec['sha256'])
        archive = root / 'no_fusion_keep_all.zip'
        extraction = root / 'native-feature-bytes'
        assert shutil.disk_usage(root).free > sum(a['bytes'] for a in plan['val_files'].values()) + plan['minimum_free_reserve_bytes']
        extraction.mkdir()
        expected = plan['val_files']
        extracted = {}
        seen = set()
        with zipfile.ZipFile(archive) as zf:
            for info in zf.infolist():
                name = info.filename
                pure = PurePosixPath(name)
                assert not pure.is_absolute() and '..' not in pure.parts and '\\' not in name
                assert pure.parts[0] == 'no_fusion_keep_all' and name not in seen
                assert not stat.S_ISLNK(info.external_attr >> 16) and not info.flag_bits & 1
                seen.add(name)
                target = extraction / pure
                if info.is_dir():
                    target.mkdir(parents=True, exist_ok=True)
                    continue
                assert name in expected and info.file_size == expected[name]['bytes']
                target.parent.mkdir(parents=True, exist_ok=True)
                h = hashlib.sha256()
                count = 0
                with zf.open(info) as member, target.open('xb') as stream:
                    for block in iter(lambda: member.read(8 * 1024**2), b''):
                        stream.write(block)
                        h.update(block)
                        count += len(block)
                assert count == expected[name]['bytes'] and h.hexdigest() == expected[name]['sha256']
                extracted[name] = dict(bytes=count, sha256=h.hexdigest())
                if len(extracted) % 2000 == 0:
                    print(json.dumps(dict(stage='exact_qualified_native_ZIP_member_readback',
                                          members=len(extracted))), flush=True)
        assert set(extracted) == set(expected)
        # Only an untracked native data path is linked; tracked author files stay exact.
        native_path = repository / 'V2V4Real/official_models/no_fusion_keep_all/npy'
        assert not native_path.exists() and not native_path.is_symlink()
        native_path.symlink_to(extraction / 'no_fusion_keep_all/npy', target_is_directory=True)
        assert not git('diff', 'HEAD', '--')
        ego, remote = root/'model_ego_epoch_3.pth', root/'model_1_epoch_3.pth'
        assert Path(str(ego).replace('ego', '1')) == remote and not ego.samefile(remote)
        result = dict(kind='rbf_DMSTrack_exact_qualified_native_CPU_input_preparation',
                      source=str(repository), author_commit=plan['commit'], source_files=len(source_files),
                      assets=assets, native_feature_members=len(extracted), native_feature_files=extracted,
                      qualified_Mac_val_feature_receipt_sha256=plan['qualified_val_receipt_sha256'],
                      native_feature_shape_audit_reused_from_exact_qualified_ZIP=True,
                      native_consumed_data_path=str(native_path), native_real_data_path=str(native_path.resolve()),
                      data_path_link_tracked_by_Git=False, author_tracked_source_changed=False,
                      existing_checkouts_or_environments_modified=False,
                      prior_HTTP_connection_failure_preserved=True,
                      asset_transport="SSH exact already-qualified public author bytes",
                      neural_detection_or_embedding_forward_repeated=False,
                      native_predictions_produced=False, native_method_accepted=False,
                      RBF_priority_or_CUDA_branch_state_source_transported=False,
                      preparation_source_sha256=hashlib.sha256(source_code.encode()).hexdigest())
        new(root / 'input-preparation-receipt.json', result)
        print(json.dumps(dict(receipt=str(root/'input-preparation-receipt.json'),
                              complete_native_CPU_inputs_prepared=True, native_method_accepted=False)), flush=True)
    except BaseException as error:
        new(root / 'continuation-failure-v2.json', dict(error_type=type(error).__name__, error=str(error),
            partial_assets_preserved=True, automatic_retry=False, native_method_accepted=False))
        raise


if 'PLAN' in globals():
    prepare(PLAN, SOURCE_CODE)
