#!/usr/bin/env python3
"""Build a private minimal source + derived TRAIN-row bundle; never uploads."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import tarfile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tools.event_track_v2x.train_forest_identity import audit_dataset
from tools.event_track_v2x.train_forest_identity_ddp import ddp_sources
from transvision.models.event_track_v2x.detection_cache_v2 import contained_file, sha_file
from transvision.models.event_track_v2x.forest_training_data import _new_json


def pack(data, expected_sha, output):
    data, output = Path(data), Path(output).absolute()
    manifest, _, rows, informative = audit_dataset(data, expected_sha)
    if manifest.get('gt_boxes_or_ids_in_shards') is not False:
        raise ValueError('derived rows must exclude raw GT boxes and identity strings')
    if output.exists() or any(p.is_symlink() for p in (output, *output.parents)):
        raise ValueError('new private package output without symlink traversal required')
    sources = ddp_sources()
    output.mkdir()
    paths = [ROOT/'transvision'/name for name in ('__init__.py', 'register.py', 'version.py')]
    paths += [ROOT/'transvision/models/__init__.py']
    paths += sorted((ROOT/'transvision/models/event_track_v2x').glob('*.py'))
    paths += [ROOT/'tools/event_track_v2x'/name for name in ('__init__.py', 'prepare_forest_training.py',
              'train_forest_identity.py', 'train_forest_identity_ddp.py')]
    inventory = []
    with tarfile.open(output/'source.tar.gz', 'x:gz') as archive:
        for path in paths:
            if path.is_symlink() or not path.is_file():
                raise ValueError('only ordinary source files may enter package')
            name = path.relative_to(ROOT).as_posix()
            inventory.append(dict(path=name, bytes=path.stat().st_size, sha256=sha_file(path)))
            archive.add(path, arcname=name, recursive=False)
    with tarfile.open(output/'train-rows.tar.gz', 'x:gz') as archive:
        for name in ['manifest.json']+[s['path'] for s in manifest['shards']]:
            archive.add(contained_file(data, name), arcname=name, recursive=False)
    if sources != ddp_sources() or sha_file(data/'manifest.json') != expected_sha:
        raise ValueError('package inputs changed during construction')
    result = dict(kind='forest_identity_ddp_package_v1', dataset_sha256=expected_sha,
        full_official_train=True, split='train', class_scope=['car'], raw_GT_included=False,
        original_detection_stream_included=False, trained_models_included=False,
        supervised_rows=rows, informative_rows=informative, source_inventory=inventory,
        source_sha256=sources, paper_eligible=False,
        artifacts=[dict(path=name, bytes=(output/name).stat().st_size, sha256=sha_file(output/name))
                   for name in ('source.tar.gz', 'train-rows.tar.gz')])
    _new_json(output/'package.json', result)
    print(json.dumps(dict(package_sha256=sha_file(output/'package.json'),
                         artifacts=result['artifacts'], source_files=len(inventory)), sort_keys=True))
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--manifest-sha256', required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    pack(a.data, a.manifest_sha256, a.output)
