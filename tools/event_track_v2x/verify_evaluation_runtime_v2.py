#!/usr/bin/env python3
"""Mandatory real-engine self-test of the sealed car-only SPD metric adapter.

No real GT/prediction files or network access. Refuses mismatched runtime trees,
disabled Python assertions and reused output directories. Optional upstream
files verify the ONLY accepted changes: np.float/int builtin-alias replacements.
This is metric conformance evidence, never a real dataset result.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
from importlib import metadata
import inspect
import json
import math
from pathlib import Path
import platform
import re
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tools.event_track_v2x.evaluate_source_ablation_v2 import (
    ADAPTER_SHA256, load_adapter, sha, validate_runtime, write_json,
)


UPSTREAM_COMMIT = '12c8791b303e0a0b50f753af204249e622d0281a'
REFERENCE = {
    'metrics/hota.py': ('c582255c3d36bdb49c3ceeb2a5bd912cee009190dfcfe465b4173d03aaa97d6a',
        'bf85a9e4f2270fb3a52f94b05382250f4b8804a39a897ea9dfbf97ff85bf331e', 5),
    'metrics/identity.py': ('e76b1bdcbf5b193662431596a984cf70a5be32e903eb5f2a3d9921b4762f93ba',
        '33b4ef7e11a24c47c6060eb2464cb92167aea42e8c0e6f652b10ca317872028c', 3),
    'metrics/_base_metric.py': ('dac8b87bf3801269aa280b1cf91184f483de4f4e5c158bd07a4b1733b1a5722b',
        'dac8b87bf3801269aa280b1cf91184f483de4f4e5c158bd07a4b1733b1a5722b', 0),
    '_timing.py': ('62e60638efc675bccc35c7da65158f0f19ce31f332121efe7cc748bd92ad2b30',
        '62e60638efc675bccc35c7da65158f0f19ce31f332121efe7cc748bd92ad2b30', 0),
}


def check_reference_bytes(name, reference, installed):
    expected_reference, expected_installed, replacements = REFERENCE[name]
    normalized, count = re.subn(rb'\bnp\.(float|int)\b', lambda m: m[1], reference)
    if (hashlib.sha256(reference).hexdigest() != expected_reference
            or hashlib.sha256(installed).hexdigest() != expected_installed
            or count != replacements or normalized != installed):
        raise ValueError('official metric reference differs beyond declared scalar aliases: '+name)
    return dict(path=name, upstream_sha256=expected_reference, installed_sha256=expected_installed,
                scalar_alias_replacements=count, byte_identical=reference == installed,
                declared_compatibility_transform_identical=True)


def reference_evidence(reference_root=None):
    import trackeval
    root = Path(inspect.getfile(trackeval)).parent
    records = []
    for name, (official, installed_sha, replacements) in REFERENCE.items():
        installed = (root/name).read_bytes()
        if hashlib.sha256(installed).hexdigest() != installed_sha:
            raise ValueError('installed metric core fingerprint differs: '+name)
        if reference_root is not None:
            path = Path(reference_root)/name
            if path.is_symlink() or not path.is_file():
                raise ValueError('regular official reference module required')
            record = check_reference_bytes(name, path.read_bytes(), installed)
        else:
            record = dict(path=name, upstream_sha256=official, installed_sha256=installed_sha,
                scalar_alias_replacements=replacements, reference_bytes_rechecked=False)
        records.append(record)
    return dict(repository='https://github.com/JonathonLuiten/TrackEval', commit=UPSTREAM_COMMIT,
        reference_bytes_rechecked=reference_root is not None, core_files=records,
        numpy_alias_reference='https://numpy.org/doc/1.20/release/1.20.0-notes.html',
        all_upstream_files_byte_identical=False, all_package_modules_audited_against_upstream=False)


def close(label, actual, expected):
    if (isinstance(actual, bool) or not isinstance(actual, (int, float)) or not math.isfinite(actual)
            or not math.isclose(actual, expected, rel_tol=1e-10, abs_tol=1e-10)):
        raise ValueError(f'{label}: expected {expected!r}, observed {actual!r}')


def sequence(scene, frames, *, switched=False, duplicates=False, shift_x=0., yaw=0.):
    gt, predictions = [], []
    for i in range(frames):
        mean = [float(i), 0., 0., 4., 2., 1.5, 0., 0., 0.]
        stamp = 1_000_000+i*150_100
        gt.append(dict(sequence_id=scene, frame_id=str(i), box_reference_timestamp_us=stamp,
            ego_translation_world=[0., 0., 0.], objects=[dict(track_id='gt', class_label='car', mean=mean)]))
        box = dict(track_id='pred2' if switched and i >= frames//2 else 'pred1', class_label='car',
                   mean=[mean[0]+shift_x, *mean[1:6], yaw, *mean[7:]], score=.9)
        boxes = [box]
        if duplicates:
            boxes.append(dict(copy.deepcopy(box), track_id='duplicate'))
        predictions.append(dict(predictions=boxes))
    return gt, predictions


def extended_cases(adapter):
    """Independent analytic expectations, not values copied from engine output."""
    cases = {}
    ga, pa = sequence('two-perfect', 2)
    gb, pb = sequence('four-switched', 4, switched=True)
    cases['count_weighted_sequence_aggregation'] = (ga+gb, pa+pb,
        dict(HOTA=math.sqrt(2/3), AssA=2/3, DetA=1., IDF1=2/3, IDTP=4., IDFP=2., IDFN=2.),
        dict(ids=1.))
    g, p = sequence('duplicates', 4, duplicates=True)
    cases['duplicate_predictions'] = (g, p,
        dict(HOTA=math.sqrt(.5), AssA=1., DetA=.5, IDF1=2/3, IDTP=4., IDFP=4., IDFN=0.), {})
    g, p = sequence('one-metre', 4, shift_x=1.)
    # A 4x2 box shifted 1m in x has IoU 3/5: 12 of 19 HOTA alphas match.
    cases['distance_vs_iou'] = (g, p, dict(HOTA=12/19, IDF1=1.), dict(amota=1., amotp=1.))
    g, p = sequence('rotated', 4, yaw=math.pi/2)
    # 90-degree rotation has IoU 1/3: 6 HOTA alphas; IDF1 uses IoU>=.5.
    cases['orientation_vs_centre_distance'] = (g, p, dict(HOTA=6/19, IDF1=0.), dict(amota=1., amotp=0.))
    result = {}
    for name, (gt, predictions, expected, expected_nu) in cases.items():
        metrics = adapter.compute_metrics(gt, predictions, ('car',))
        observed = metrics['trackeval']['car']['summary']
        for key, value in expected.items():
            close(name+'/'+key, observed[key], value)
        for key, value in expected_nu.items():
            close(name+'/nuscenes/'+key, metrics['nuscenes'][key], value)
        result[name] = dict(passed=True, frames=len(gt), expected_trackeval=expected,
            expected_nuscenes=expected_nu, observed_trackeval=observed, observed_nuscenes=metrics['nuscenes'])
    return dict(passed=True, class_scope=['car'], synthetic_metric_conformance_only=True, cases=result)


def run(output, *, reference_root=None):
    if not __debug__:
        raise ValueError('Python optimization disables sealed golden assertions; refusing verification')
    if not (3, 9) <= sys.version_info[:2] < (3, 12):
        raise ValueError('sealed evaluation packages require a separate Python 3.9-3.11 environment')
    output = Path(output).absolute()
    if output.exists() or any(p.is_symlink() for p in (output, *output.parents)):
        raise ValueError('new output directory without symlink traversal required')
    sources = {str(p.relative_to(ROOT)): sha(p) for p in
        (Path(__file__), ROOT/'tools/event_track_v2x/evaluate_source_ablation_v2.py')}
    output.mkdir()
    write_json(output/'plan.json', dict(kind='sealed_spd_evaluator_selftest_plan_v1', source_sha256=sources,
        adapter_sha256=ADAPTER_SHA256, class_scope=['car'], real_dataset_read=False,
        real_tracking_method_result=False, reference_commit=UPSTREAM_COMMIT))
    try:
        adapter = load_adapter()
        runtime = adapter.runtime_evidence()
        validate_runtime(runtime)
        reference = reference_evidence(reference_root)
        golden = adapter.golden_cases()
        if golden.get('passed') is not True:
            raise ValueError('sealed evaluator golden cases did not pass')
        extended = extended_cases(adapter)
        after = adapter.runtime_evidence()
        validate_runtime(after)
        if after != runtime or any(sha(ROOT/p) != h for p, h in sources.items()):
            raise ValueError('evaluation runtime or verifier changed during conformance checks')
        packages = sorted([dict(name=d.metadata['Name'], version=d.version) for d in metadata.distributions()],
                          key=lambda r: r['name'].lower())
        write_json(output/'runtime.json', dict(runtime, python=platform.python_version(),
            platform=platform.platform(), machine=platform.machine(), executable=sys.executable,
            installed_distributions=packages, trackeval_upstream_reference=reference))
        write_json(output/'golden-cases.json', golden)
        write_json(output/'extended-cases.json', extended)
        receipt = dict(kind='sealed_spd_evaluator_selftest_receipt_v1', status='complete',
            sealed_source_and_version_match=True, golden_cases_passed=True, extended_cases_passed=True,
            original_cases=3, additional_cases=len(extended['cases']), real_dataset_read=False,
            test_payloads_read=False, real_tracking_method_result=False, paper_eligible=False,
            files={name: sha(output/name) for name in ('plan.json', 'runtime.json', 'golden-cases.json', 'extended-cases.json')})
        write_json(output/'receipt.json', receipt)
        return receipt
    except BaseException as error:
        write_json(output/'failure.json', dict(status='failed', error_type=type(error).__name__, error=str(error),
            partial_outputs_not_conformance_results=True))
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--reference-root', type=Path)
    args = parser.parse_args()
    print(json.dumps(run(args.output, reference_root=args.reference_root), sort_keys=True))


if __name__ == '__main__':
    main()
