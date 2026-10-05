"""Create-once source package for optional GPU resource scan measurement."""
import argparse
import datetime
from pathlib import Path
import re
import shutil

from rbf_nested_seen_val_v2_common import R, new, register, sha
import scan_paper_resources as scan

NAME = 'rbf-paper-GPU-resource-scan-measurement-v2-20261005'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--software-log', type=Path, required=True)
    parser.add_argument('--prior-log', type=Path, action='append', default=[])
    args = parser.parse_args()
    text = args.software_log.read_text()
    assert re.search(r'\b47 passed\b', text) and not any(s in text for s in ('FAILED', 'ERROR', 'skipped'))
    sources = set(scan.source_files(True)) | {Path(scan.__file__).resolve(), Path(__file__).resolve()}
    for relative in (
        'transvision/__init__.py', 'transvision/register.py', 'transvision/version.py',
        'transvision/models/__init__.py', 'tools/__init__.py', 'tools/event_track_v2x/__init__.py',
        'tools/event_track_v2x/rbf_nested_seen_val_v2_common.py',
        'tools/event_track_v2x/build_detection_cache_v2.py',
        'tests/event_track_v2x/test_paper_gpu_resource_scan.py', 'tests/event_track_v2x/test_paper_resource_scan.py',
        'tests/event_track_v2x/test_paper_pipeline.py', 'tests/event_track_v2x/test_detection_cache_v2.py',
        'docs/recover-before-fuse/remaining-experiment-code-preparation-20261004.md'):
        path = scan.ROOT/relative
        if path.exists(): sources.add(path)
    inventory = {str(p.relative_to(scan.ROOT)): dict(bytes=p.stat().st_size, sha256=sha(p)) for p in sorted(sources)}
    out = R/'source-freezes'/NAME
    assert not out.exists()
    out.mkdir()
    for relative, record in inventory.items():
        target = out/relative; target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(scan.ROOT/relative, target)
        assert sha(target) == record['sha256'] and sha(scan.ROOT/relative) == record['sha256']
    logs = {}
    for index, path in enumerate([*args.prior_log, args.software_log]):
        target = out/f'software-attempt-{index+1}.log'; shutil.copyfile(path, target)
        logs[target.name] = dict(bytes=target.stat().st_size, sha256=sha(target), original=str(path))
    value = dict(kind=NAME, checked_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        sources=inventory, logs=logs, software_tests=47,
        software_scope='CPU synthetic replays with mocked CUDA counters, independent evaluator subprocess, CPU fresh-process regression',
        original_frozen_experiment_sources_modified=False, experiment_tasks_created=0, uploads_performed=False,
        supersedes_incomplete_test_dependency_package=dict(
            path=str(R/'source-freezes/rbf-paper-GPU-resource-scan-measurement-v1-20261005/source-freeze.json'),
            sha256=sha(R/'source-freezes/rbf-paper-GPU-resource-scan-measurement-v1-20261005/source-freeze.json')),
        actual_GPU_measurement_accepted=False, memory_75_80_percent_accepted=False,
        full_dataset_accepted=False, complete_forest_numeric_accepted=False, equal_resources_claimed=False,
        paper_results_verified=False,
        remaining=['admit actual frozen model/source runtime numerically', 'bind predeclared resource budgets and isolated GPU',
                   'execute complete train resource scan and independently verify costs', 'complete same-resource baselines and evaluation'])
    new(out/'source-freeze.json', value); register(out/'source-freeze.json', NAME)
    print(dict(source_freeze=str(out/'source-freeze.json'), sha256=sha(out/'source-freeze.json'), source_files=len(inventory), software_tests=47))


if __name__ == '__main__': main()
