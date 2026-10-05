"""Create-once local source candidate; does not upload or dispatch experiments."""
import argparse
import datetime
import json
from pathlib import Path
import shutil

from rbf_nested_seen_val_v2_common import R, new, register, sha

NAME = 'rbf-recovery-off-bound-candidate-v1-20261005'


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--software-log', type=Path, required=True)
    p.add_argument('--configuration', type=Path, required=True)
    args = p.parse_args()
    log = args.software_log.read_text()
    assert '38 passed' in log and not any(x in log for x in ('FAILED', 'ERROR', 'skipped'))
    config = json.loads(args.configuration.read_bytes())
    assert config['backend'] == 'exclusive_event_boundary_recovery_off_v1'
    assert config['method'] == 'rbf' and config['allocation'] == 'bound'
    assert config['state']['candidate_protocol'] == 'rbf-all-class-top64-v1'
    source = Path(__file__).resolve().parents[2]
    out = R/'source-freezes'/NAME
    assert not out.exists(), 'preserve previous freeze and failures'
    out.mkdir()
    files = set()
    for group in ('transvision/models/event_track_v2x', 'tools/event_track_v2x', 'tests/event_track_v2x'):
        files.update((source/group).glob('*.py'))
    for name in ('transvision/__init__.py', 'transvision/register.py', 'transvision/version.py',
                 'transvision/models/__init__.py', 'tools/__init__.py', 'tests/__init__.py'):
        if (source/name).exists(): files.add(source/name)
    manifest = {}
    for src in sorted(files):
        assert src.is_file() and not src.is_symlink()
        rel = src.relative_to(source)
        target = out/rel; target.parent.mkdir(parents=True, exist_ok=True)
        before = sha(src); shutil.copyfile(src, target)
        assert sha(target) == before == sha(src)
        manifest[str(rel)] = dict(bytes=target.stat().st_size, sha256=before)
    shutil.copyfile(args.software_log, out/'workspace-software-tests.log')
    shutil.copyfile(args.configuration, out/'bound-configuration.json')
    for name in ('workspace-software-tests.log','bound-configuration.json'):
        target=out/name; manifest[name]=dict(bytes=target.stat().st_size,sha256=sha(target))
    value = dict(kind='rbf_recovery_off_bound_source_candidate_v1',
        checked_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), sources=manifest,
        software_tests=38, scope='current_workspace_bound_allocation_candidate',
        declared_intervention='next-event support restricted to previous active union output classes',
        allocation='bound', source_selection='RBF source tools and tests; catalog files do not add experiment requirements',
        original_remote_final_model_source_admission=False, learned_priority_binding_implemented=False,
        full_real_data_forest_independent_acceptance=False, uploaded=False, GPU_experiment_dispatched=False,
        paper_performance_complete=False, immutable_test_log_sha256=sha(out/'workspace-software-tests.log'))
    new(out/'source-freeze.json',value);register(out/'source-freeze.json',value['kind'])
    print(json.dumps(dict(source_freeze=str(out/'source-freeze.json'),sha256=sha(out/'source-freeze.json'),
                          sources=len(manifest),workspace_software_tests=38)))


if __name__ == '__main__': main()
