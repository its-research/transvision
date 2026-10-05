"""Freeze one-shot Top1 seed3407 reader-to-original-CPU continuation."""
import argparse
import ast
import datetime
import json
from pathlib import Path
import shutil

import continue_final_top1_seed3407 as driver
from rbf_nested_seen_val_v2_common import R, new, register, sha

NAME = 'rbf-final-top1-seed3407-after-readback-continuation-v1-20261005'
ORIGINAL = R/'source-freezes/rbf-final-top1-seed1337-after-readback-continuation-v1-20261005/continue_final_top1_seed1337.py'


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--software-log', type=Path, required=True)
    args = parser.parse_args(); log = args.software_log.read_text()
    assert '23 passed' in log and not any(x in log for x in ('FAILED', 'ERROR', 'skipped'))
    assert sha(ORIGINAL) == '0d3d2f2e45646784ddb6684d76f28ba57d5a11adb760adae234e92527bd0cb96'
    assert sha(driver.DRIVER) == driver.DRIVER_SHA and sha(driver.DRIVER.parent/'source-freeze.json') == driver.FREEZE_SHA
    assert sha(driver.READER) == driver.READER_SHA
    own = Path(__file__).resolve().parent
    before = ast.parse(ORIGINAL.read_bytes()); after = ast.parse((own/'continue_final_top1_seed3407.py').read_bytes())
    def observer(tree): return ast.dump(next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'observe'))
    assert observer(before) == observer(after)
    reverted = (own/'continue_final_top1_seed3407.py').read_text().replace('3407', '1337').replace('5d960f26c30d4c66b2d687d6ae9a99ee', '237f30c74248439c91f07a7e868ca383')
    assert ast.dump(before) == ast.dump(ast.parse(reverted)), 'only seed and task bindings may change'
    references = {str(ORIGINAL): sha(ORIGINAL)}
    for root in (driver.DRIVER.parent, driver.READER.parent):
        control = json.loads((root/'source-freeze.json').read_bytes())
        references[str(root/'source-freeze.json')] = sha(root/'source-freeze.json')
        for name, item in control['sources'].items():
            assert sha(root/name) == item['sha256']; references[str(root/name)] = item['sha256']
        for key in ('references', 'unchanged_references'):
            for item in control.get(key, []):
                assert sha(item['path']) == item['sha256']; references[item['path']] = item['sha256']
    out = R/'source-freezes'/NAME; assert not out.exists(); out.mkdir()
    for name in ('continue_final_top1_seed3407.py', 'prepare_final_top1_seed3407_continuation.py', 'rbf_nested_seen_val_v2_common.py'):
        shutil.copyfile(own/name, out/name)
    test = own.parents[1]/'tests/event_track_v2x/test_final_top1_seed3407_continuation.py'
    shutil.copyfile(test, out/test.name); shutil.copyfile(args.software_log, out/'software-tests.log')
    value = dict(kind='rbf_final_top1_seed3407_after_readback_source_v1', checked_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        sources={p.name: dict(bytes=p.stat().st_size, sha256=sha(p)) for p in sorted(out.iterdir())},
        references=[dict(path=p, sha256=h) for p,h in sorted(references.items())], software_tests=23,
        original_observer_AST_unchanged=True, entire_controller_AST_unchanged_except_seed_and_task=True, original_reader_and_CPU_driver_unchanged=True,
        seed=3407, task_id=driver.TASK, observe_PID_start_and_command=True, unknown_is_not_terminal=True,
        byte_admission_required=True, ledger_binding_required=True, live_task_and_final_model_rechecked=True,
        automatic_retry=False, experiment_accepted=False, paper_performance_complete=False)
    new(out/'source-freeze.json', value); register(out/'source-freeze.json', value['kind'])
    print(json.dumps(dict(source_freeze=str(out/'source-freeze.json'), sha256=sha(out/'source-freeze.json'), software_tests=23)))


if __name__ == '__main__': main()
