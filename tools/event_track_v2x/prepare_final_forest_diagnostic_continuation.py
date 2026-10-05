"""Create-once local recovery controller freeze; no experiments are dispatched."""
import argparse
import ast
import datetime
import json
from pathlib import Path
import re
import shutil

import continue_final_forest_seed2027_after_diagnostic as controller
from rbf_nested_seen_val_v2_common import R,new,register,sha


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--software-log',type=Path,required=True)
    args=parser.parse_args();log=args.software_log.read_text();counts=re.findall(r'(\d+) passed',log)
    assert counts and int(counts[-1])>=49 and 'FAILED' not in log and 'ERROR' not in log
    output=R/'source-freezes'/controller.NAME;assert not output.exists()
    base=controller.base_module();original_binding=base.source_gate()
    assert sha(controller.OLD/'source-freeze.json')==controller.OLD_SHA
    assert sha(controller.DIAG_SOURCE/'source-freeze.json')==controller.DIAG_SHA
    references={}
    for root in (controller.OLD,controller.DIAG_SOURCE):
        path=root/'source-freeze.json';references[str(path)]=sha(path);v=json.loads(path.read_bytes())
        for name,item in v['sources'].items():references[str(root/name)]=item['sha256'] if isinstance(item,dict) else item
        for item in v.get('references',[]):references[item['path']]=item['sha256']
    for path in (base.READER,base.DRIVER,base.DRIVER.parent/'source-freeze.json'):
        references[str(path)]=sha(path)
    for path,digest in references.items():assert sha(path)==digest
    own=Path(__file__).resolve().parent
    files=('continue_final_forest_seed2027_after_diagnostic.py','prepare_final_forest_diagnostic_continuation.py','rbf_nested_seen_val_v2_common.py')
    for name in files:ast.parse((own/name).read_bytes())
    output.mkdir()
    for name in files:shutil.copyfile(own/name,output/name)
    test=Path('tests/event_track_v2x/test_final_forest_diagnostic_continuation.py')
    destination=output/test;destination.parent.mkdir(parents=True);shutil.copyfile(own.parents[1]/test,destination)
    shutil.copyfile(args.software_log,output/'software-tests.log')
    shutil.copyfile(own.parents[1]/'docs/recover-before-fuse/remaining-experiment-code-preparation-20261004.md',output/'remaining-experiments.md')
    value=dict(kind=controller.NAME,checked_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        sources={str(p.relative_to(output)):dict(sha256=sha(p),bytes=p.stat().st_size) for p in sorted(output.rglob('*')) if p.is_file()},
        references=[dict(path=p,sha256=h) for p,h in sorted(references.items())],software_tests=int(counts[-1]),
        unchanged_original_chain=original_binding,seed=2027,task_id=controller.TASK,
        original_corrupt_partial_and_failures_preserved=True,six_prior_good_archives_not_redownloaded=True,
        original_terminal_reader_and_CPU_driver_unchanged=True,automatic_retry=False,
        no_uploads_or_remote_task_mutations=True,experiment_accepted=False,paper_performance_complete=False)
    new(output/'source-freeze.json',value);register(output/'source-freeze.json',value['kind'])
    print(json.dumps(dict(source_freeze=str(output/'source-freeze.json'),sha256=sha(output/'source-freeze.json'),software_tests=int(counts[-1]))),flush=True)


if __name__=='__main__':main()
