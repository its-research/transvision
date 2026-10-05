"""Freeze the seen-val fixed baseline search and selector interfaces, create once."""
import argparse
import ast
import datetime
import json
from pathlib import Path
import re
import shutil

import accept_seen_val_fixed_search_selector as driver
from rbf_nested_seen_val_v2_common import R, new, register, sha

NAME='rbf-seen-val-fixed-K1-K4-search-selector-v1-20261005'


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--software-log',type=Path,required=True)
    args=parser.parse_args();log=args.software_log.read_text()
    matches=re.findall(r'(\d+) passed',log)
    assert matches and int(matches[-1])>=60 and not any(s in log for s in ('FAILED','ERROR'))
    output=R/'source-freezes'/NAME;assert not output.exists()
    own=Path(__file__).resolve().parent;references={}
    roots=[(driver.CPU,driver.CPU_SHA)]
    roots.extend((R/'source-freezes'/name,digest) for name,digest in driver.ORIGINALS.values())
    for root,digest in roots:
        freeze=driver.frozen(root,digest);references[str(root/'source-freeze.json')]=digest
        references.update({str(root/name):item['sha256'] for name,item in freeze['sources'].items()})
        for key in ('references','unchanged_references'):
            references.update({item['path']:item['sha256'] for item in freeze.get(key,[])})
        if 'qualification' in freeze:
            item=freeze['qualification'];references[item['path']]=item['sha256']
    for width in (1,4):
        for stage in ('search','selector'):driver.oracle_gate(width,stage)
    for path,digest in references.items():assert sha(path)==digest
    names=('accept_seen_val_fixed_search_selector.py','prepare_seen_val_fixed_search_selector.py','rbf_nested_seen_val_v2_common.py')
    for name in names:ast.parse((own/name).read_bytes())
    output.mkdir()
    for name in names:shutil.copyfile(own/name,output/name)
    test=Path('tests/event_track_v2x/test_seen_val_fixed_search_selector.py')
    target=output/test;target.parent.mkdir(parents=True);shutil.copyfile(own.parents[1]/test,target)
    shutil.copyfile(args.software_log,output/'software-tests.log')
    shutil.copyfile(own.parents[1]/'docs/recover-before-fuse/remaining-experiment-code-preparation-20261004.md',output/'remaining-experiments.md')
    value=dict(kind='rbf_seen_val_fixed_search_selector_source_v1',checked_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        sources={str(p.relative_to(output)):dict(sha256=sha(p),bytes=p.stat().st_size) for p in sorted(output.rglob('*')) if p.is_file()},
        references=[dict(path=p,sha256=h) for p,h in sorted(references.items())],software_tests=int(matches[-1]),
        software_scope='scope/lineage rejection, unchanged oracle sources, file-based mocked numerical orchestration; no val forest experiment',
        original_oracles_unchanged=True,selector_does_not_rerun_search=True,required_sequences=21,required_events=3316,
        atol=1e-8,rtol=1e-8,selector_decimal_precision=70,actual_seen_val_forest_accepted=False,
        full_Stage2_complete=False,complete_online_method_accepted=False,same_resource_performance_accepted=False,paper_performance_complete=False)
    new(output/'source-freeze.json',value);register(output/'source-freeze.json',value['kind'])
    print(json.dumps(dict(source_freeze=str(output/'source-freeze.json'),sha256=sha(output/'source-freeze.json'),software_tests=int(matches[-1]))),flush=True)


if __name__=='__main__':main()
