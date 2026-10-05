"""Freeze the tested, distinct seen-val output reader without running it."""
import argparse
import ast
import datetime
import json
from pathlib import Path
import shutil

from rbf_nested_seen_val_v2_common import R,new,register,sha
from rbf_seen_val_forest_output_binding import DISPATCH,DISPATCH_SHA,ORIGINAL_READER,ORIGINAL_READER_SHA,INDEX,INDEX_SHA

NAME='rbf-seen-val-bound-forest-output-reader-v1-20261005'


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--software-log',type=Path,required=True)
    args=parser.parse_args()
    out=R/'source-freezes'/NAME;assert not out.exists()
    log=args.software_log.read_text()
    assert '37 passed' in log and 'FAILED' not in log and 'ERROR' not in log
    assert sha(DISPATCH/'source-freeze.json')==DISPATCH_SHA
    prior=json.loads((DISPATCH/'source-freeze.json').read_bytes())
    assert sha(ORIGINAL_READER)==ORIGINAL_READER_SHA and sha(INDEX)==INDEX_SHA
    own=Path(__file__).resolve().parent
    dependencies=('publish_rbf_seen_val_forest_inputs.py','submit_rbf_seen_val_bound_forest.py',
        'rbf_nested_seen_val_v2_common.py','rbf_final_refit_teacher_binding.py',
        'submit_rbf_final_identity.py','submit_rbf_seen_val_joint_identity.py')
    for name in dependencies:
        assert sha(own/name)==prior['sources'][name]['sha256']
    paths=[own/name for name in (*dependencies,'rbf_seen_val_forest_output_binding.py',
        'read_rbf_seen_val_forest_outputs.py','prepare_seen_val_forest_output_reader.py')]
    paths.append(own.parents[1]/'tests/event_track_v2x/test_seen_val_forest_output_reader.py')
    for p in paths:ast.parse(p.read_text())
    refs={p['path']:p['sha256'] for p in prior['references']}
    refs.update({str(DISPATCH/'source-freeze.json'):DISPATCH_SHA,str(ORIGINAL_READER):ORIGINAL_READER_SHA,str(INDEX):INDEX_SHA})
    for p,h in refs.items():assert sha(p)==h
    out.mkdir()
    for p in paths:shutil.copyfile(p,out/p.name)
    shutil.copyfile(args.software_log,out/'software-tests.log')
    document=own.parents[1]/'docs/recover-before-fuse/remaining-experiment-code-preparation-20261004.md'
    shutil.copyfile(document,out/'remaining-experiments.md')
    result=dict(kind='rbf_seen_val_complete_bound_forest_output_reader_source_v1',
        checked_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        sources={p.name:dict(sha256=sha(p),bytes=p.stat().st_size) for p in sorted(out.iterdir())},
        references=[dict(path=p,sha256=h) for p,h in sorted(refs.items())],
        software_tests=37,software_scope='existing val input interface and synthetic output corruption tests; no real forest output admission',
        original_transfer_and_float64_log_softmax_helpers_unchanged=True,NN_atol=1e-4,NN_rtol=1e-4,
        source_and_input_publication_unchanged=True,full_21_sequence_3316_event_scope_required=True,
        downloaded_prediction_references_not_NN_reexecution=True,extracted_members_bound_to_archive_bytes=True,
        actual_seen_val_forest_task_created=False,actual_seen_val_output_admitted=False,
        continuous_states_and_forest_semantics_admitted=False,full_Stage2_complete=False,paper_performance_complete=False)
    new(out/'source-freeze.json',result);register(out/'source-freeze.json',result['kind'])
    print(json.dumps(dict(source_freeze=str(out/'source-freeze.json'),sha256=sha(out/'source-freeze.json'),software_tests=37)),flush=True)


if __name__=='__main__':main()
