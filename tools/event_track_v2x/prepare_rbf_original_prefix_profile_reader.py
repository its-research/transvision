"""Freeze a distinct prefix-only reader and its software refusal evidence."""
import argparse
import ast
import datetime
import json
from pathlib import Path

from rbf_nested_seen_val_v2_common import R, new, register, sha


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--qualification', type=Path, required=True)
    args = parser.parse_args()
    workspace = Path(__file__).resolve().parent
    qualifier = json.loads(args.qualification.read_bytes())
    assert qualifier['kind'] == 'rbf_original_prefix_profile_reader_software_refusal_qualification_v1'
    assert qualifier['reader_sha256'] == sha(workspace/'read_rbf_original_prefix_function_profile.py')
    assert qualifier['qualifier_sha256'] == sha(workspace/'qualify_rbf_original_prefix_profile_reader.py')
    assert qualifier['software_refusal_count'] == len(qualifier['software_refusals']) == 30
    assert qualifier['original_replay_body_executions'] == qualifier['actual_clearml_queries'] == qualifier['actual_GPU_profiles'] == 0
    root = R/'source-freezes/rbf-original-real-prefix-function-profile-independent-reader-v1-20261004'
    assert not root.exists(), 'source freeze must never be overwritten'
    cpu = R/'source-freezes/rbf-final-refit-full-forest-independent-CPU-v3-20261004'
    cpu_freeze = json.loads((cpu/'source-freeze.json').read_bytes())
    references = {}

    def ref(path, expected=None):
        digest = sha(path)
        assert expected is None or expected == digest
        references[str(path)] = digest

    ref(cpu/'source-freeze.json','4cd2f58b77bfc458ff54ef1dd477888b3a627d651804fa157a806c02cb7555f7')
    for name, digest in cpu_freeze['sources'].items():
        ref(cpu/name,digest)
    for record in cpu_freeze['unchanged_independent_sources']:
        ref(Path(record['path']),record['sha256'])
    executor = R/'source-freezes/rbf-original-real-prefix-GPU-function-profile-executor-v1-20261004'
    ref(executor/'preparation.json','b19682dfa7a37077e91460bccb24bd1639d39a1229b2341e39a393a14fc24472')
    prep = json.loads((executor/'preparation.json').read_bytes())
    for name, record in prep['sources'].items():
        ref(executor/name,record['sha256'])
    observer = R/'source-freezes/rbf-original-frozen-replay-external-function-profile-v1-20261004'
    ref(observer/'source-freeze.json','9faa1edd566417ec5ac50cf1b00e1b827c9d6fd5172e75af6e98061bca33a7c0')
    ref(observer/'rbf_frozen_replay_cost_profile.py',prep['plan']['profile_observer_sha256'])
    safe_root = R/'source-freezes/rbf-final-refit-main-topK-independent-output-reader-v1-20261004'
    ref(safe_root/'source-freeze.json','df0188b8751932bf0db7d1180a838187d13add48305e213942617c6447575b87')
    safe_helper = safe_root/'read_rbf_final_refit_forest_outputs.py'
    ref(safe_helper)
    ref(R/'receipts/rbf-final-refit-three-seed-all-row-full-independent-numeric-acceptance-20261004.json',
        prep['plan']['numeric_reference_admission_sha256'])
    root.mkdir()
    sources = {}
    for name in ('read_rbf_original_prefix_function_profile.py','qualify_rbf_original_prefix_profile_reader.py',
                 'prepare_rbf_original_prefix_profile_reader.py','rbf_nested_seen_val_v2_common.py'):
        data = (workspace/name).read_bytes()
        ast.parse(data);compile(data,str(root/name),'exec')
        with (root/name).open('xb') as stream:
            stream.write(data)
        sources[name] = dict(sha256=sha(root/name),bytes=len(data))
    assert sources['rbf_nested_seen_val_v2_common.py']['sha256'] == cpu_freeze['sources']['rbf_nested_seen_val_v2_common.py']
    new(root/'software-qualification.json',qualifier)
    sources['software-qualification.json'] = dict(sha256=sha(root/'software-qualification.json'),
        bytes=(root/'software-qualification.json').stat().st_size)
    new(root/'source-freeze.json',dict(kind='rbf_original_real_prefix_function_profile_independent_reader_source_v1',
        checked_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),sources=sources,
        references=[dict(path=path,sha256=digest) for path,digest in sorted(references.items())],
        safe_byte_helper=dict(path=str(safe_helper),sha256=references[str(safe_helper)]),
        fixed_real_prefix_events_per_rank=32,original_schedule_events=7445,original_schedule_sequences=46,
        all_class_competition_protocol='rbf-all-class-top64-v1',seed=2027,NN_atol=1e-4,NN_rtol=1e-4,
        independent_state_cache203_atol=1e-8,independent_state_cache203_rtol=1e-8,
        original_full_sequence_gate_unchanged=True,
        source_bound_prefix_reader_prepared=True,actual_profile_or_task_output_read=False,
        full_sequence_stage2_latency_and_paper_acceptance=False))
    receipt = R/'receipts/rbf-original-real-prefix-function-profile-independent-reader-readiness-20261004.json'
    new(receipt,dict(kind='rbf_original_prefix_profile_reader_readiness_v1',source_freeze=str(root/'source-freeze.json'),
        source_freeze_sha256=sha(root/'source-freeze.json'),reader_sha256=sources['read_rbf_original_prefix_function_profile.py']['sha256'],
        software_qualification_sha256=sources['software-qualification.json']['sha256'],software_refusals=30,
        original_replay_body_executions=0,actual_GPU_profiles=0,uploaded_or_dispatched=False,
        independent_real_prefix_artifact_readback_pending=True,full_forest_or_paper_accepted=False))
    register(receipt,'rbf-original-prefix-function-profile-independent-reader-readiness')
    print(json.dumps(dict(readiness_receipt=str(receipt),source_freeze_sha256=sha(root/'source-freeze.json'),
        actual_GPU_profile=False,full_forest_or_paper_accepted=False)))


if __name__ == '__main__':
    main()
