"""Software-only refusal and empty-profiler failure checks; no replay runs."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import types

from rbf_frozen_replay_cost_profile import _selected_stats, profile_replay


R = Path('/Volumes/Data/test/recover-before-fuse')


def main():
    root = R/'source-freezes/rbf-Task-monitor-and-frozen-call-path-audit-v1-20261004/effective-main-source-snapshots'
    source = root/'exclusive_paper_runtime.py'
    raw = source.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    assert digest == 'a0512672665a726e059fe8d6b0cfce4505ec40b381184267b664a6f9110b463c'
    module = compile(raw, str(source), 'exec', dont_inherit=True)
    code = next(c for c in module.co_consts if isinstance(c, types.CodeType) and c.co_name == 'replay')
    # No imports or original replay body execute. Matching bytecode allows the
    # observer's pre-execution gates to be tested without a GPU or Torch.
    replay = types.FunctionType(code, {})
    schedule_path = R/'artifacts/rbf-original-joint-coupled-forest-GPU4-v1-source-20261001/independent-cloud-readback/events-seed2027.bytes'
    schedule_sha = hashlib.sha256(schedule_path.read_bytes()).hexdigest()
    assert schedule_sha == '3440542d0fb6b7c52a6b97a886a888ab5acca73ab1b73a37139e1c49b37c6c25'
    schedule = json.loads(schedule_path.read_bytes())
    sequence = sorted(schedule['origin_us_by_sequence'])[0]
    events = [event for event in schedule['events'] if event['sequence_id'] == sequence][:2]
    checkpoint_path = R/'artifacts/rbf-all-class-final-refit-independent-byte-freeze-v1-20261004/seed2027/checkpoint'
    ck = json.loads(checkpoint_path.read_bytes())
    model = dict(candidate_protocol='rbf-all-class-top64-v1', dataset='spd', fit_split='train',
        checkpoint_sha256=hashlib.sha256(checkpoint_path.read_bytes()).hexdigest(),
        model_sha256=ck['model_sha256'], seed=2027, frozen_cache_identity=ck['frozen_cache_identity'])
    runtime = dict(events_from_independently_admitted_real_schedule=True, causal_event_bytes_unchanged=True,
        read_only_model_loaded_from_frozen_checkpoint=True, TF32_matmul=False, TF32_cudnn=False,
        actual_device='cpu:software-refusal-fixture')
    base = dict(source_root=root, source_manifest={source.name: digest}, cache=None,
        events=events, protocol=None, configuration={}, scorer=None, model_binding=model,
        runtime_identity=runtime, schedule_path=schedule_path, schedule_sha256=schedule_sha,
        selected_sequences=[sequence], per_sequence_prefix_count=2)
    directory = Path(tempfile.mkdtemp(prefix='rbf-profile-software-refusals-', dir='/private/tmp'))
    checks = []

    def refusal(name, update, expected_message, function=replay):
        kwargs = copy.deepcopy(base)
        update(kwargs)
        output = directory/name
        try:
            profile_replay(function, output=output, **kwargs)
        except ValueError as error:
            assert expected_message in str(error), (name, str(error))
            assert not output.exists(), 'pre-execution refusal created a measurement directory'
            checks.append(name)
        else:
            raise AssertionError('invalid contract accepted: '+name)

    refusal('source-hash', lambda x: x['source_manifest'].update({source.name: '0'*64}), 'frozen source differs')
    wrong_code = compile('def replay(*args, **kwargs):\n return {}\n', str(source), 'exec', dont_inherit=True)
    wrong = types.FunctionType(next(c for c in wrong_code.co_consts if isinstance(c, types.CodeType)), {})
    refusal('replaced-replay-code', lambda x: None, 'bytecode differs', function=wrong)
    refusal('future-event-mutation', lambda x: x['events'][0].update(decision_us=x['events'][0]['decision_us']+1), 'unchanged complete per-sequence causal prefixes')
    refusal('omitted-event', lambda x: x.update(events=x['events'][:1]), 'unchanged complete per-sequence causal prefixes')
    refusal('reordered-events', lambda x: x['events'].reverse(), 'unchanged complete per-sequence causal prefixes')
    refusal('invented-empty-event', lambda x: x['events'].append(dict(x['events'][0], event_id='software-qualification-invented', deliveries=[])), 'unchanged complete per-sequence causal prefixes')
    refusal('validation-input', lambda x: x['model_binding'].update(fit_split='val'), 'restricted to all-class SPD train')
    refusal('early-class-filter', lambda x: x['model_binding'].update(candidate_protocol='rbf-car-first-top64-v1'), 'restricted to all-class SPD train')
    refusal('TF32', lambda x: x['runtime_identity'].update(TF32_matmul=True), 'real schedule binding')
    refusal('runtime-extra-fields', lambda x: x['runtime_identity'].update(unsafe_runtime_field='software-fixture'), 'unallowlisted runtime')
    refusal('schedule-hash', lambda x: x.update(schedule_sha256='0'*64), 'real schedule bytes differ')
    refusal('zero-prefix', lambda x: x.update(per_sequence_prefix_count=0), 'fixed causal prefix count')
    refusal('unknown-sequence', lambda x: x.update(selected_sequences=['not-in-real-schedule']), 'fixed causal prefix count')
    refusal('missing-GPU-wait', lambda x: x['runtime_identity'].update(actual_device='cuda:software-fixture'), 'explicit device synchronization')
    refusal('unfrozen-model', lambda x: x['runtime_identity'].update(read_only_model_loaded_from_frozen_checkpoint=False), 'frozen checkpoint load binding')

    class QualificationDeviceWaitFailure(RuntimeError):
        pass

    def fail_before_replay():
        raise QualificationDeviceWaitFailure('software-only forced pre-replay wait failure')

    kwargs = copy.deepcopy(base)
    kwargs['runtime_identity']['actual_device'] = 'cuda:software-fixture'
    output = directory/'empty-profiler-failure-transparency'
    try:
        profile_replay(replay, output=output, synchronize=fail_before_replay, **kwargs)
    except QualificationDeviceWaitFailure:
        receipt = json.loads((output/'measurement-receipt.json').read_bytes())
        functions = json.loads((output/'function-profile.json').read_bytes())['functions']
        assert functions == [] and receipt['status'] == 'failed'
        assert receipt['profiled_replay_wall_seconds'] is None
        assert receipt['original_replay_receipt'] is None
        assert receipt['failure']['error_type'] == 'QualificationDeviceWaitFailure'
        checks.append('empty-profiler-failure-transparency')
    else:
        raise AssertionError('pre-replay wait failure was swallowed')
    result = dict(kind='rbf_external_profile_software_refusals_and_failure_transparency_v1',
        passed=True, checks=checks, check_count=len(checks), original_replay_body_executions=0,
        real_schedule_sha256=schedule_sha, original_frozen_replay_source_sha256=digest,
        observer_sha256=hashlib.sha256(Path(__file__).with_name('rbf_frozen_replay_cost_profile.py').read_bytes()).hexdigest(),
        software_fixture_output=str(directory), any_actual_GPU_or_Torch_execution=False,
        profiling_producer_or_independent_profile_admission=False,
        actual_replay_numerical_or_performance_admission=False)
    path = directory/'qualification.json'
    path.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(dict(result_path=str(path), **result)), flush=True)


if __name__ == '__main__':
    main()
