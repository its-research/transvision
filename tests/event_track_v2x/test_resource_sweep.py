"""Measured fresh child processes, shared trained factors and explicit failures."""
import json
import os
import subprocess

import pytest

from transvision.models.event_track_v2x import resource_sweep as resource
from transvision.models.event_track_v2x.detection_cache_v2 import canonical, sha_file
from transvision.models.event_track_v2x.forest_training_data import frozen_cache_identity
from transvision.models.event_track_v2x.persistent_component_tracking import PersistentComponentConfig
from transvision.models.event_track_v2x.completion_component_tracking import PersistentCompletionConfig
from transvision.models.event_track_v2x.covered_completion_tracking import PersistentCoveredCompletionConfig
from transvision.models.event_track_v2x.allocation_training import (
    allocation_sources, export_training, fit_priority, training_binding,
)
from transvision.models.event_track_v2x.forest_training_checkpoint import load_identity_checkpoint
from tools.event_track_v2x.run_persistent_forest_v2 import replay_rows
from tools.event_track_v2x.train_forest_identity import FitConfig, fit_dataset
from test_run_persistent_forest_v2 import replay_inputs
from test_detection_cache_v2 import sources
from test_forest_training_data import prepared_rows


def spec(cache, rows, directory, backends=('component', 'joint_beam'), checkpoint=None):
    schedule = directory/'resource-schedule.json'
    schedule.write_bytes(canonical(rows))
    return dict(kind=resource.KIND, cache=dict(path=str(cache.root), sha256=cache.manifest_sha256),
        schedule=dict(path=str(schedule), sha256=sha_file(schedule)),
        jobs=[dict(id=name, backend=name, configuration={}, checkpoint=checkpoint) for name in backends],
        repetitions=2, order_seed=1337, timeout_seconds=60.)


def test_all_registered_backends_in_new_processes_share_actual_trained_factors(prepared_rows, tmp_path):
    data, _, cache, rows = prepared_rows
    trained = tmp_path/'trained'
    fit = fit_dataset(data, sha_file(data/'manifest.json'), trained,
        config=FitConfig(epochs=1, batch_size=4, hidden=8, heads=2, dropout=0.), require_full_train=False)
    cp = fit['seeds'][0]
    directory = (trained/cp['checkpoint_manifest']).parent
    priorities = {}
    for backend, config in (('learned_component', PersistentComponentConfig()),
                            ('learned_completion', PersistentCompletionConfig()),
                            ('learned_covered_completion', PersistentCoveredCompletionConfig()),
                            ('learned_beam_recovery', resource.BeamRecoveryConfig())):
        scorer, _ = load_identity_checkpoint(directory, cp['checkpoint_sha256'], config=config.state)
        teacher, exported, priority = (tmp_path/(backend+'-'+name) for name in ('teacher', 'exported', 'priority'))
        binding = training_binding(config, scorer.signature, frozen_cache_identity(cache))
        replay_rows(cache, rows, teacher, config, learned_scorer=scorer, allocation_teacher=True,
            plan=dict(source_sha256=allocation_sources(), allocation_training_binding=binding, full_official_train_verified=False))
        export_training(teacher, sha_file(teacher/'receipt.json'), exported)
        fitted_priority = fit_priority(exported, sha_file(exported/'manifest.json'), priority,
            epochs=1, hidden=4, require_full_train=False)
        priorities[backend] = dict(path=str(priority/'1337'),
            sha256=fitted_priority['seeds'][0]['checkpoint_sha256'])
    value = spec(cache, rows, tmp_path, backends=tuple(resource.KINDS),
                 checkpoint=dict(path=str(directory), sha256=cp['checkpoint_sha256']))
    for job in value['jobs']:
        if job['backend'] in resource.LEARNED_COMPONENT_KINDS:
            job['allocation_checkpoint'] = priorities[job['backend']]
    output = tmp_path/'sweep'
    result = resource.run_sweep(value, output, allow_fixture=True)
    assert result['status'] == 'complete' and len(result['runs']) == 2*len(resource.KINDS)
    assert result['cohort_mode'] == 'fixture-only' and not result['paper_eligible']
    assert result['raw_factors_equal_within_seed'] and result['repeated_predictions_identical']
    assert len({r['factor_stream_sha256'] for r in result['runs']}) == 1
    for run in result['runs']:
        assert run['runtime']['pid'] != os.getpid()
        assert run['runtime']['torch_threads'] == run['runtime']['torch_interop_threads'] == 1
        assert run['child_wall_seconds'] >= run['frame_latency_seconds_p50_p95_p99_max'][-1] >= 0
        assert run['process_peak_rss_bytes'] > 0
        assert run['retained_artifact_bytes'] > run['database_bytes'] > 0
    assert result['separate_fresh_process_per_run']
    plan = json.loads((output/'plan.json').read_bytes())
    for block in range(2):
        assert {r['job_index'] for r in plan['execution_order'] if r['repetition'] == block} == set(range(len(resource.KINDS)))
    wrong_priority = json.loads(json.dumps(value))
    next(job for job in wrong_priority['jobs'] if job['backend'] == 'learned_completion')['allocation_checkpoint'] = priorities['learned_component']
    with pytest.raises(ValueError, match='tracking binding differs'):
        resource.validate_spec(wrong_priority, allow_fixture=True)
    wrong_coverage = json.loads(json.dumps(value))
    next(job for job in wrong_coverage['jobs'] if job['backend'] == 'learned_covered_completion')['allocation_checkpoint'] = priorities['learned_completion']
    with pytest.raises(ValueError, match='tracking binding differs'):
        resource.validate_spec(wrong_coverage, allow_fixture=True)
    first = result['runs'][0]
    timings = output/first['output']/'frame-timings.jsonl'
    original = timings.read_bytes()
    with timings.open('ab') as stream:
        stream.write(b'{}\n')
    with pytest.raises(ValueError, match='artifact changed'):
        resource.inspect_run(output/first['output'], plan, plan['jobs'][first['job_index']])
    timings.write_bytes(original)
    assert resource.inspect_run(output/first['output'], plan, plan['jobs'][first['job_index']])['factor_stream_sha256'] == first['factor_stream_sha256']
    formal = dict(value, repetitions=3)
    with pytest.raises(ValueError, match='full-train three-seed'):
        resource.run_sweep(formal, tmp_path/'cannot-relabel-fixture')
    assert not (tmp_path/'cannot-relabel-fixture').exists()


def test_formal_entry_refuses_geometry_before_creating_outputs(replay_inputs, tmp_path):
    cache, rows = replay_inputs
    value = spec(cache, rows, tmp_path)
    value['repetitions'] = 3
    with pytest.raises(ValueError, match='full-train checkpoints'):
        resource.run_sweep(value, tmp_path/'not-created')
    assert not (tmp_path/'not-created').exists()


@pytest.mark.parametrize('change', [dict(repetitions=True), dict(timeout_seconds=float('nan')),
                                    dict(order_seed=True), dict(unexpected=True)])
def test_bad_spec_fails_before_output(replay_inputs, tmp_path, change):
    cache, rows = replay_inputs
    value = dict(spec(cache, rows, tmp_path), **change)
    with pytest.raises(ValueError):
        resource.run_sweep(value, tmp_path/'not-created', allow_fixture=True)
    assert not (tmp_path/'not-created').exists()


def test_timeout_preserves_plan_and_failure_not_success(replay_inputs, tmp_path, monkeypatch):
    cache, rows = replay_inputs
    value = spec(cache, rows, tmp_path)
    def timeout(*args, **kwargs):
        raise subprocess.TimeoutExpired(args[0], kwargs['timeout'])
    monkeypatch.setattr(resource.subprocess, 'run', timeout)
    with pytest.raises(subprocess.TimeoutExpired):
        resource.run_sweep(value, tmp_path/'timed-out', allow_fixture=True)
    failure = json.loads((tmp_path/'timed-out/failure.json').read_bytes())
    assert failure['error_type'] == 'TimeoutExpired' and failure['completed_runs'] == 0
    assert not (tmp_path/'timed-out/receipt.json').exists()


def test_actual_child_failure_is_not_a_completed_comparison(replay_inputs, tmp_path):
    cache, rows = replay_inputs
    value = spec(cache, rows, tmp_path, backends=('monolithic',))
    value['jobs'][0]['configuration'] = dict(state=dict(max_replay_operations=1))
    with pytest.raises(subprocess.CalledProcessError):
        resource.run_sweep(value, tmp_path/'failed', allow_fixture=True)
    assert (tmp_path/'failed/run-0000/failure.json').is_file()
    assert (tmp_path/'failed/failure.json').is_file()
    assert not (tmp_path/'failed/receipt.json').exists()


def test_job_alias_cannot_change_declared_gaussian_updater():
    with pytest.raises(ValueError, match='update rule'):
        resource.configuration(dict(backend='pkf', configuration=dict(update_rule='jpda-ci')))


@pytest.mark.parametrize('backend,completions', [
    ('beam_recovery', 1), ('beam_recovery', 0), ('beam_recovery_disabled', 1),
])
def test_recovery_receipt_flags_cannot_mislabel_execution(backend, completions):
    enabled = backend == 'beam_recovery'
    receipt = dict(frontier_completion_enabled=enabled and completions > 0,
        coverage_aware_proposal_admission=enabled and completions > 0,
        beam_recovery_backbone_enabled=True, additional_beam_recovery_enabled=enabled,
        irreversible_beam_enabled=not enabled)
    job = dict(backend=backend, configuration=dict(recovery_completions_per_component=completions))
    resource.validate_execution_modes(receipt, job)
    for name in receipt:
        with pytest.raises(ValueError, match=name):
            resource.validate_execution_modes(dict(receipt, **{name: not receipt[name]}), job)
        with pytest.raises(ValueError, match=name):
            resource.validate_execution_modes({k:v for k,v in receipt.items() if k != name}, job)


def test_duplicate_family_and_seed_is_rejected(replay_inputs, tmp_path):
    cache, rows = replay_inputs
    value = spec(cache, rows, tmp_path, backends=('component',))
    value['jobs'].append(dict(value['jobs'][0], id='same-config-other-label'))
    with pytest.raises(ValueError, match='duplicate configuration'):
        resource.validate_spec(value, allow_fixture=True)


def test_cross_backend_input_mismatch_preserves_first_completed_run(replay_inputs, tmp_path, monkeypatch):
    cache, rows = replay_inputs
    value = spec(cache, rows, tmp_path)
    original, calls = resource.inspect_run, []
    def differing(*args):
        result = original(*args)
        calls.append(result)
        if len(calls) == 2:
            result['factor_stream_sha256'] = '0'*64
        return result
    monkeypatch.setattr(resource, 'inspect_run', differing)
    output = tmp_path/'mismatch'
    with pytest.raises(ValueError, match='different raw observations or factors'):
        resource.run_sweep(value, output, allow_fixture=True)
    assert json.loads((output/'failure.json').read_bytes())['completed_runs'] == 1
    assert (output/'run-0000-verified.json').exists()
    assert not (output/'receipt.json').exists()
