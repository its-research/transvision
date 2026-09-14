"""Fresh-process CPU resource comparisons, not claims of equal compute or wins.

The formal entry requires a complete SPD val schedule, full-train checkpoints
for all three seeds per configuration, and no ground-truth inputs. Every run
uses one fresh process. Repetitions are randomized in predeclared blocks; no
cache flushing, exclusive-host claim, warmup deletion or best-run selection.
"""
from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import random
import resource
import subprocess
import sys
import time

import numpy as np
import scipy

from .detection_cache_v2 import canonical, contained_file, sha_file
from .forest_tracking import ForestTrackingConfig
from .forest_cache_stream import VerifiedForestCache
from .forest_training_checkpoint import load_identity_checkpoint
from .forest_training_data import frozen_cache_identity
from .persistent_forest import PersistentForestConfig
from .persistent_component_tracking import PersistentComponentConfig
from .completion_component_tracking import PersistentCompletionConfig
from .covered_completion_tracking import PersistentCoveredCompletionConfig
from .persistent_beam_tracking import PersistentBeamConfig, PersistentRankedBeamConfig
from .persistent_joint_beam import PersistentJointBeamConfig
from .persistent_class_bound_beam import PersistentClassBoundBeamConfig
from .persistent_slot_bound_beam import PersistentSlotBoundBeamConfig
from .persistent_sparse_slot_bound_beam import PersistentSparseSlotBoundBeamConfig
from .persistent_reachable_slot_bound_beam import PersistentReachableSlotBoundBeamConfig
from .beam_recovery_tracking import BeamRecoveryConfig
from .persistent_probabilistic_tracking import PersistentProbabilisticConfig
from .jpda_marginals import JPDALimits
from .allocation_training import load_priority, training_binding


ROOT = Path(__file__).resolve().parents[3]
KINDS = dict(monolithic=PersistentForestConfig, component=PersistentComponentConfig,
    learned_component=PersistentComponentConfig, node_beam=PersistentBeamConfig,
    component_completion=PersistentCompletionConfig, learned_completion=PersistentCompletionConfig,
    component_covered_completion=PersistentCoveredCompletionConfig,
    learned_covered_completion=PersistentCoveredCompletionConfig,
    batch_beam=PersistentRankedBeamConfig, joint_beam=PersistentJointBeamConfig,
    class_bound_joint_beam=PersistentClassBoundBeamConfig,
    slot_bound_joint_beam=PersistentSlotBoundBeamConfig,
    sparse_slot_bound_joint_beam=PersistentSparseSlotBoundBeamConfig,
    reachable_slot_bound_joint_beam=PersistentReachableSlotBoundBeamConfig,
    beam_recovery=BeamRecoveryConfig, beam_recovery_disabled=BeamRecoveryConfig,
    learned_beam_recovery=BeamRecoveryConfig,
    jpda_ci=PersistentProbabilisticConfig, jpda_kalman=PersistentProbabilisticConfig,
    pkf=PersistentProbabilisticConfig)
UPDATERS = dict(jpda_ci='jpda-ci', jpda_kalman='jpda-kalman', pkf='pkf')
LEARNED_COMPONENT_KINDS = frozenset(('learned_component', 'learned_completion', 'learned_covered_completion',
                                   'learned_beam_recovery'))
COVERED_COMPLETION_KINDS = frozenset(('component_covered_completion', 'learned_covered_completion'))
COMPLETION_KINDS = frozenset(('component_completion', 'learned_completion')) | COVERED_COMPLETION_KINDS
SEEDS = (1337, 2027, 3407)
KIND = 'rbf_cpu_resource_sweep_v1'
THREAD_ENV = dict(OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1',
    VECLIB_MAXIMUM_THREADS='1', NUMEXPR_NUM_THREADS='1', BLIS_NUM_THREADS='1',
    PYTHONHASHSEED='0', PYTHONDONTWRITEBYTECODE='1')


def _write(path, value):
    with path.open('xb') as stream:
        stream.write(canonical(value))


def _digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def _new_directory(path):
    path = Path(path).absolute()
    if path.exists() or any(p.is_symlink() for p in (path, *path.parents)):
        raise ValueError('new nonsymlink output directory required')
    path.mkdir()
    return path


def source_snapshot():
    # Includes package initializers and all local tool modules, including lazy
    # imports; no reliance on a clean Git tree or filenames in the user plan.
    paths = [ROOT/'transvision/__init__.py', ROOT/'transvision/models/__init__.py',
             ROOT/'transvision/register.py', ROOT/'transvision/version.py']
    for directory in (ROOT/'transvision/models/event_track_v2x', ROOT/'tools/event_track_v2x'):
        paths.extend(sorted(directory.glob('*.py')))
    return {p.relative_to(ROOT).as_posix(): sha_file(p) for p in paths}


def configuration(job):
    if job['backend'] not in KINDS or type(job.get('configuration', {})) is not dict:
        raise ValueError('known backend and configuration object required')
    values = dict(job.get('configuration', {}))
    if job['backend'] in ('beam_recovery','beam_recovery_disabled','learned_beam_recovery'):
        enabled=job['backend']!='beam_recovery_disabled'
        if 'enable_recovery' in values and values['enable_recovery'] is not enabled:
            raise ValueError('beam recovery backend and explicit switch differ')
        values['enable_recovery']=enabled
    values['state'] = ForestTrackingConfig(**values.get('state', {}))
    if job['backend'] in UPDATERS:
        if 'update_rule' in values and values['update_rule'] != UPDATERS[job['backend']]:
            raise ValueError('backend and update rule differ')
        values['update_rule'] = UPDATERS[job['backend']]
        values['inference'] = JPDALimits(**values.get('inference', {}))
    return KINDS[job['backend']](**values)


def _reference(value, filename=None):
    if type(value) is not dict or set(value) != {'path', 'sha256'}:
        raise ValueError('path and sha256 reference required')
    path = Path(value['path']).absolute()
    target = contained_file(path, filename) if filename else path
    if any(p.is_symlink() for p in (target, *target.parents)) or sha_file(target) != value['sha256']:
        raise ValueError('input file identity differs or symlink traversal')
    return dict(path=str(path), sha256=value['sha256'])


def validate_spec(spec, *, allow_fixture=False):
    if sys.platform not in ('darwin', 'linux'):
        raise ValueError('resource RSS units are defined only for macOS and Linux')
    fields = {'kind', 'cache', 'schedule', 'jobs', 'repetitions', 'order_seed', 'timeout_seconds'}
    if type(spec) is not dict or set(spec) != fields or spec['kind'] != KIND:
        raise ValueError('exact resource sweep schema required')
    if (type(spec['repetitions']) is not int or not 1 <= spec['repetitions'] <= 100
            or not allow_fixture and spec['repetitions'] < 3
            or type(spec['order_seed']) is not int
            or type(spec['timeout_seconds']) not in (int, float)
            or not math.isfinite(spec['timeout_seconds']) or spec['timeout_seconds'] <= 0):
        raise ValueError('positive bounded repetitions, integer order seed and finite timeout required')
    if type(spec['jobs']) is not list or not spec['jobs'] or len(spec['jobs']) > 256:
        raise ValueError('one to 256 predeclared jobs required')
    cache_ref = _reference(spec['cache'], 'manifest.json')
    schedule_ref = _reference(spec['schedule'])
    jobs, groups, identifiers, inputs = [], {}, set(), {}
    for raw in spec['jobs']:
        if (type(raw) is not dict or not {'id', 'backend', 'configuration', 'checkpoint'} <= set(raw)
                or set(raw)-{'id', 'backend', 'configuration', 'checkpoint', 'allocation_checkpoint'}
                or type(raw['id']) is not str or not raw['id'] or raw['id'] in identifiers):
            raise ValueError('unique job ID and exact known job fields required')
        identifiers.add(raw['id'])
        config = configuration(raw)
        job = dict(raw, configuration=asdict(config))
        if raw['checkpoint'] is None:
            if not allow_fixture:
                raise ValueError('formal resource study requires full-train checkpoints')
            seed, scorer = None, None
        else:
            job['checkpoint'] = _reference(raw['checkpoint'], 'checkpoint.json')
            scorer, meta = load_identity_checkpoint(**dict(root=job['checkpoint']['path'],
                manifest_sha256=job['checkpoint']['sha256']), config=config.state, device='cpu')
            if not allow_fixture and (meta.get('full_official_train') is not True or meta['seed'] not in SEEDS):
                raise ValueError('full-train three-seed checkpoint required; no fixture relabelling')
            seed = meta['seed']
        allocation = raw.get('allocation_checkpoint')
        if (raw['backend'] in LEARNED_COMPONENT_KINDS) != (allocation is not None):
            raise ValueError('learned component backend requires its explicit priority checkpoint only')
        if allocation is not None:
            if scorer is None:
                raise ValueError('resource priority comparison requires the learned identity checkpoint')
            job['allocation_checkpoint'] = _reference(allocation, 'checkpoint.json')
        job['seed'] = seed
        family = _digest([raw['backend'], asdict(config)])
        if seed in groups.setdefault(family, set()):
            raise ValueError('duplicate configuration and training seed')
        groups[family].add(seed)
        # All methods for one seed must use the exact same identity model.
        identity = None if scorer is None else scorer.signature
        previous = inputs.setdefault(seed, identity)
        if previous != identity:
            raise ValueError('same seed uses different frozen identity scorer')
        job['family_sha256'] = family
        jobs.append(job)
    if not allow_fixture and any(seeds != set(SEEDS) for seeds in groups.values()):
        raise ValueError('each configuration requires all three training seeds')
    # Formal parser excludes partial validation and all test before payload use.
    if not allow_fixture:
        from tools.event_track_v2x.run_tracking_v2 import schedule_rows
        rows = schedule_rows(Path(schedule_ref['path']), schedule_ref['sha256'])
    else:
        rows = json.loads(Path(schedule_ref['path']).read_bytes())
    cache = VerifiedForestCache(cache_ref['path'], cache_ref['sha256'])
    metadata = json.loads(cache.manifest_json)
    if not allow_fixture and (metadata['split'] != 'val' or metadata['frame_count'] != 7189
            or set(metadata['sequences']) != {r['sequence_id'] for r in rows}):
        raise ValueError('full official SPD val cache required')
    frozen = frozen_cache_identity(cache)
    for job in jobs:
        if job['checkpoint'] is not None:
            meta = json.loads((Path(job['checkpoint']['path'])/'checkpoint.json').read_bytes())
            if meta['frozen_cache_identity'] != frozen:
                raise ValueError('frozen training and evaluation upstream differ')
        if job.get('allocation_checkpoint') is not None:
            _, meta = load_priority(job['allocation_checkpoint']['path'], job['allocation_checkpoint']['sha256'],
                binding=training_binding(configuration(job), inputs[job['seed']], frozen),
                require_full_train=not allow_fixture)
            if meta['seed'] != job['seed']:
                raise ValueError('priority seed differs from identity seed')
    order, generator = [], random.Random(spec['order_seed'])
    for repetition in range(spec['repetitions']):
        indices = list(range(len(jobs)))
        generator.shuffle(indices)
        order.extend(dict(repetition=repetition, job_index=i) for i in indices)
    return dict(spec, cache=cache_ref, schedule=schedule_ref, jobs=jobs, execution_order=order,
        cohort_mode='fixture-only' if allow_fixture else 'full-spd-val', scheduled_frames=len(rows),
        source_sha256=source_snapshot(), thread_environment=THREAD_ENV, device='cpu',
        paper_eligible=False, physically_equal_resources_claimed=False, exclusive_host_verified=False,
        order_policy='randomized_complete_blocks_no_warmup_exclusion', page_cache_reset=False)


def worker(plan, job, output):
    import torch
    from tools.event_track_v2x.run_persistent_forest_v2 import replay_rows
    if source_snapshot() != plan['source_sha256']:
        raise ValueError('worker source identity differs')
    if any(os.environ.get(k) != v for k, v in THREAD_ENV.items()):
        raise ValueError('worker thread environment differs')
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    config = configuration(job)
    formal = plan['cohort_mode'] == 'full-spd-val'
    _reference(plan['schedule'])
    if formal:
        from tools.event_track_v2x.run_tracking_v2 import schedule_rows
        rows = schedule_rows(Path(plan['schedule']['path']), plan['schedule']['sha256'])
    else:
        if plan['cohort_mode'] != 'fixture-only':
            raise ValueError('unknown resource cohort mode')
        rows = json.loads(Path(plan['schedule']['path']).read_bytes())
    cache = VerifiedForestCache(plan['cache']['path'], plan['cache']['sha256'])
    metadata = json.loads(cache.manifest_json)
    if formal and (metadata['split'] != 'val' or metadata['frame_count'] != 7189
            or set(metadata['sequences']) != {r['sequence_id'] for r in rows}):
        raise ValueError('worker requires full official SPD val cache')
    frozen = frozen_cache_identity(cache)
    scorer, policy = None, None
    if job['checkpoint'] is not None:
        scorer, meta = load_identity_checkpoint(job['checkpoint']['path'], job['checkpoint']['sha256'], config=config.state)
        if (formal and meta.get('full_official_train') is not True
                or meta['frozen_cache_identity'] != frozen or meta['seed'] != job['seed']):
            raise ValueError('worker checkpoint training identity differs')
    elif formal:
        raise ValueError('formal resource worker requires a trained checkpoint')
    if job.get('allocation_checkpoint') is not None:
        policy, _ = load_priority(job['allocation_checkpoint']['path'], job['allocation_checkpoint']['sha256'],
            binding=training_binding(config, scorer.signature, frozen), require_full_train=formal)
    result = replay_rows(cache, rows, output, config, learned_scorer=scorer, allocation_policy=policy,
        plan=dict(kind='rbf_resource_worker_plan_v1', sweep_plan_sha256=_digest(plan), job=job,
            source_sha256=plan['source_sha256'], paper_eligible=False))
    if source_snapshot() != plan['source_sha256']:
        raise ValueError('sources changed during worker replay')
    runtime = dict(python=sys.version, executable=sys.executable, executable_sha256=sha_file(Path(sys.executable).resolve()),
        platform=platform.platform(), host=platform.node(), machine=platform.machine(), processor=platform.processor(),
        numpy=np.__version__, scipy=scipy.__version__, torch=torch.__version__, cpu_count=os.cpu_count(), pid=os.getpid(),
        torch_threads=torch.get_num_threads(), torch_interop_threads=torch.get_num_interop_threads())
    usage = resource.getrusage(resource.RUSAGE_SELF)
    _write(Path(output)/'resource-worker.json', dict(kind='rbf_resource_worker_v1',
        replay_receipt_sha256=sha_file(Path(output)/'receipt.json'), sweep_plan_sha256=_digest(plan),
        runtime=runtime, process_peak_rss_bytes=usage.ru_maxrss*(1 if sys.platform == 'darwin' else 1024),
        rss_scope='worker_high_water_after_replay_source_check_excludes_marker_serialization_interpreter_teardown',
        process_user_cpu_seconds=usage.ru_utime, process_system_cpu_seconds=usage.ru_stime,
        scheduled_frames=len(rows), source_sha256=plan['source_sha256'], paper_eligible=False))
    return result


def validate_execution_modes(receipt, job):
    """Check executed features against the concrete configuration, not aliases."""
    config = configuration(job)
    if job['backend'] == 'learned_beam_recovery' and receipt.get('learned_allocation_enabled') is not True:
        raise ValueError('executed learned allocation differs from declared beam recovery')
    recovery_backbone = type(config) is BeamRecoveryConfig
    recovery = recovery_backbone and config.enable_recovery
    recovery_completion = recovery and config.recovery_completions_per_component > 0
    expected = dict(
        frontier_completion_enabled=job['backend'] in COMPLETION_KINDS or recovery_completion,
        coverage_aware_proposal_admission=job['backend'] in COVERED_COMPLETION_KINDS or recovery_completion,
        beam_recovery_backbone_enabled=recovery_backbone,
        additional_beam_recovery_enabled=recovery,
        irreversible_beam_enabled=job['backend'] in (
            'node_beam', 'batch_beam', 'joint_beam', 'class_bound_joint_beam',
            'slot_bound_joint_beam', 'sparse_slot_bound_joint_beam',
            'reachable_slot_bound_joint_beam') or recovery_backbone and not recovery,
    )
    for name, enabled in expected.items():
        if receipt.get(name) is not enabled:
            raise ValueError('executed '+name+' differs from declared backend configuration')


def inspect_run(output, plan, job):
    output = Path(output)
    marker = json.loads(contained_file(output, 'resource-worker.json').read_bytes())
    if (marker['sweep_plan_sha256'] != _digest(plan) or marker['source_sha256'] != plan['source_sha256']
            or sha_file(output/'receipt.json') != marker['replay_receipt_sha256']):
        raise ValueError('worker final receipt binding differs')
    receipt = json.loads((output/'receipt.json').read_bytes())
    for name, key in (('plan.json', 'plan_sha256'), ('predictions.jsonl', 'predictions_sha256'),
            ('tracking.jsonl', 'tracking_sha256'), ('frame-timings.jsonl', 'frame_timings_sha256')):
        if sha_file(contained_file(output, name)) != receipt[key]:
            raise ValueError('resource run artifact changed: '+name)
    actual_plan = json.loads((output/'plan.json').read_bytes())
    if actual_plan['job'] != job or actual_plan['sweep_plan_sha256'] != _digest(plan):
        raise ValueError('resource job identity differs')
    if (receipt['status'] != 'complete' or receipt['completed_frames'] != plan['scheduled_frames']
            or receipt['scheduled_frames'] != plan['scheduled_frames'] or receipt['allocation_teacher']):
        raise ValueError('incomplete or offline-teacher resource run')
    validate_execution_modes(receipt, job)
    factors, seconds, full_seconds, keys = hashlib.sha256(), [], [], []
    with (output/'frame-timings.jsonl').open('rb') as stream:
        for line in stream:
            row = json.loads(line)
            a, b = row['step_seconds'], row['frame_seconds']
            if any(type(x) not in (int, float) or not math.isfinite(x) for x in (a, b)) or not 0 <= a <= b:
                raise ValueError('invalid frame latency')
            keys.append((row['sequence_id'], row['frame_id']))
            seconds.append(a); full_seconds.append(b)
    with (output/'tracking.jsonl').open('rb') as stream:
        audit_keys = []
        for line in stream:
            row = json.loads(line)['tracking']
            key = (row['sequence_id'], row['event_id'])
            audit_keys.append(key)
            factors.update(canonical([*key, row['factor_rows_sha256']])+b'\n')
    if len(keys) != plan['scheduled_frames'] or len(set(keys)) != len(keys) or keys != audit_keys:
        raise ValueError('per-frame resource and inference coverage differs')
    quantiles = np.quantile(seconds, [.5, .95, .99, 1.]).tolist()
    frame_quantiles = np.quantile(full_seconds, [.5, .95, .99, 1.]).tolist()
    if quantiles != receipt['latency_seconds_p50_p95_p99_max'] or frame_quantiles != receipt['frame_latency_seconds_p50_p95_p99_max']:
        raise ValueError('latency summary differs from raw per-frame measurements')
    databases = 0
    for head in receipt['sequence_heads'].values():
        path = contained_file(output, head['database'])
        if sha_file(path) != head['database_sha256']:
            raise ValueError('result database changed')
        databases += path.stat().st_size
    if (databases != receipt['database_bytes'] or receipt['process_peak_rss_bytes'] <= 0
            or marker['process_peak_rss_bytes'] < receipt['process_peak_rss_bytes']):
        raise ValueError('invalid physical resource measurements')
    return dict(factor_stream_sha256=factors.hexdigest(), predictions_sha256=receipt['predictions_sha256'],
        replay_receipt_sha256=marker['replay_receipt_sha256'], worker_receipt_sha256=sha_file(output/'resource-worker.json'),
        latency_seconds_p50_p95_p99_max=quantiles, frame_latency_seconds_p50_p95_p99_max=frame_quantiles,
        process_peak_rss_bytes=marker['process_peak_rss_bytes'], rss_scope=marker['rss_scope'],
        process_user_cpu_seconds=marker['process_user_cpu_seconds'], process_system_cpu_seconds=marker['process_system_cpu_seconds'],
        database_bytes=databases,
        retained_artifact_bytes=sum(p.stat().st_size for p in output.iterdir() if p.is_file()), runtime=marker['runtime'])


def run_sweep(spec, output, *, allow_fixture=False):
    plan = validate_spec(spec, allow_fixture=allow_fixture)
    output = _new_directory(output)
    _write(output/'plan.json', plan)
    results, factors, predictions, runtime = [], {}, {}, None
    env = dict(os.environ, **THREAD_ENV)
    env.pop('PYTHONPATH', None)
    env.pop('PYTHONHOME', None)
    try:
        for index, item in enumerate(plan['execution_order']):
            job = plan['jobs'][item['job_index']]
            run_dir = output/f'run-{index:04d}'
            command = [sys.executable, str(ROOT/'tools/event_track_v2x/run_resource_sweep_v2.py'),
                '--worker-plan', str(output/'plan.json'), '--worker-plan-sha256', sha_file(output/'plan.json'),
                '--job-index', str(item['job_index']), '--output', str(run_dir)]
            before = time.monotonic()
            with (output/f'run-{index:04d}.log').open('xb') as log:
                subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT,
                    check=True, timeout=plan['timeout_seconds'])
            child_wall = time.monotonic()-before
            measured = inspect_run(run_dir, plan, job)
            actual_runtime = {k: v for k, v in measured['runtime'].items() if k != 'pid'}
            if runtime is not None and actual_runtime != runtime:
                raise ValueError('same-sweep runtime or machine identity changed')
            runtime = actual_runtime
            # Check raw observations AND frozen potentials across all methods
            # for the same seed; seeds intentionally have different learned q.
            expected = factors.setdefault(job['seed'], measured['factor_stream_sha256'])
            if measured['factor_stream_sha256'] != expected:
                raise ValueError('same-seed methods received different raw observations or factors')
            repeated = predictions.setdefault(item['job_index'], measured['predictions_sha256'])
            if measured['predictions_sha256'] != repeated:
                raise ValueError('repeated fresh-process predictions differ')
            results.append(dict(index=index, **item, job_id=job['id'], seed=job['seed'],
                family_sha256=job['family_sha256'], output=run_dir.name,
                child_wall_seconds=child_wall, **measured))
            _write(output/f'run-{index:04d}-verified.json', results[-1])
            print(json.dumps(dict(kind='resource_sweep_progress', completed=len(results),
                planned=len(plan['execution_order']), job_id=job['id'])), flush=True)
        if source_snapshot() != plan['source_sha256']:
            raise ValueError('sweep sources changed')
        receipt = dict(kind=KIND, status='complete', plan_sha256=sha_file(output/'plan.json'), runs=results,
            cohort_mode=plan['cohort_mode'], raw_factors_equal_within_seed=True,
            repeated_predictions_identical=True, separate_fresh_process_per_run=True,
            resource_limits_are_algorithmic_not_matched_physical_budgets=True,
            physically_equal_resources_claimed=False, paper_eligible=False)
        _write(output/'receipt.json', receipt)
        return receipt
    except BaseException as error:
        _write(output/'failure.json', dict(kind=KIND, status='failed', completed_runs=len(results),
            plan_sha256=sha_file(output/'plan.json'), error_type=type(error).__name__, error=str(error),
            completed_runs_are_not_a_complete_comparison=True, paper_eligible=False))
        raise
