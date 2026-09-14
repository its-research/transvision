#!/usr/bin/env python3
"""Train-only resource scan: immutable plan, fresh-process runs, constrained
freeze.

No GPU jobs or evaluator runs are launched by planning. Run is explicit. Frozen choices are software receipts, not certificates of official/full dataset coverage.
"""
import argparse
import copy
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from transvision.models.event_track_v2x.detection_cache_v2 import canonical, sha_file  # noqa: E402
from transvision.models.event_track_v2x.paper_protocol import PAPER, SEEDS, PaperProtocol  # noqa: E402


def write(path, value):
    with Path(path).open('xb') as stream:
        stream.write(canonical(value))


def checked(binding):
    if set(binding) != {'path', 'sha256'}:
        raise ValueError('explicit path and sha256 required')
    path = Path(binding['path'])
    if not path.is_absolute() or not path.is_file() or sha_file(path) != binding['sha256']:
        raise ValueError('missing or changed asset: ' + str(path))
    return path


def fresh_directory(path):
    path = Path(path).absolute()
    if path.exists() or any(p.is_symlink() for p in (path, *path.parents)):
        raise ValueError('new directory without symlink ancestors required')
    path.mkdir()
    return path


def bindings(plan):
    return [plan[k] for k in ('cache_manifest', 'schedule', 'gt_manifest', 'environment_lock')]


def plan_scan(spec, output):
    protocol = PaperProtocol(**spec['protocol'])
    protocol.require_train()
    if protocol.candidates != PAPER or type(spec['fixture']) is not bool:
        raise ValueError('explicit main-protocol fixture status required')
    if tuple(spec['seeds']) != SEEDS:
        raise ValueError('freeze requires exactly the three declared seeds')
    for asset in bindings(spec):
        checked(asset)
    cache = json.loads(checked(spec['cache_manifest']).read_bytes())
    gt = json.loads(checked(spec['gt_manifest']).read_bytes())
    if cache['split'] != 'train' or gt['protocol'] != spec['protocol'] or gt['fixture'] != spec['fixture']:
        raise ValueError('cache/GT selection split or fixture differs')
    if cache.get('fixture', False) and not spec['fixture']:
        raise ValueError('fixture cache cannot support a real scan')
    objective = spec['objective']
    if objective not in ('HOTA', 'IDF1', 'AMOTA', 'AMOTP'):
        raise ValueError('unsupported predeclared objective')
    limits = spec['constraints']
    allowed = {'peak_rss_bytes', 'state_file_bytes', 'p95_seconds', 'model_forward_calls', 'posterior_search_steps', 'action_search_steps', 'assignment_solves'}
    if not limits or not set(limits) <= allowed or any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or v <= 0 for v in limits.values()):
        raise ValueError('positive predeclared resource constraints required')
    jobs, seen = [], set()
    for candidate in spec['candidates']:
        name = candidate['name']
        if not name or any(c not in 'abcdefghijklmnopqrstuvwxyz0123456789-_' for c in name) or name in seen:
            raise ValueError('unique safe candidate names required')
        seen.add(name)
        configuration = json.loads(checked(candidate['configuration']).read_bytes())
        if configuration['state']['candidate_protocol'] != PAPER or configuration['allocation'] == 'teacher':
            raise ValueError('main protocol inference configuration required')
        # Explicit candidate configs materialize the frozen resource grid. Do not
        # change teacher-trained allocation bindings by silently editing limits.
        for seed in SEEDS:
            identity = None if configuration['method'] == 'geometry' else candidate['checkpoints'][str(seed)]
            priority = candidate.get('priorities', {}).get(str(seed))
            if identity is not None:
                checkpoint_path = checked(identity)
                if checkpoint_path.name != 'checkpoint.json' or json.loads(checkpoint_path.read_bytes())['seed'] != seed:
                    raise ValueError('identity checkpoint manifest/seed differs')
            if configuration['allocation'] == 'learned':
                if priority is None:
                    raise ValueError('each learned configuration/seed requires a matching priority checkpoint')
                priority_path = checked(priority)
                if priority_path.name != 'checkpoint.json' or json.loads(priority_path.read_bytes())['seed'] != seed:
                    raise ValueError('priority checkpoint manifest/seed differs')
            elif priority is not None:
                raise ValueError('priority supplied for a non-learned allocation')
            jobs.append(dict(id=name + '-' + str(seed), candidate=name, seed=seed, configuration=configuration, checkpoint=identity, priority=priority))
    if not jobs:
        raise ValueError('no resource candidates')
    output = fresh_directory(output)
    result = {k: copy.deepcopy(spec[k]) for k in ('protocol', 'fixture', 'seeds', 'objective', 'constraints', 'cache_manifest', 'schedule', 'gt_manifest', 'environment_lock')}
    result.update(
        kind='rbf_resource_scan_plan_v1', jobs=jobs, source_sha256=sha_file(__file__), selection_split='train', equal_resources_claimed=False, full_dataset_verified=False)
    write(output / 'plan.json', result)
    return result


def run_scan(plan_path, plan_sha256, output, *, python, evaluator_python, timeout=3600):
    plan_path = checked(dict(path=str(Path(plan_path).absolute()), sha256=plan_sha256))
    plan = json.loads(plan_path.read_bytes())
    PaperProtocol(**plan['protocol']).require_train()
    if plan['kind'] != 'rbf_resource_scan_plan_v1' or plan['source_sha256'] != sha_file(__file__):
        raise ValueError('unknown or stale source plan')
    for asset in bindings(plan):
        checked(asset)
    if platform.system() not in ('Darwin', 'Linux'):
        raise ValueError('ru_maxrss units not defined on this platform')
    for executable in (python, evaluator_python):
        if not Path(executable).is_absolute() or not Path(executable).is_file():
            raise ValueError('explicit existing Python executables required')
    if type(timeout) is not int or timeout <= 0:
        raise ValueError('positive subprocess timeout required')
    output = fresh_directory(output)
    source_files = list((ROOT / 'transvision/models/event_track_v2x').glob('*.py')) + [
        ROOT / 'tools/event_track_v2x/persistent_mht_tracking.py', ROOT / 'tools/event_track_v2x/run_paper.py', ROOT / 'tools/event_track_v2x/evaluate_paper.py'
    ]
    source_hashes = {str(p): sha_file(p) for p in source_files}
    runtime = dict(
        source_sha256=source_hashes,
        device='cpu',
        system=platform.system(),
        machine=platform.machine(),
        platform=platform.platform(),
        python=dict(path=python, sha256=sha_file(python)),
        evaluator_python=dict(path=evaluator_python, sha256=sha_file(evaluator_python)),
        environment_lock=plan['environment_lock'],
        scan_source_sha256=sha_file(__file__),
        runner_source_sha256=sha_file(ROOT / 'tools/event_track_v2x/run_paper.py'),
        evaluator_source_sha256=sha_file(ROOT / 'tools/event_track_v2x/evaluate_paper.py'),
        threads={
            'OMP_NUM_THREADS': '1',
            'OPENBLAS_NUM_THREADS': '1'
        })
    write(output / 'runtime.json', runtime)
    completed = []
    try:
        for job in plan['jobs']:
            root = fresh_directory(output / job['id'])
            configuration = root / 'configuration.json'
            write(configuration, job['configuration'])
            run = root / 'replay'
            command = [
                python,
                str(ROOT / 'tools/event_track_v2x/run_paper.py'), 'replay', '--cache',
                str(checked(plan['cache_manifest']).parent), '--cache-sha256', plan['cache_manifest']['sha256'], '--schedule',
                str(checked(plan['schedule'])), '--schedule-sha256', plan['schedule']['sha256'], '--configuration',
                str(configuration), '--configuration-sha256',
                sha_file(configuration), '--dataset', plan['protocol']['dataset'], '--split', 'train', '--output',
                str(run)
            ]
            for field in ('checkpoint', 'priority'):
                if job[field] is not None:
                    command += ['--' + field, str(checked(job[field]).parent), '--' + field + '-sha256', job[field]['sha256']]
            if plan['fixture']:
                command.append('--fixture')
            env = dict(os.environ, **runtime['threads'], PYTHONDONTWRITEBYTECODE='1')
            # A fresh process for every trial makes process-lifetime peak RSS
            # method-specific. No process pool or reused GPU model instance.
            for stage, argv in [('replay', command),
                                ('evaluation', [
                                    evaluator_python,
                                    str(ROOT / 'tools/event_track_v2x/evaluate_paper.py'), '--gt-manifest',
                                    str(checked(plan['gt_manifest'])), '--gt-sha256', plan['gt_manifest']['sha256'], '--replay',
                                    str(run), '--receipt-sha256', 'PENDING', '--output',
                                    str(root / 'evaluation')
                                ])]:
                if stage == 'evaluation':
                    argv[argv.index('PENDING')] = sha_file(run / 'receipt.json')
                process = subprocess.run(argv, env=env, capture_output=True, timeout=timeout)
                with (root / (stage + '.stdout')).open('xb') as stream:
                    stream.write(process.stdout)
                with (root / (stage + '.stderr')).open('xb') as stream:
                    stream.write(process.stderr)
                if process.returncode:
                    raise RuntimeError(job['id'] + ' ' + stage + ' failed: ' + str(process.returncode))
            completed.append(
                dict(
                    id=job['id'],
                    files={
                        name: sha_file(root / name)
                        for name in ('configuration.json', 'replay/receipt.json', 'replay/plan.json', 'replay/resources.json', 'replay/predictions.jsonl', 'evaluation/report.json')
                    }))
        for asset in bindings(plan):
            checked(asset)
        if any(sha_file(p) != digest for p, digest in source_hashes.items()):
            raise ValueError('implementation changed during resource scan')
        if sha_file(plan_path) != plan_sha256 or any(sha_file(runtime[k]['path']) != runtime[k]['sha256']
                                                     for k in ('python', 'evaluator_python')) or sha_file(__file__) != plan['source_sha256']:
            raise ValueError('scan inputs or runtime changed')
        receipt = dict(
            kind='rbf_resource_scan_receipt_v1',
            status='completed',
            plan_sha256=plan_sha256,
            runtime_sha256=sha_file(output / 'runtime.json'),
            jobs=completed,
            fixture=plan['fixture'],
            full_dataset_verified=False,
            paper_results_verified=False)
        write(output / 'receipt.json', receipt)
        return receipt
    except BaseException as error:
        write(output / 'failure.json', dict(status='failed', completed_jobs=[j['id'] for j in completed], error=str(error)))
        raise


def freeze_scan(plan_path, plan_sha256, run, receipt_sha256, output):
    plan = json.loads(checked(dict(path=str(Path(plan_path).absolute()), sha256=plan_sha256)).read_bytes())
    PaperProtocol(**plan['protocol']).require_train()
    if plan['kind'] != 'rbf_resource_scan_plan_v1' or plan['source_sha256'] != sha_file(__file__):
        raise ValueError('unknown or stale source plan')
    for asset in bindings(plan):
        checked(asset)
    events = [json.loads(line) for line in checked(plan['schedule']).read_bytes().splitlines() if line.strip()]
    events_sha256 = hashlib.sha256(canonical(events)).hexdigest()
    run = Path(run).absolute()
    receipt = json.loads(checked(dict(path=str(run / 'receipt.json'), sha256=receipt_sha256)).read_bytes())
    if (receipt['status'] != 'completed' or receipt['plan_sha256'] != plan_sha256 or receipt['fixture'] != plan['fixture']
            or sha_file(run / 'runtime.json') != receipt['runtime_sha256']):
        raise ValueError('scan evidence differs')
    if [r['id'] for r in receipt['jobs']] != [j['id'] for j in plan['jobs']]:
        raise ValueError('incomplete or reordered trial cohort')
    runtime = json.loads((run / 'runtime.json').read_bytes())
    scale = {'Darwin': 1, 'Linux': 1024}[runtime['system']]
    candidates = {}
    for job, evidence in zip(plan['jobs'], receipt['jobs']):
        root = run / job['id']
        required_files = {'configuration.json', 'replay/receipt.json', 'replay/plan.json', 'replay/resources.json', 'replay/predictions.jsonl', 'evaluation/report.json'}
        if set(evidence['files']) != required_files:
            raise ValueError('incomplete artifact evidence')
        for name, sha in evidence['files'].items():
            if sha_file(root / name) != sha:
                raise ValueError('trial artifact changed')
        report = json.loads((root / 'evaluation/report.json').read_bytes())
        replay = json.loads((root / 'replay/receipt.json').read_bytes())
        recorded_plan = json.loads((root / 'replay/plan.json').read_bytes())
        costs = json.loads((root / 'replay/resources.json').read_bytes())
        if (report['protocol'] != plan['protocol'] or report['fixture'] != plan['fixture'] or report['status'] != 'evaluated' or replay['status'] != 'software_replay_completed'
                or recorded_plan['configuration'] != job['configuration'] or recorded_plan['cache_sha256'] != plan['cache_manifest']['sha256']
                or recorded_plan['events_sha256'] != events_sha256 or replay['fixture'] != plan['fixture']
                or report['input_sha256'].get(str(root / 'replay/receipt.json')) != evidence['files']['replay/receipt.json']
                or report['input_sha256'].get(str(root / 'replay/predictions.jsonl')) != evidence['files']['replay/predictions.jsonl']
                or report['input_sha256'].get(plan['gt_manifest']['path']) != plan['gt_manifest']['sha256']):
            raise ValueError('trial metrics are not bound to the expected inputs')
        if job['checkpoint'] is not None and (recorded_plan['model_binding']['seed'] != job['seed']
                                              or recorded_plan['model_binding']['checkpoint_sha256'] != job['checkpoint']['sha256']):
            raise ValueError('checkpoint seed differs')
        values = dict(
            peak_rss_bytes=replay['peak_rss_native_units'] * scale,
            state_file_bytes=sum(d['file_bytes'] for d in costs['databases'].values()),
            p95_seconds=costs['latency']['p95_seconds'],
            **costs['costs'])
        score = report['metrics'][plan['objective']]
        if not math.isfinite(score) or any(not math.isfinite(values[k]) or values[k] < 0 for k in plan['constraints']):
            raise ValueError('nonfinite or negative measurement')
        candidates.setdefault(job['candidate'], []).append(
            dict(seed=job['seed'], score=score, measured={k: values[k]
                                                          for k in plan['constraints']}, feasible=all(values[k] <= limit for k, limit in plan['constraints'].items())))
    summaries = []
    for name, rows in sorted(candidates.items()):
        if sorted(r['seed'] for r in rows) != list(SEEDS):
            raise ValueError('missing three-seed candidate results')
        summaries.append(dict(candidate=name, mean_score=sum(r['score'] for r in rows) / len(rows), feasible=all(r['feasible'] for r in rows), trials=rows))
    feasible = [r for r in summaries if r['feasible']]
    if not feasible:
        raise ValueError('no candidate meets all constraints on every seed; nothing frozen')
    direction = 1 if plan['objective'] == 'AMOTP' else -1
    selected = min(feasible, key=lambda r: (direction * r['mean_score'], r['candidate']))
    configuration = next(j['configuration'] for j in plan['jobs'] if j['candidate'] == selected['candidate'])
    output = fresh_directory(output)
    write(output / 'configuration.json', configuration)
    result = dict(
        kind='rbf_train_selected_configuration_v1',
        status='software_selection_frozen',
        selection_split='train',
        fixture=plan['fixture'],
        selected_candidate=selected['candidate'],
        objective=plan['objective'],
        constraints=plan['constraints'],
        candidates=summaries,
        plan_sha256=plan_sha256,
        receipt_sha256=receipt_sha256,
        configuration_sha256=sha_file(output / 'configuration.json'),
        full_dataset_verified=False,
        equal_resources_claimed=False,
        paper_results_verified=False,
        deterministic_baseline_seeds='repeated fresh-process trials, not independently trained models')
    write(output / 'receipt.json', result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    plan = sub.add_parser('plan')
    plan.add_argument('--spec', required=True)
    plan.add_argument('--output', required=True)
    run = sub.add_parser('run')
    freeze = sub.add_parser('freeze')
    for command in (run, freeze):
        for key in ('plan', 'plan-sha256', 'output'):
            command.add_argument('--' + key, required=True)
    run.add_argument('--python', required=True)
    run.add_argument('--evaluator-python', required=True)
    run.add_argument('--timeout', type=int, default=3600)
    freeze.add_argument('--run', required=True)
    freeze.add_argument('--receipt-sha256', required=True)
    args = parser.parse_args()
    if args.command == 'plan':
        result = plan_scan(json.loads(Path(args.spec).read_bytes()), args.output)
    elif args.command == 'run':
        result = run_scan(args.plan, args.plan_sha256, args.output, python=args.python, evaluator_python=args.evaluator_python, timeout=args.timeout)
    else:
        result = freeze_scan(args.plan, args.plan_sha256, args.run, args.receipt_sha256, args.output)
    print(json.dumps(result, sort_keys=True))


if __name__ == '__main__':
    main()
