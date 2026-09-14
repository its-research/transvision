"""Paper tables and descriptive paired uncertainty from sealed result
records."""
import json
from pathlib import Path

import numpy as np

from .detection_cache_v2 import canonical, sha_file
from .paper_protocol import SEEDS, require_same_protocol

METRICS = ('HOTA', 'AssA', 'DetA', 'IDF1', 'AMOTA', 'AMOTP', 'FP', 'FN', 'IDS', 'Frag')


def paired_bootstrap(left, right, *, clusters, seed=1337, draws=10000):
    """Arrays are seed x sequence.

    Resample recordings, retaining all their seeds.
    """
    left, right = np.asarray(left, float), np.asarray(right, float)
    if (left.ndim != 2 or left.shape != right.shape or left.shape[1] != len(clusters) or not left.size or not np.isfinite(left).all() or not np.isfinite(right).all()
            or type(draws) is not int or draws < 1):
        raise ValueError('aligned finite seed-by-sequence metrics required')
    names = sorted(set(clusters))
    if len(names) < 2:
        raise ValueError('at least two independent clusters required for an interval')
    delta = (left - right).mean(axis=0)
    groups = [np.asarray([i for i, c in enumerate(clusters) if c == name]) for name in names]
    rng, values = np.random.default_rng(seed), []
    for _ in range(draws):
        selected = np.concatenate([groups[i] for i in rng.integers(0, len(groups), len(groups))])
        values.append(float(delta[selected].mean()))
    return dict(
        mean_difference=float(delta.mean()),
        lower=float(np.quantile(values, .025)),
        upper=float(np.quantile(values, .975)),
        seed=seed,
        draws=draws,
        cluster_count=len(names),
        uncertainty='descriptive_paired_cluster_bootstrap')


def duration_summary(episodes):
    """Preserve censored observations; never infer a recovery beyond
    observation."""
    complete, censored = [], []
    for e in episodes:
        if (set(e) != {'start_us', 'end_us', 'censored'} or type(e['censored']) is not bool or type(e['start_us']) is not int or type(e['end_us']) is not int or e['start_us'] < 0
                or e['end_us'] < e['start_us']):
            raise ValueError('invalid identity duration episode')
        (censored if e['censored'] else complete).append((e['end_us'] - e['start_us']) / 1e6)
    return dict(
        completed_seconds=complete,
        censored_observed_seconds=censored,
        completed_mean_seconds=float(np.mean(complete)) if complete else None,
        count=len(episodes),
        recovered_count=len(complete),
        censored_count=len(censored))


def genuine_recovery(events):
    flags = ('previously_explicit', 'previously_frontier_covered', 'expanded_now', 'changed_current_action', 'gt_used_online', 'offline_identity_correct')
    success = 0
    for e in events:
        if any(type(e.get(k)) is not bool for k in flags):
            raise ValueError('explicit recovery witnesses required')
        if e['gt_used_online']:
            raise ValueError('GT leakage invalidates recovery evidence')
        success += int(not e['previously_explicit'] and e['previously_frontier_covered'] and e['expanded_now'] and e['changed_current_action'] and e['offline_identity_correct'])
    return dict(eligible_events=len(events), recovered_events=success, recovery_rate=success / len(events) if events else None)


def validate_result(record):
    if record.get('status') != 'evaluated' or record.get('table') not in (1, 2, 3, 4):
        raise ValueError('only completed, table-assigned evaluations may be tabulated')
    if record['evidence_kind'] not in ('fixture', 'real'):
        raise ValueError('explicit fixture/real evidence kind required')
    if type(record['deterministic']) is not bool:
        raise ValueError('explicit determinism required')
    if record['deterministic']:
        if record['seed'] is not None:
            raise ValueError('deterministic repetitions are not training seeds')
    elif record['seed'] not in SEEDS:
        raise ValueError('unsupported training seed')
    if record['protocol']['evaluation_class'] != 'car':
        raise ValueError('car-only evaluation required')
    if set(record['metrics']) != set(METRICS) or any(not np.isfinite(v) for v in record['metrics'].values()):
        raise ValueError('complete finite metric vector required')
    if record['table'] in (2, 4):
        require_same_protocol([record])
    for name in ('prediction', 'evaluation'):
        binding = record['artifacts'][name]
        if sha_file(binding['path']) != binding['sha256']:
            raise ValueError(name + ' evidence changed')
    evaluation = json.loads(Path(record['artifacts']['evaluation']['path']).read_bytes())
    if (evaluation.get('status') != 'evaluated' or evaluation.get('metrics') != record['metrics'] or evaluation.get('protocol') != record['protocol']
            or evaluation.get('fixture') != (record['evidence_kind'] == 'fixture')
            or record['artifacts']['prediction']['sha256'] not in evaluation.get('input_sha256', {}).values()):
        raise ValueError('table numbers/protocol are not bound to the evaluated prediction')


def build_tables(records):
    records = list(records)
    if not records:
        raise ValueError('no evaluated results')
    for r in records:
        validate_result(r)
    if len({r['evidence_kind'] for r in records}) != 1:
        raise ValueError('cannot mix fixture and real results')
    groups = {}
    for r in records:
        # Native/system tables explicitly separate protocol variants.
        key = (r['table'], r['protocol']['dataset'], r['method'], canonical(r['protocol']))
        groups.setdefault(key, []).append(r)
    tables = {str(i): [] for i in range(1, 5)}
    for (table, dataset, method, _), rows in sorted(groups.items()):
        if any(r['deterministic'] != rows[0]['deterministic'] for r in rows):
            raise ValueError('mixed method determinism')
        expected = {None} if rows[0]['deterministic'] else set(SEEDS)
        if {r['seed'] for r in rows} != expected or len(rows) != len(expected):
            raise ValueError('missing or repeated seed; cannot select the best run')
        require_same_protocol(rows)
        matrix = np.asarray([[r['metrics'][m] for m in METRICS] for r in rows])
        tables[str(table)].append(
            dict(
                dataset=dataset,
                method=method,
                protocol=rows[0]['protocol'],
                mean=dict(zip(METRICS,
                              matrix.mean(0).tolist())),
                sample_std=dict(zip(METRICS,
                                    matrix.std(0, ddof=1).tolist())) if len(rows) > 1 else None,
                seed_results=[dict(seed=r['seed'], metrics=r['metrics']) for r in rows],
                resource_results=[r['resources'] for r in rows],
                diagnostic_events=[r.get('diagnostic_events', {}) for r in rows],
                recovery_statistics=[r.get('recovery_statistics', {}) for r in rows],
                amotp_definition=rows[0]['amotp_definition']))
    # Same-input table comparisons must share all frozen producer identities.
    for table in (2, 4):
        for dataset in {r['protocol']['dataset'] for r in records if r['table'] == table}:
            require_same_protocol([r for r in records if r['table'] == table and r['protocol']['dataset'] == dataset])
    return dict(
        kind='rbf_paper_tables_v1',
        evidence_kind=records[0]['evidence_kind'],
        tables=tables,
        missing_tables=[int(t) for t, rows in tables.items() if not rows],
        scientific_success_assessed=False,
        cross_dataset_amotp_averaged=False)


def write_tables(records, output):
    report = build_tables(records)
    output = Path(output)
    output.mkdir(exist_ok=False)
    with (output / 'tables.json').open('xb') as stream:
        stream.write(canonical(report))
    for table, rows in report['tables'].items():
        lines = ['| Dataset | Method | HOTA | IDF1 | AMOTA |', '|---|---|---:|---:|---:|']
        for row in rows:
            lines.append('| ' + row['dataset'] + ' | ' + row['method'] + ' | ' + ' | '.join(f"{row['mean'][metric]:.6f}" for metric in ('HOTA', 'IDF1', 'AMOTA')) + ' |')
        prefix = 'FIXTURE SOFTWARE TEST — NOT PAPER RESULTS\n\n' if report['evidence_kind'] == 'fixture' else ''
        with (output / f'table-{table}.md').open('x') as stream:
            stream.write(prefix + '\n'.join(lines) + '\n')
    return report
