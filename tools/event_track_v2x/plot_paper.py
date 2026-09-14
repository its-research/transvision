#!/usr/bin/env python3
"""Render Figure 1-5 only from explicit, hash-bound evidence; no demo
values."""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def render(manifest, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    from transvision.models.event_track_v2x.detection_cache_v2 import canonical, sha_file
    from transvision.models.event_track_v2x.paper_protocol import require_same_protocol
    from transvision.models.event_track_v2x.paper_reports import duration_summary
    if manifest['figure'] not in range(1, 6) or manifest['evidence_kind'] not in ('fixture', 'real'):
        raise ValueError('explicit paper figure/evidence kind required')
    if not manifest['sources']:
        raise ValueError('source artifacts required')
    for record in manifest['sources']:
        if sha_file(record['path']) != record['sha256']:
            raise ValueError('figure source changed')
    data = manifest['data']
    number = manifest['figure']
    if number == 1:
        if len(data['moments']) != 3 or data['gt_used_online'] is not False:
            raise ValueError('three causal snapshots and offline-only GT required')
        fig, axes = plt.subplots(2, 3, figsize=(12, 6))
        for column, row in enumerate(data['moments']):
            ax = axes[0, column]
            for kind, marker in [('observations', 'x'), ('ground_truth', 'o'), ('predictions', 's')]:
                for point in row[kind]:
                    ax.scatter(point['x'], point['y'], marker=marker, label=kind)
                    ax.annotate(str(point['id']), (point['x'], point['y']))
            ax.set(xlabel='world x (m)', ylabel='world y (m)', title=str(row['timestamp_us']) + ' us', aspect='equal')
            branches = row['branch_mass']
            axes[1, column].bar([b['id'] for b in branches], [b['mass'] for b in branches])
            axes[1, column].set(ylabel='model branch mass', ylim=(0, 1))
    elif number == 2:
        if data['raw_factor_bytes'] < 0 or data['window_us'] <= 0 or len(data['stages']) != 3:
            raise ValueError('explicit factor cost, window and three stages required')
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        for ax, stage in zip(axes, data['stages']):
            labels = ['leaf: ' + x for x in stage['explicit_leaves']] + ['frontier: ' + x for x in stage['disjoint_frontier']]
            ax.text(.02, .92, '\n'.join(labels), va='top', family='monospace', transform=ax.transAxes)
            ax.text(.02, .1, 'action: ' + stage['action'] + '\naudit: ' + stage['audit_sha256'][:12], transform=ax.transAxes)
            ax.set_title(stage['name'])
            ax.axis('off')
        fig.suptitle(f"raw factors {data['raw_factor_bytes']} bytes; window {data['window_us']/1e6:g} s")
    elif number == 3:
        if data['independent_enumeration'] is not True:
            raise ValueError('Figure 3 requires an independent exhaustive oracle')
        fig, axes = plt.subplots(1, 2, figsize=(9, 4))
        for row in data['rows']:
            if not 0 <= row['exact_omitted_mass'] <= row['upper_bound'] + 1e-12 <= 1 + 1e-12:
                raise ValueError('model mass bound violated')
        axes[0].scatter([r['exact_omitted_mass'] for r in data['rows']], [r['upper_bound'] for r in data['rows']])
        axes[0].plot([0, 1], [0, 1], '--', color='gray')
        axes[0].set(xlabel='exact model omitted mass', ylabel='numeric model upper bound')
        axes[1].scatter([r['evidence_likelihood_ratio'] for r in data['rows']], [r['eta_y'] for r in data['rows']])
        axes[1].set(xlabel='evidence likelihood ratio', ylabel='updated model eta')
    elif number == 4:
        rows = data['rows']
        require_same_protocol(rows)
        if len({r['hardware_id'] for r in rows}) != 1:
            raise ValueError('mixed hardware resource curve')
        if set(data['expected_mht_widths']) != {r['width'] for r in rows if r['method'] == 'mht'}:
            raise ValueError('missing frozen MHT resource control')
        fig, axes = plt.subplots(2, 2, figsize=(10, 8))
        for method in sorted({r['method'] for r in rows}):
            subset = [r for r in rows if r['method'] == method]
            for column, key in enumerate(('total_peak_memory_bytes', 'p95_latency_seconds')):
                order = sorted(subset, key=lambda r: r[key])
                if any(r['includes_raw_factors_frontier_states'] is not True for r in order):
                    raise ValueError('incomplete resource accounting')
                for row, metric in enumerate(('HOTA', 'IDF1')):
                    axes[row, column].plot([r[key] for r in order], [r[metric] for r in order], 'o-', label=method)
                    axes[row, column].set(xlabel=key, ylabel=metric)
                    axes[row, column].legend()
    else:
        episodes = data['episodes']
        duration_summary([{k: e[k] for k in ('start_us', 'end_us', 'censored')} for e in episodes])
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        for index, e in enumerate(episodes):
            axes[0].plot([e['start_us'] / 1e6, e['end_us'] / 1e6], [index, index], color='gray')
            axes[0].scatter(e['end_us'] / 1e6, index, marker='>' if e['censored'] else 'o')
        for index, e in enumerate(data['recovery_events']):
            if type(e['success']) is not bool:
                raise ValueError('explicit failed and successful recovery outcomes required')
            axes[1].scatter(e['timestamp_us'] / 1e6, index, marker='o' if e['success'] else 'x')
        axes[0].set(xlabel='time (s); > means right censored', ylabel='identity error episode')
        axes[1].set(xlabel='time (s); x means failed recovery', ylabel='recovery event')
    if manifest['evidence_kind'] == 'fixture':
        fig.text(.5, .005, 'SOFTWARE FIXTURE — NOT PAPER RESULTS', ha='center', color='red')
    fig.tight_layout()
    output = Path(output)
    if output.exists():
        plt.close(fig)
        raise ValueError('figure output already exists')
    output.mkdir()
    try:
        fig.savefig(output / f'figure-{number}.svg')
        fig.savefig(output / f'figure-{number}.png', dpi=160)
        (output / 'manifest.json').write_bytes(canonical(manifest))
    finally:
        plt.close(fig)
    return dict(figure=number, evidence_kind=manifest['evidence_kind'], sources_verified=True, artifacts={p.name: sha_file(p) for p in output.iterdir() if p.is_file()})


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest', required=True)
    p.add_argument('--output', required=True)
    a = p.parse_args()
    print(json.dumps(render(json.loads(Path(a.manifest).read_text()), a.output), sort_keys=True))


if __name__ == '__main__':
    main()
