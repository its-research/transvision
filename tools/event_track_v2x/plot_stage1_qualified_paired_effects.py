#!/usr/bin/env python3
"""Export pinned Stage1 exploratory figures; never qualify a failed replay."""
import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--evidence-root', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    root = args.evidence_root / 'artifacts'
    track = root / 'rbf-stage1-all-thirteen-full-stream-provenance-20261006T170240089706Z/independent-sequence-statistics/statistics-receipt.json'
    nusc_dir = root / 'rbf-stage1-nuscenes-complete-shared-bootstrap-after-admission-20261006T181425503427Z'
    nusc = nusc_dir / 'actual-full-bootstrap/bootstrap-receipt.json'
    reader = nusc_dir / 'independent-statistics-output-readback.json'
    pins = {
        track: 'c4115d1eacc5020e085956a305b774ef709af115f8f7a7cd7311af35a316a0a7',
        nusc: '3def79e22a108c8dbe275f433d332e616f66532e7736a162b65ba7b91592b176',
        reader: '13d112cc76e5cc307d21c9e83f99f7b628091b8ef9aaddaa3694efe954ff13d8',
        nusc_dir / 'actual-full-bootstrap/all13-complete-bootstrap-draw-metrics.npy': '02cd4dd172176b74f94694f76bf062c90d3c20423a194d54e41d2d9eda799474',
        track.parent / 'shared-sequence-draws.npy': '1fe443fc7a659043da35b2397127a258dc333d3aa0b0aebf8a4dd63788e2da9c',
    }
    for path, digest in pins.items():
        assert sha(path) == digest, str(path)
    t = json.loads(track.read_text())
    n = json.loads(nusc.read_text())
    plan = json.loads((nusc_dir / 'actual-full-bootstrap/plan.json').read_text())
    assert sha(nusc_dir / 'actual-full-bootstrap/plan.json') == n['plan_sha256']
    pins[nusc_dir / 'actual-full-bootstrap/plan.json'] = n['plan_sha256']
    draws = np.load(track.parent / 'shared-sequence-draws.npy', allow_pickle=False)
    assert np.array_equal(draws, np.random.Generator(np.random.PCG64(1337)).integers(0, 21, size=(10000, 21), dtype=np.int64))
    assert n['all13_complete_10000_sequence_draw_metrics_present']
    assert t['original_M3_seed2027_failure_preserved'] and n['original_M3_seed2027_failure_preserved']
    assert n['models']['M3']['qualified_10000_sequence_intervals'] is None
    assert n['models']['M1']['original_run_count'] == 1
    assert n['models']['M1']['raw_descriptive_metrics']['amota']['sample_std'] is None
    assert set(n['qualified_strict_replay_paired_comparisons']) == {'M1-minus-M0', 'M2-minus-M0', 'M4-minus-M0'}
    raw = np.load(nusc_dir / 'actual-full-bootstrap/all13-complete-bootstrap-draw-metrics.npy', allow_pickle=False)
    assert raw.shape == (13, 10000, 15) and np.isfinite(raw).all()
    campaign_path = root / 'rbf-stage1-nuscenes-all-local-score-subsets-config-order-v2-20261006T180941649939Z/campaign-receipt.json'
    assert sha(campaign_path) == plan['input_sha256'][str(campaign_path)]
    pins[campaign_path] = sha(campaign_path)
    # The pinned producer iterates campaign results; JSON key order is not an axis contract.
    run_order = [entry['run_id'] for entry in json.loads(campaign_path.read_text())['results']]
    assert len(run_order) == 13 and set(run_order) == set(n['per_run'])
    metric_names = n['nuScenes_bootstrap_metric_scope']
    assert len(metric_names) == 15 and len(set(metric_names)) == 15
    rows = []
    table = []
    for model in ['M0', 'M1', 'M2', 'M3', 'M4']:
        runs = n['models'][model]['run_ids']
        assert runs == t['models'][model]['original_run_ids']
        metrics = {}
        for name in ['HOTA', 'AssA', 'DetA', 'IDF1', 'amota', 'amotp', 'fp', 'fn', 'ids', 'frag']:
            engine = 'trackeval' if name[0].isupper() else 'nuscenes'
            vals = [float(t['per_run'][r]['raw'][engine][name]) for r in runs]
            mean = math.fsum(vals) / len(vals)
            sd = statistics.stdev(vals) if len(vals) > 1 else None
            described = t['models'][model][engine][name]
            assert math.isclose(mean, described['mean'], abs_tol=1e-12, rel_tol=0)
            assert sd is None if described['sample_std'] is None else math.isclose(sd, described['sample_std'], abs_tol=1e-12, rel_tol=0)
            metrics[name] = {'mean': mean, 'sample_std': sd}
        table.append({'model': model, 'run_ids': runs, 'run_count': len(runs), 'strict_replay_qualified': n['models'][model]['all_strict_replay_pass'], 'metrics': metrics})
        if model in ['M0', 'M3']:
            continue
        for metric, scale in [('amota', 100.0), ('HOTA', 100.0), ('frag', 1.0)]:
            pair = f'{model}-minus-M0'
            if metric == 'HOTA':
                point = t['qualified_strict_replay_paired_comparisons'][pair]['observed_difference'][metric]
                interval = t['qualified_strict_replay_paired_comparisons'][pair]['sequence_bootstrap_95_percent_interval'][metric]
                recalculated = metrics[metric]['mean'] - t['models']['M0']['trackeval'][metric]['mean']
            else:
                point = n['models'][model]['raw_descriptive_metrics'][metric]['mean'] - n['models']['M0']['raw_descriptive_metrics'][metric]['mean']
                interval = n['qualified_strict_replay_paired_comparisons'][pair][metric]
                mi = metric_names.index(metric)
                candidate = raw[[run_order.index(r) for r in runs], :, mi].mean(axis=0)
                baseline = raw[[run_order.index(r) for r in n['models']['M0']['run_ids']], :, mi].mean(axis=0)
                lo, hi = np.quantile(candidate - baseline, [0.025, 0.975], method='linear')
                assert math.isclose(lo, interval['low'], abs_tol=1e-12, rel_tol=0)
                assert math.isclose(hi, interval['high'], abs_tol=1e-12, rel_tol=0)
                recalculated = metrics[metric]['mean'] - n['models']['M0']['raw_descriptive_metrics'][metric]['mean']
            assert math.isclose(point, recalculated, abs_tol=1e-12, rel_tol=0)
            assert interval['low'] <= point <= interval['high']
            rows.append({'model': model, 'metric': metric, 'point': point * scale, 'low': interval['low'] * scale, 'high': interval['high'] * scale})
    values = {'scope': 'SPD seen-val car; 21 native sequences, 3316 frames; exploratory', 'rows': rows, 'raw_model_metrics': table,
              'bootstrap': '10000 paired whole-sequence draws; PCG64 seed1337; original seed mean, no seed resampling',
              'M3': 'Historical seed2027 byte replay failure retained; raw descriptive values only; no qualified interval',
              'M1': 'One deterministic execution; no seed sample SD', 'HOTA_AMOTA_display_unit': 'percentage points',
              'Frag_display_unit': 'nuScenes fragmentation count difference; lower is better',
              'amotp_unit': 'SPD nuScenes center distance meters; lower is better; not V2V native IoU AMOTP',
              'input_sha256': {str(k): v for k, v in pins.items()}, 'whole_goal_qualified': False}
    (args.output / 'reviewed-figure-values.json').write_text(json.dumps(values, indent=2, sort_keys=True) + '\n')
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'axes.labelcolor': '#333333', 'text.color': '#252525', 'svg.fonttype': 'none'})
    fig, axs = plt.subplots(1, 3, figsize=(13.2, 5.9), sharey=True)
    fig.subplots_adjust(left=.13, right=.97, top=.72, bottom=.31, wspace=.28)
    fig.text(.04, .94, 'Stage 1: paired effects relative to M0', fontsize=18, weight='bold')
    fig.text(.04, .885, 'SPD seen-val car | 21 native sequences | 3,316 frames | exploratory', fontsize=11, color='#555555')
    fig.text(.04, .835, 'M4 has fewer fragments alongside lower AMOTA and HOTA.', fontsize=12)
    colors = {'M1': '#426B88', 'M2': '#A17936', 'M4': '#645C85'}
    markers = {'M1': 'o', 'M2': 's', 'M4': 'D'}
    order = ['M1', 'M2', 'M3', 'M4']
    specs = [('amota', 'AMOTA', 'Difference (percentage points)', (-11, 3), [-10, -5, 0], 'Higher is better'),
             ('HOTA', 'HOTA', 'Difference (percentage points)', (-4.7, .7), [-4, -2, 0], 'Higher is better'),
             ('frag', 'Frag', 'Difference (count)', (-135, 80), [-100, -50, 0, 50], 'Lower is better')]
    for ax, (metric, title, label, limits, ticks, direction) in zip(axs, specs):
        ax.axvline(0, color='#777777', linewidth=1, linestyle='--', zorder=1)
        ax.grid(axis='x', color='#E6E6E6', linewidth=.6)
        for row in rows:
            if row['metric'] != metric:
                continue
            model = row['model']; y = order.index(model)
            ax.errorbar(row['point'], y, xerr=[[row['point'] - row['low']], [row['high'] - row['point']]],
                        fmt=markers[model], color=colors[model], markersize=6, linewidth=1.7, capsize=4, zorder=3)
            ax.annotate(f"{row['point']:+.2f}", (row['point'], y), xytext=(0, -17), textcoords='offset points', ha='center', fontsize=9, color='#333333')
        ax.text(.5, 2, 'No qualified interval', transform=ax.get_yaxis_transform(), ha='center', va='center', color='#777777', fontsize=9)
        ax.set_title(title + '\n' + direction, fontsize=11, pad=13)
        ax.set_xlabel(label, labelpad=11)
        ax.set_xlim(*limits); ax.set_xticks(ticks)
        ax.set_ylim(3.55, -.45)
        ax.set_yticks(range(4), ['M1 (1 run)', 'M2 (3 seeds)', 'M3', 'M4 (3 seeds)'])
        for spine in ['top', 'right', 'left']:
            ax.spines[spine].set_visible(False)
        ax.spines['bottom'].set_color('#AAAAAA')
        ax.tick_params(axis='y', length=0)
    footers = [
        'Points: original model means minus the 3-seed M0 mean. Lines: 95% paired whole-sequence percentile intervals.',
        '10,000 shared draws, PCG64 seed 1337. Uncertainty is conditional on these 21 seen-val sequences; seeds are not resampled.',
        'M1: one deterministic run, no seed sample SD. M3: historical seed2027 byte replay failure retained; no qualified interval.',
        'Frag uses the nuScenes score-selected count. This figure does not establish causal recovery gains or generalization.'
    ]
    for i, line in enumerate(footers):
        fig.text(.04, .18 - i * .039, line, fontsize=9, color='#555555')
    for ext in ['png', 'svg']:
        fig.savefig(args.output / f'stage1-paired-effects.{ext}', dpi=220, facecolor='white')
    plt.close(fig)
    # A separate grayscale artifact makes the marker/label distinction inspectable.
    from PIL import Image
    Image.open(args.output / 'stage1-paired-effects.png').convert('L').save(args.output / 'stage1-paired-effects-grayscale.png')
    for path, digest in pins.items():
        assert sha(path) == digest, str(path)
    receipt = {'kind': 'rbf_stage1_pinned_exploratory_figure_export_v1', 'input_sha256': values['input_sha256'],
               'source_sha256': sha(Path(__file__)), 'output_sha256': {p.name: sha(p) for p in args.output.iterdir()},
               'all9_visible_point_interval_bindings_checked': True, 'nuScenes_selected_intervals_recomputed_from_all10000_draws': True,
               'HOTA_interval_reused_from_pinned_qualified_whole_sequence_statistics': True, 'visual_inspection_pending': True,
               'matplotlib_version': matplotlib.__version__, 'numpy_version': np.__version__, 'paper_complete': False, 'goal_status': 'active'}
    (args.output / 'export-receipt.json').write_text(json.dumps(receipt, indent=2, sort_keys=True) + '\n')
    print(json.dumps({'output': str(args.output), 'all9_visible_bindings_checked': True, 'goal_status': 'active'}))


if __name__ == '__main__':
    main()
