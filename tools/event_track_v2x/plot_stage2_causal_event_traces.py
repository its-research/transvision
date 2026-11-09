#!/usr/bin/env python3
"""Source-bound scientific figure of accepted finite event cases and negatives."""
import argparse
import hashlib
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--plan', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    plan = json.loads(args.plan.read_text())
    assert sha(Path(__file__)) == plan['source_sha256']
    pins = {Path(k): v for k, v in plan['input_sha256'].items()}
    for p, h in pins.items():
        assert sha(p) == h
    admission = json.loads(Path(plan['admission']).read_text())
    witness = json.loads(Path(plan['witness']).read_text())
    assert witness['all_witness_metadata_independently_verified']
    assert witness['accepted_recovery_runs'] == 15 and witness['no_recovery_runs'] == 3
    runs = {r['run']: r for r in admission['checks']}
    specs = [('crossing-seed1337-K1', 'Crossing, K=1', '#356d91', 'o', '-'),
             ('occlusion_reappearance-seed1337-K4', 'Occlusion, K=4', '#8f702f', 's', '--'),
             ('occlusion_reappearance-seed1337-K1', 'Occlusion, K=1', '#ac564a', '^', ':')]
    traces = []
    for key, label, color, marker, line in specs:
        r = runs[key]
        traces.append(dict(run=key, label=label, event_ids=[e['event_id'] for e in r['events']],
            current_c_roots=[e['chosen_roots'][2] for e in r['events']],
            exact_model_regret=[e['exact_same_action_model_regret'] for e in r['events']],
            accepted_current_output_recovery=r['recovery_claimed']))
    raw = json.loads(Path(plan['occlusion_candidate']).read_text())['raw']
    assert traces[0]['current_c_roots'][-1] == traces[1]['current_c_roots'][-1] == 'b'
    assert traces[2]['current_c_roots'][-1] == 'a' and not traces[2]['accepted_current_output_recovery']
    values = dict(traces=traces, local_observations=[r for r in raw if r['node']['source_id'] == 0],
        input_sha256={str(k): v for k, v in pins.items()}, fixture_seed=1337, expansion_budget=3,
        axis='actual event ordinal, not elapsed time; compute events have no new detections',
        scope='finite synthetic explicit-factor CPU cases; untrained fixture seed; no real-data or resource-efficiency claim',
        all_18_configurations_accepted=18, recovered_configurations=15, fixed_budget_no_recovery_configurations=3)
    args.output.mkdir(exist_ok=False)
    (args.output / 'figure-values.json').write_text(json.dumps(values, indent=2)+'\n')
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'svg.fonttype': 'none',
        'text.color': '#242424', 'axes.labelcolor': '#333333'})
    fig, ax = plt.subplots(1, 3, figsize=(14.8, 6.1))
    fig.subplots_adjust(left=.065, right=.975, top=.72, bottom=.30, wspace=.32)
    fig.text(.04, .94, 'Stage 2: current-decision recovery and a fixed-budget failure', fontsize=18, weight='bold')
    fig.text(.04, .885, 'Finite synthetic graph | fixture seed 1337 | K=1 / K=4 | 3 paid expansion steps per event', fontsize=11)
    fig.text(.04, .835, 'All 18 cases independently checked: 15 current-output recoveries; 3 occlusion K=1 failures retained.', fontsize=11)
    local = [r for r in raw if r['node']['source_id'] == 0]
    aa = [r for r in local if r['mean'][1] == -2.]
    bb = [r for r in local if r['mean'][1] == 2.]
    ax[0].plot([r['state_us']/1e6 for r in aa], [r['mean'][0] for r in aa], color='#356d91', marker='o', label='A, y=-2 m')
    ax[0].scatter([r['state_us']/1e6 for r in bb], [r['mean'][0] for r in bb], color='#8f702f', marker='s', label='B, y=+2 m')
    ax[0].annotate('B missing at 1.2 s', xy=(1.2, -.03), xytext=(1.035, -.32),
        arrowprops=dict(arrowstyle='->', color='#666666'), fontsize=9)
    ax[0].set(xlabel='Information timestamp (s)', ylabel='Observed x position (m)', xticks=[1., 1.2, 1.4],
        xlim=(.97,1.43), ylim=(-.48,.48), title='Observed detections\nDistinct lanes; x order swaps')
    ax[0].legend(loc='upper left', fontsize=9, frameon=False)
    for trace, (_, label, color, marker, line) in zip(traces, specs):
        xs = list(range(len(trace['event_ids'])))
        ax[1].step(xs, [1 if r == 'b' else 0 for r in trace['current_c_roots']], where='post',
            color=color, marker=marker, markersize=3.5, linestyle=line, label=label)
        ax[2].step(xs, trace['exact_model_regret'], where='post', color=color, marker=marker, markersize=3.5, linestyle=line, label=label)
    ax[1].set(xlabel='Actual event ordinal', ylabel='Current root assigned to old node c',
        yticks=[0,1], yticklabels=['a','b'], ylim=(-.12,1.3), title='Current old-node identity\nPreviously only frontier-covered b')
    ax[2].set(xlabel='Actual event ordinal', ylabel='Exact normalized model regret',
        ylim=(-.025,.67), title='Same legal action set\nLower is better')
    ax[2].legend(loc='upper right', fontsize=8.7, frameon=False)
    for a in ax[1:]:
        a.set_xlim(-.3,14.3); a.set_xticks([0,2,4,6,8,10,12,14])
    for a in ax:
        a.grid(axis='y', linewidth=.6, color='#E7E7E7')
        a.spines['top'].set_visible(False); a.spines['right'].set_visible(False)
        a.spines['bottom'].set_color('#999999'); a.spines['left'].set_color('#999999')
    notes = [
        'Crossing has 14 events; occlusion has 15. Ordinals differ because occlusion includes a one-object-visible middle event.',
        'After both objects return, 12 predeclared compute events add no observations. Budgets and numeric tolerances remain fixed.',
        'b is the evaluator identity and a supported finite model alternative; it starts outside active classes and inside the frontier.',
        'Missing detections are not an existence-likelihood factor. This figure does not establish learned or real-data recovery gains.',
        'K=1 and K=4 use the same expansion cap; this is not a same-resource measurement. No seed confidence interval is inferred.'
    ]
    for i, note in enumerate(notes):
        fig.text(.04,.20-i*.034,note,fontsize=9,color='#555555')
    for ext in ['png','svg']:
        fig.savefig(args.output/f'stage2-causal-event-traces.{ext}',dpi=200,facecolor='white')
    plt.close(fig)
    from PIL import Image
    Image.open(args.output/'stage2-causal-event-traces.png').convert('L').save(args.output/'stage2-causal-event-traces-grayscale.png')
    for p,h in pins.items():assert sha(p)==h
    receipt=dict(kind='rbf_stage2_source_bound_finite_event_trace_figure_v1',source_sha256=sha(Path(__file__)),
        output_sha256={p.name:sha(p) for p in args.output.iterdir()}, input_sha256=values['input_sha256'],
        actual_displayed_values_bound=True, visual_inspection_pending=True, matplotlib_version=matplotlib.__version__,
        learned_or_real_data_recovery_claimed=False, full_Stage2_or_paper_complete=False, goal_status='active')
    (args.output/'export-receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps({'output':str(args.output),'visible_traces':len(traces)}))


if __name__ == '__main__':
    main()
