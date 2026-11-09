#!/usr/bin/env python3
"""Create a fresh train-only candidate; never claims independent admission."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tools.event_track_v2x.v2v4real_motion_fit_inputs import load, sha
from tools.event_track_v2x.v2v4real_motion_fit import RECIPE, offline_matches, select_gate, fit_model, apply_model

SOURCES = ['tools/event_track_v2x/'+n+'.py' for n in (
    'fit_v2v4real_causal_motion', 'v2v4real_motion_fit_inputs', 'v2v4real_motion_fit',
    'v2v4real_causal_detector_motion')]


def write(path, value):
    path.open('x').write(json.dumps(value, sort_keys=True, indent=2, allow_nan=False)+'\n')


def execute(args):
    output = Path(args.output).absolute()
    if output.exists() or any(p.is_symlink() for p in (output, *output.parents)):
        raise ValueError('fresh separate motion candidate output required')
    sources = {str(ROOT/p): sha(ROOT/p) for p in SOURCES}
    rows, targets, groups, pins, detector = load(args.detector_receipt, args.detector_sha256,
        args.target_acceptance, args.target_sha256, args.partition, args.partition_sha256)
    output.mkdir(parents=True)
    write(output/'recipe-before-fitting.json', dict(recipe=RECIPE, sources=sources, input_sha256=pins,
          created_at_utc=datetime.now(timezone.utc).isoformat(), official_test_used=False))
    for seed, frames in rows.items():
        for row in frames:
            row['gt_matches'] = offline_matches(row['states_source_legacy7'], row['raw_scores'],
                                                row['target_states_source'])
    selection = select_gate(rows, targets, groups,
        progress=lambda x: print(json.dumps(dict(stage='train_motion_gate_selection',
            speed=x['max_speed_mps'], gap=x['max_gap_ordinals'], MSE=x['mean_seed_group_vector_MSE'])), flush=True))
    write(output/'selection.json', selection)
    results = []
    for seed, frames in sorted(rows.items()):
        model, motion = fit_model(frames, targets, groups, selection['selected'])
        model['seed'] = seed
        folder = output/f'seed{seed}'; folder.mkdir()
        write(folder/'model.json', model)
        chunks = [apply_model({'side': row['side']}, pred, model) for row, pred in zip(frames, motion)]
        offsets = np.cumsum([0]+[len(row['raw_scores']) for row in frames])
        arrays = {k: np.concatenate([chunk[k] for chunk in chunks]) for k in chunks[0]}
        arrays.update(offsets=offsets,
            predecessor_candidates=np.concatenate([p['predecessor_candidates'] for p in motion]),
            raw_velocity_source=np.concatenate([p['velocity_source'] for p in motion]))
        np.savez_compressed(folder/'motion-arrays.npz', **arrays)
        index = []
        for i, (row, pred) in enumerate(zip(frames, motion)):
            index.append(dict(raw_row_sha256=row['row_sha256'], sequence_id=row['sequence_id'],
                side=row['side'], frame_ordinal=row['frame_ordinal'], offset=int(offsets[i]),
                candidates=len(row['raw_scores']), dependencies=pred['dependencies'], reasons=pred['reasons']))
        (folder/'motion-frame-index.jsonl').open('x').write(''.join(json.dumps(x)+'\n' for x in index))
        result = dict(seed=seed, source_rows=len(frames), candidates=int(offsets[-1]),
            measured_valid=int(arrays['velocity_measured_valid'].sum()),
            initial_prior_applied=int(arrays['initial_prior_applied'].sum()),
            full9_residual_samples={s:{k:v['samples'] for k,v in info['statuses'].items()}
                                   for s,info in model['sides'].items()},
            artifacts={p.name:dict(bytes=p.stat().st_size,sha256=sha(p)) for p in folder.iterdir()})
        results.append(result)
        print(json.dumps(dict(stage='train_motion_candidate_fitted', **result)), flush=True)
    for path, expected in {**sources, **pins}.items():
        if sha(path) != expected:
            raise ValueError('source/input changed during training candidate')
    proof = dict(kind='v2v4real_motion_covariance_train_fit_candidate_v1', fit_split='train',
        official_test_used=False, fit_groups=groups, input_sha256=pins, source_sha256=sources,
        recipe_sha256=sha(output/'recipe-before-fitting.json'), selection_sha256=sha(output/'selection.json'),
        detector_input_receipt_sha256=args.detector_sha256, target_acceptance_sha256=args.target_sha256,
        seeds=results, all_candidate_velocities_finite=True, independent_acceptance_verified=False,
        nine_state_cache_admitted=False, tracking_accepted=False, paper_metrics_complete=False)
    write(output/'candidate-receipt.json', proof)
    print(json.dumps(dict(receipt=str(output/'candidate-receipt.json'), sha256=sha(output/'candidate-receipt.json'))),flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('detector-receipt','detector-sha256','target-acceptance','target-sha256',
                 'partition','partition-sha256','output'):
        parser.add_argument('--'+name,required=True)
    execute(parser.parse_args())
