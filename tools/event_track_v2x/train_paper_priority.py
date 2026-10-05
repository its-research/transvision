#!/usr/bin/env python3
"""Export actual train-side paper teacher traces and reuse the priority
trainer."""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def export(replay, receipt_sha256, output):
    import numpy as np

    from transvision.models.event_track_v2x.allocation_policy import FEATURES, RECIPE, TARGET
    from transvision.models.event_track_v2x.allocation_training import DATA_KIND, _validate_group, allocation_sources, training_binding, validate_backend_binding
    from transvision.models.event_track_v2x.paper_runtime_selection import allocation_configuration
    from transvision.models.event_track_v2x.detection_cache_v2 import canonical, sha_file
    root = Path(replay)
    output = Path(output)
    if sha_file(root / 'receipt.json') != receipt_sha256:
        raise ValueError('teacher receipt changed')
    receipt = json.loads((root / 'receipt.json').read_bytes())
    plan = json.loads((root / 'plan.json').read_bytes())
    if (receipt['status'] != 'software_replay_completed' or plan['protocol']['split'] != 'train' or plan['configuration']['allocation'] != 'teacher'):
        raise ValueError('actual completed train-only teacher replay required')
    for name, sha in receipt['files'].items():
        if sha_file(root / name) != sha:
            raise ValueError('teacher artifact changed')
    sources = allocation_sources()
    if any(plan['source_sha256'].get(k) != v for k, v in sources.items()):
        raise ValueError('teacher feature/solver sources differ')
    # Cache producer binding was sealed by the identity checkpoint loader.
    producer = plan['model_binding'].get('frozen_cache_identity')
    if producer is None:
        raise ValueError('teacher lacks frozen upstream producer binding')
    c = plan['configuration']
    config = allocation_configuration(c)
    binding = training_binding(config, plan['scorer_signature'], producer)
    validate_backend_binding(binding, plan_sources=plan['source_sha256'])
    groups = {}
    frames = 0
    for line in (root / 'audit.jsonl').read_bytes().splitlines():
        audit = json.loads(line)
        frames += 1
        if audit.get('training_trace_only') is not True or audit.get('offline_counterfactual_probes') is not True:
            raise ValueError('nonteacher audit')
        scene = audit['sequence_id']
        groups.setdefault(scene, [])
        for event in audit['allocation_trace']:
            record = event['allocation_training']
            if record['feature_recipe'] != RECIPE or record['target_recipe'] != TARGET:
                raise ValueError('teacher recipe differs')
            options = record['candidates']
            x = np.asarray([r['features'] for r in options])
            y = np.asarray([r['target'] for r in options])
            _validate_group(x, y)
            for r, features, target in zip(options, x, y):
                expected = features[0] * (r['model_bound_before'] - r['model_bound_after']) / max(1, r['charged_steps'])
                if abs(expected - target) > 1e-12:
                    raise ValueError('teacher target arithmetic differs')
            groups[scene].append(dict(features=x.tolist(), targets=y.tolist()))
    if frames != receipt['completed_events'] or sorted(groups) != receipt['completed_sequences']:
        raise ValueError('incomplete teacher cohort')
    if output.exists():
        raise ValueError('create-once teacher data required')
    output.mkdir()
    shards = []
    for index, (scene, rows) in enumerate(sorted(groups.items())):
        path = output / f'sequence-{index:04d}.jsonl'
        with path.open('xb') as f:
            for row in rows:
                f.write(canonical(row) + b'\n')
        shards.append(dict(sequence_id=scene, path=path.name, sha256=sha_file(path), groups=len(rows), rows=sum(len(r['targets']) for r in rows)))
    manifest = dict(
        kind=DATA_KIND,
        split='train',
        feature_recipe=RECIPE,
        feature_names=FEATURES,
        target_recipe=TARGET,
        shards=shards,
        replay_receipt_sha256=receipt_sha256,
        source_sha256=sources,
        binding=binding,
        full_official_train_trace=False,
        labels_are_model_not_true_risk=True,
        future_or_gt_inputs=False,
        paper_eligible=False,
        fixture=receipt['fixture'])
    (output / 'manifest.json').write_bytes(canonical(manifest))
    return manifest


def main():
    p = argparse.ArgumentParser(description=__doc__)
    s = p.add_subparsers(dest='command', required=True)
    e = s.add_parser('export')
    for key in ('replay', 'receipt-sha256', 'output'):
        e.add_argument('--' + key, required=True)
    f = s.add_parser('fit')
    for key in ('data', 'manifest-sha256', 'output'):
        f.add_argument('--' + key, required=True)
    f.add_argument('--epochs', type=int, default=10)
    f.add_argument('--fixture', action='store_true')
    a = p.parse_args()
    if a.command == 'export':
        result = export(a.replay, a.receipt_sha256, a.output)
    else:
        from transvision.models.event_track_v2x.allocation_training import fit_priority
        result = fit_priority(a.data, a.manifest_sha256, a.output, epochs=a.epochs, require_full_train=not a.fixture, select_best_train_holdout=True)
    print(json.dumps(result, sort_keys=True))


if __name__ == '__main__':
    main()
