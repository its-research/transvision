#!/usr/bin/env python3
"""Explicit protocol, immutable configuration, preflight and paper replay
CLI."""
import argparse
import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def preflight(manifest):
    from transvision.models.event_track_v2x.detection_cache_v2 import sha_file
    from transvision.models.event_track_v2x.paper_protocol import PaperProtocol
    PaperProtocol(**manifest['protocol'])
    required = {'dataset_manifest', 'detector_checkpoint', 'feature_checkpoint', 'calibration', 'identity_checkpoint', 'schedule', 'environment_lock'}
    if manifest.get('allocation') == 'learned':
        required.add('priority_checkpoint')
    roles = [a['role'] for a in manifest.get('assets', [])]
    if len(roles) != len(set(roles)):
        raise ValueError('duplicate asset role')
    findings = [dict(asset=role, status='missing', detail='Required paper asset has no explicit path and hash') for role in sorted(required - set(roles))]
    for asset in manifest.get('assets', []):
        path = Path(asset['path'])
        status = 'missing' if not path.is_file() else 'hash_mismatch' if sha_file(path) != asset['sha256'] else 'verified'
        findings.append(dict(asset=asset['role'], path=str(path.absolute()), status=status))
    for package in manifest.get('python_modules', ['numpy', 'torch', 'scipy']):
        try:
            found = importlib.util.find_spec(package) is not None
        except (ImportError, ModuleNotFoundError):
            found = False
        findings.append(dict(asset=package, status='verified' if found else 'missing'))
    return dict(
        kind='rbf_asset_preflight_v1',
        ready=all(f['status'] == 'verified' for f in findings),
        findings=findings,
        training_started=False,
        formal_success_receipt=False,
        dependency_check='module_discovery_only_not_compiled_operator_execution')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    commands = p.add_subparsers(dest='command', required=True)
    check = commands.add_parser('preflight')
    check.add_argument('manifest')
    config = commands.add_parser('configuration')
    config.add_argument('--method', default='rbf', choices=['rbf', 'geometry', 'single_vehicle', 'single_remote', 'topk', 'irreversible', 'mht', 'jpda', 'pkf'])
    config.add_argument('--allocation', default='bound', choices=['bound', 'fixed', 'learned', 'teacher'])
    config.add_argument('--width', type=int, default=4)
    run = commands.add_parser('replay')
    for name in ('cache', 'cache-sha256', 'schedule', 'schedule-sha256', 'configuration', 'configuration-sha256', 'dataset', 'split', 'output'):
        run.add_argument('--' + name, required=True)
    for name in ('checkpoint', 'checkpoint-sha256', 'priority', 'priority-sha256'):
        run.add_argument('--' + name)
    run.add_argument('--device', default='cpu')
    run.add_argument('--fixture', action='store_true')
    a = p.parse_args()
    if a.command == 'preflight':
        result = preflight(json.loads(Path(a.manifest).read_text()))
        print(json.dumps(result, indent=2))
        return 0 if result['ready'] else 2
    from transvision.models.event_track_v2x.paper_runtime import default_configuration, replay
    if a.command == 'configuration':
        print(json.dumps(default_configuration(a.method, allocation=a.allocation, width=a.width), sort_keys=True, indent=2))
        return 0
    from transvision.models.event_track_v2x.detection_cache_v2 import sha_file
    from transvision.models.event_track_v2x.forest_cache_stream import VerifiedForestCache
    from transvision.models.event_track_v2x.forest_potentials import GeometryForestScorer
    from transvision.models.event_track_v2x.forest_tracking import PaperForestTrackingConfig as ForestTrackingConfig
    from transvision.models.event_track_v2x.forest_training_checkpoint import load_identity_checkpoint
    from transvision.models.event_track_v2x.forest_training_data import frozen_cache_identity
    from transvision.models.event_track_v2x.paper_native_cache import NativePaperCache
    from transvision.models.event_track_v2x.paper_protocol import PaperProtocol
    protocol = PaperProtocol(a.dataset, a.split)
    for name in ('schedule', 'configuration'):
        if sha_file(getattr(a, name)) != getattr(a, name + '_sha256'):
            raise ValueError(name + ' binding differs')
    configuration = json.loads(Path(a.configuration).read_text())
    events = [json.loads(line) for line in Path(a.schedule).read_text().splitlines() if line.strip()]
    cache = (VerifiedForestCache if a.dataset == 'spd' else NativePaperCache)(a.cache, a.cache_sha256)
    if json.loads(cache.manifest_json).get('fixture', False) and not a.fixture:
        raise ValueError('native fixture cache requires fixture run')
    state = ForestTrackingConfig(**configuration['state'])
    policy, binding = None, {}
    if configuration['method'] == 'geometry':
        if a.checkpoint or a.priority:
            raise ValueError('geometry has no learned checkpoint')
        scorer = GeometryForestScorer(process_noise=state.process_noise)
    else:
        if not a.checkpoint or not a.checkpoint_sha256:
            raise ValueError('missing identity checkpoint and hash')
        scorer, checkpoint = load_identity_checkpoint(a.checkpoint, a.checkpoint_sha256, config=state, device=a.device)
        if not configuration.get('history_features', True):
            from transvision.models.event_track_v2x.forest_potentials import LearnedForestScorer
            scorer = LearnedForestScorer(scorer.model, **scorer.options, history_enabled=False)
        if checkpoint.get('dataset') != a.dataset or checkpoint['frozen_cache_identity'] != frozen_cache_identity(cache):
            raise ValueError('checkpoint dataset/frozen upstream producers differ')
        if not a.fixture and checkpoint.get('fixture', True):
            raise ValueError('fixture checkpoint cannot support real run')
        binding = dict(
            candidate_protocol=protocol.candidates,
            dataset=a.dataset,
            fit_split='train',
            checkpoint_sha256=a.checkpoint_sha256,
            model_sha256=checkpoint['model_sha256'],
            seed=checkpoint['seed'],
            frozen_cache_identity=checkpoint['frozen_cache_identity'])
        if configuration['allocation'] == 'learned':
            if not a.priority or not a.priority_sha256:
                raise ValueError('missing frozen allocation checkpoint')
            from transvision.models.event_track_v2x.allocation_training import load_priority, training_binding
            from transvision.models.event_track_v2x.covered_completion_tracking import PersistentCoveredCompletionConfig
            policy, priority_manifest = load_priority(
                a.priority,
                a.priority_sha256,
                binding=training_binding(PersistentCoveredCompletionConfig(state=state, **configuration['limits']), scorer.signature, frozen_cache_identity(cache)),
                require_full_train=not a.fixture)
            binding['priority_sha256'] = a.priority_sha256
    print(
        json.dumps(
            replay(cache, events, a.output, protocol=protocol, configuration=configuration, scorer=scorer, model_binding=binding, allocation_policy=policy, fixture=a.fixture),
            sort_keys=True))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
