"""Separate recovery ablation; original cache/replay and restricted priority path.

Teacher and learned priorities require a recovery-off-specific training binding.
No unrestricted learned-priority checkpoint transfer is supported.
"""
import json
import resource
import time
from dataclasses import asdict
from pathlib import Path
from .recovery_off_tracking import RecoveryOffConfig, RecoveryOffTracker
from .exclusive_paper_runtime import default_configuration as _bound_configuration
from .detection_cache_v2 import canonical, sha_file
from .experiment_progress import ExperimentProgress
from .forest_cache_stream import CacheDelivery
from .forest_tracking import PaperForestTrackingConfig as ForestTrackingConfig
from .paper_protocol import PaperProtocol
from .persistent_cache_stream import PersistentForestCacheStream

BACKEND = 'exclusive_event_boundary_recovery_off_v1'


def make_backend(path, sequence, configuration, *, allocation_policy=None):
    if configuration['method'] != 'rbf' or configuration['backend'] != BACKEND:
        raise ValueError('explicit recovery-off RBF backend binding required')
    from .recovery_off_allocation import RecoveryOffTeacher, RecoveryOffLearned
    policy = configuration['allocation']
    if policy not in {'bound', 'teacher', 'learned'}:
        raise ValueError('unsupported recovery-off allocation')
    if (policy == 'learned') != (allocation_policy is not None):
        raise ValueError('learned allocation requires its own frozen policy exclusively')
    config = RecoveryOffConfig(state=ForestTrackingConfig(**configuration['state']),
                               **configuration['limits'])
    cls = {'bound': RecoveryOffTracker, 'teacher': RecoveryOffTeacher, 'learned': RecoveryOffLearned}[policy]
    return cls(path, sequence_id=sequence, config=config,
               **({} if allocation_policy is None else {'allocation_policy': allocation_policy}))


def default_configuration(method='rbf', *, allocation='bound', width=4):
    if method != 'rbf' or allocation not in {'bound', 'teacher', 'learned'}:
        raise ValueError('unsupported recovery-off allocation')
    base = _bound_configuration(method, allocation=allocation, width=width)
    return dict(base, backend=BACKEND, limits=dict(base['limits'], recovery_off_version=1))


def write_once(path, payload):
    with Path(path).open('xb') as stream:
        stream.write(canonical(payload))


def replay(cache, events, output, *, protocol, configuration, scorer, model_binding, allocation_policy=None, fixture=False):
    if type(protocol) is not PaperProtocol or type(fixture) is not bool:
        raise TypeError('explicit protocol and fixture flag required')
    if configuration['state']['candidate_protocol'] != protocol.candidates:
        raise ValueError('runner/candidate protocol differs')
    from .forest_potentials import GeometryForestScorer, LearnedForestScorer
    is_geometry = type(scorer) is GeometryForestScorer
    if configuration['method'] == 'geometry':
        if not is_geometry:
            raise ValueError('geometry baseline requires geometry scorer')
    elif type(scorer) is not LearnedForestScorer:
        raise ValueError('learned paper method requires a frozen learned scorer')
    if not is_geometry and (model_binding.get('candidate_protocol') != protocol.candidates or model_binding.get('dataset') != protocol.dataset
                            or model_binding.get('fit_split') != 'train'):
        raise ValueError('model training protocol differs')
    if not is_geometry and scorer.options.get('history_enabled', True) != configuration.get('history_features', True):
        raise ValueError('history ablation differs from actual forward')
    manifest = json.loads(cache.manifest_json)
    if manifest['split'] != protocol.split:
        raise ValueError('cache split differs')
    if configuration['allocation'] == 'teacher':
        protocol.require_train()
    events = tuple(events)
    seen, previous, sequences = set(), {}, set()
    for event in events:
        if set(event) != {'sequence_id', 'frame_id', 'reference_us', 'decision_us', 'event_id', 'deliveries'}:
            raise ValueError('unexpected schedule fields, including possible GT')
        sequence = event['sequence_id']
        if (type(event['reference_us']) is not int or type(event['decision_us']) is not int or not 0 <= event['reference_us'] <= event['decision_us']
                or event['reference_us'] <= previous.get(sequence, -1) or (sequence, event['event_id']) in seen):
            raise ValueError('nonmonotonic/duplicate schedule')
        previous[sequence] = event['reference_us']
        seen.add((sequence, event['event_id']))
        sequences.add(sequence)
        for raw in event['deliveries']:
            delivery = CacheDelivery(**raw)
            _, metadata = cache.describe(delivery)
            if delivery.sequence_id != sequence or not max(metadata['box_reference_timestamp_us'],
                                                           metadata['source_image_timestamp_us']) <= delivery.arrival_us <= event['decision_us']:
                raise ValueError('unavailable delivery before output creation')
    if not events or not sequences <= set(manifest['sequences']):
        raise ValueError('empty or out-of-cohort schedule')
    output = Path(output).absolute()
    if output.exists() or any(p.is_symlink() for p in (output, *output.parents)):
        raise ValueError('new output without symlink ancestors required')
    output.mkdir()
    source_root = Path(__file__).resolve().parents[3]
    sources = {str(p.relative_to(source_root)): sha_file(p) for p in (source_root / 'transvision/models/event_track_v2x').glob('*.py')}
    sources['tools/event_track_v2x/persistent_mht_tracking.py'] = sha_file(source_root / 'tools/event_track_v2x/persistent_mht_tracking.py')
    plan = dict(
        kind='rbf_paper_replay_v1',
        protocol=asdict(protocol),
        configuration=configuration,
        cache_sha256=cache.manifest_sha256,
        model_binding=model_binding,
        scorer_signature=scorer.signature,
        events_sha256=__import__('hashlib').sha256(canonical(events)).hexdigest(),
        fixture=fixture,
        source_sha256=sources,
        expected_events=len(events),
        expected_sequences=sorted(sequences))
    write_once(output / 'plan.json', plan)
    trackers, streams, timings, databases = {}, {}, [], {}
    progress = ExperimentProgress(
        "paper_replay_events", len(events),
        context=dict(sequence_ids=sorted(sequences), events_sha256=plan['events_sha256'],
                     output_directory=str(output)),
        eta_scope='this replay schedule only; excludes remaining schedules and independent acceptance')
    try:
        with (output / 'predictions.jsonl').open('xb') as predictions, (output / 'audit.jsonl').open('xb') as audits:
            for index, event in enumerate(events):
                sequence = event['sequence_id']
                if sequence not in trackers:
                    path = output / ('sequence-' + str(len(trackers)) + '.sqlite')
                    trackers[sequence] = make_backend(path, sequence, configuration, allocation_policy=allocation_policy)
                    # Match the frozen training feature recipe. The sealed
                    # cache's reference clocks define the sequence origin;
                    # availability still gates every delivery independently.
                    origin = min(json.loads(metadata)['box_reference_timestamp_us']
                                 for key, (_, metadata) in cache.index.items() if key[0] == sequence)
                    streams[sequence] = PersistentForestCacheStream(cache, trackers[sequence], scorer, origin_us=origin)
                deliveries = [CacheDelivery(**d) for d in event['deliveries']]
                side = {'single_vehicle': 'vehicle-side', 'single_remote': 'infrastructure-side'}.get(configuration['method'])
                if side:
                    deliveries = [d for d in deliveries if d.side == side]
                started = time.perf_counter()
                result = streams[sequence].step(deliveries, **{k: event[k] for k in ('frame_id', 'reference_us', 'decision_us', 'event_id')})
                timings.append(dict(event=index, seconds=time.perf_counter() - started))
                predictions.write(result.prediction_json + b'\n')
                audits.write(result.tracking_audit_json + b'\n')
                progress.update(index + 1)
        for sequence, tracker in trackers.items():
            databases[sequence] = dict(path=tracker.path.name, sha256=tracker.close())
        trackers.clear()
        if any(sha_file(source_root / name) != sha for name, sha in sources.items()):
            raise ValueError('source changed during replay')
        write_once(output / 'timings.json', timings)
        from .paper_resources import summarize
        write_once(output / 'resources.json', summarize(output, databases, timings))
        receipt = dict(
            kind='rbf_paper_replay_receipt_v1',
            status='software_replay_completed',
            completed_events=len(timings),
            completed_sequences=sorted(databases),
            databases=databases,
            fixture=fixture,
            full_dataset_verified=False,
            paper_results_verified=False,
            peak_rss_native_units=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            peak_rss_scope='process_lifetime_not_isolated_method',
            files={name: sha_file(output / name)
                   for name in ('plan.json', 'predictions.jsonl', 'audit.jsonl', 'timings.json', 'resources.json')})
        write_once(output / 'receipt.json', receipt)
        return receipt
    except BaseException as error:
        write_once(output / 'failure.json', dict(status='failed', completed_events=len(timings), error=str(error)))
        raise
    finally:
        for tracker in trackers.values():
            try:
                tracker.close()
            except Exception:
                pass
