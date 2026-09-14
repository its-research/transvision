#!/usr/bin/env python3
"""Hash-bound clean-link learned+CI and M0--M4 replay; GT is never an input.

The caller supplies a frozen pair-model manifest, not forest-identity weights. Two arrived source frames and the declared 100 ms deadline are required. This entry point does not
claim support for arbitrary delayed/retransmitted streams.
"""
import argparse
import hashlib
import json
import resource
import sys
import time
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np  # noqa: E402

from transvision.models.event_track_v2x.detection_cache_v2 import canonical, sha_file  # noqa: E402
from transvision.models.event_track_v2x.forest_cache_stream import CacheDelivery, VerifiedForestCache  # noqa: E402
from transvision.models.event_track_v2x.paper_native_cache import NativePaperCache  # noqa: E402
from transvision.models.event_track_v2x.paper_pair_baselines import METHODS, load_assets, make_tracker, snapshot  # noqa: E402
from transvision.models.event_track_v2x.paper_protocol import PaperProtocol  # noqa: E402
from transvision.models.event_track_v2x.tracking_v2 import TrackingConfigV2  # noqa: E402


def write(path, value):
    with Path(path).open('xb') as stream:
        stream.write(canonical(value))


def run(cache, events, output, *, protocol, configuration, model, calibration, binding, fixture=False):
    if type(fixture) is not bool or binding.get('fixture') != fixture or configuration['method'] not in METHODS:
        raise ValueError('explicit method and fixture provenance required')
    if set(configuration) != {'method', 'tracking'} or set(configuration['tracking']) != set(asdict(TrackingConfigV2())):
        raise ValueError('complete frozen pair configuration required')
    manifest = json.loads(cache.manifest_json)
    if manifest['split'] != protocol.split or manifest.get('fixture', False) and not fixture:
        raise ValueError('cache split/fixture differs')
    config = TrackingConfigV2(**configuration['tracking'])
    events = tuple(events)
    previous, pairs, origins = {}, {}, {}
    for event in events:
        if set(event) != {'sequence_id', 'frame_id', 'reference_us', 'decision_us', 'event_id', 'deliveries'}:
            raise ValueError('unexpected schedule field, including possible GT')
        seq, reference = event['sequence_id'], event['reference_us']
        if (any(not isinstance(event[k], str) or not event[k] for k in ('sequence_id', 'frame_id', 'event_id')) or type(reference) is not int
                or type(event['decision_us']) is not int or reference < 0 or reference <= previous.get(seq, -1) or event['decision_us'] != reference + 100_000):
            raise ValueError('strict reference ordering and fixed 100 ms clean-link deadline required')
        previous[seq] = reference
        origins.setdefault(seq, reference)
        deliveries = [CacheDelivery(**d) for d in event['deliveries']]
        if len(deliveries) != 2 or {d.side for d in deliveries} != {'vehicle-side', 'infrastructure-side'}:
            raise ValueError('paired clean-link control requires exactly two distinct sources')
        key = (seq, event['event_id'])
        if key in pairs:
            raise ValueError('duplicate event')
        ordered = sorted(deliveries, key=lambda d: d.side != 'vehicle-side')
        for delivery in ordered:
            _, metadata = cache.describe(delivery)
            if delivery.sequence_id != seq or not max(metadata['box_reference_timestamp_us'], metadata['source_image_timestamp_us']) <= delivery.arrival_us <= event['decision_us']:
                raise ValueError('source not arrived by paired-control deadline; no future payload read')
            if delivery.side == 'vehicle-side' and (metadata['frame_id'] != event['frame_id'] or metadata['box_reference_timestamp_us'] != reference):
                raise ValueError('paired control requires vehicle reference frame identity/time')
        pairs[key] = ordered
    if not events:
        raise ValueError('empty paired replay')
    # Validate the model/protocol/config before creating an output directory.
    make_tracker(configuration['method'], model, calibration, next(iter(origins)), next(iter(origins.values())), protocol=protocol, binding=binding, config=config)
    output = Path(output).absolute()
    if output.exists() or any(p.is_symlink() for p in (output, *output.parents)):
        raise ValueError('new output directory without symlink ancestors required')
    sources = {str(p.relative_to(ROOT)): sha_file(p) for p in (ROOT / 'transvision/models/event_track_v2x').glob('*.py')}
    sources[str(Path(__file__).relative_to(ROOT))] = sha_file(__file__)
    output.mkdir()
    plan = dict(
        kind='rbf_paper_pair_replay_v1',
        protocol=asdict(protocol),
        configuration=configuration,
        cache_sha256=cache.manifest_sha256,
        model_binding=binding,
        fixture=fixture,
        source_sha256=sources,
        events_sha256=hashlib.sha256(canonical(events)).hexdigest(),
        expected_events=len(events),
        expected_sequences=sorted(origins),
        clean_link_paired_only=True,
        gt_used_online=False)
    write(output / 'plan.json', plan)
    trackers, costs, timings, states = {}, {}, [], {}
    before_model = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    def counted_forward(module, inputs, result):
        costs['model_forward_calls'] = costs.get('model_forward_calls', 0) + 1
        costs['pair_logit_elements'] = costs.get('pair_logit_elements', 0) + result[0].numel()

    handle = model.register_forward_hook(counted_forward)
    files = ('predictions.jsonl', 'association.jsonl', 'diagnostic.jsonl', 'audit.jsonl')
    streams = {}
    try:
        streams = {name: (output / name).open('xb') for name in files}
        for event in events:
            started = time.perf_counter()
            seq = event['sequence_id']
            if seq not in trackers:
                trackers[seq] = make_tracker(configuration['method'], model, calibration, seq, origins[seq], protocol=protocol, binding=binding, config=config, counters=costs)
            frames = [cache.load_arrived(d, event['decision_us']) for d in pairs[(seq, event['event_id'])]]
            values = trackers[seq].step(*frames)
            prediction, association = values[:2]
            diagnostic = values[2] if len(values) == 3 else None
            streams['predictions.jsonl'].write(canonical(prediction) + b'\n')
            streams['association.jsonl'].write(canonical(association) + b'\n')
            streams['diagnostic.jsonl'].write(canonical(diagnostic) + b'\n')
            audit = dict(
                event_id=event['event_id'],
                sequence_id=seq,
                frame_id=event['frame_id'],
                raw_observation_sha256=prediction['source_cache_sha256'],
                selected_detections=prediction['selected_detections'],
                prediction_sha256=prediction['commit_sha256'],
                association_sha256=hashlib.sha256(canonical(association)).hexdigest(),
                costs_cumulative=dict(costs),
                fixture=fixture,
                method=configuration['method'])
            streams['audit.jsonl'].write(canonical(audit) + b'\n')
            timings.append(dict(event=len(timings), seconds=time.perf_counter() - started))
        for stream in streams.values():
            stream.close()
        for i, (seq, tracker) in enumerate(sorted(trackers.items())):
            path = output / f'state-{i:04d}.json'
            write(path, snapshot(tracker))
            states[seq] = dict(path=path.name, sha256=sha_file(path), bytes=path.stat().st_size)
        if (any(sha_file(ROOT / name) != sha for name, sha in sources.items()) or any(not np.array_equal(before_model[k].numpy(),
                                                                                                         v.detach().cpu().numpy()) for k, v in model.state_dict().items())):
            raise ValueError('pair implementation or model changed during replay')
        seconds = [t['seconds'] for t in timings]
        write(output / 'timings.json', timings)
        write(
            output / 'resources.json',
            dict(
                kind='rbf_pair_resource_vector_v1',
                costs=dict(
                    model_forward_calls=costs.get('model_forward_calls', 0),
                    pair_logit_elements=costs.get('pair_logit_elements', 0),
                    encoded_detections=costs.get('encoded_detections', 0),
                    cross_assignment_solves=costs.get('cross_assignment_solves', 0),
                    temporal_assignment_solves=costs.get('temporal_assignment_solves', 0),
                    assignment_solves=costs.get('cross_assignment_solves', 0) + costs.get('temporal_assignment_solves', 0)),
                state_snapshot_bytes=sum(s['bytes'] for s in states.values()),
                raw_cache_bytes=sum(p.stat().st_size for p in cache.root.rglob('*') if p.is_file()),
                output_stream_bytes=sum((output / name).stat().st_size for name in files),
                latency=dict(
                    events=len(seconds),
                    p50_seconds=float(np.quantile(seconds, .5)),
                    p95_seconds=float(np.quantile(seconds, .95)),
                    scope='load_encode_pair_fuse_track_and_stream_write'),
                equal_resources_verified=False,
                disk_bytes_are_resident_memory=False,
                raw_cache_bytes_scope='entire_bound_cache_not_only_selected_candidates'))
        receipt = dict(
            kind='rbf_paper_pair_replay_receipt_v1',
            status='software_replay_completed',
            completed_events=len(timings),
            completed_sequences=sorted(trackers),
            databases={},
            states=states,
            fixture=fixture,
            full_dataset_verified=False,
            paper_results_verified=False,
            peak_rss_native_units=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            peak_rss_scope='process_lifetime_not_isolated_method',
            files={name: sha_file(output / name)
                   for name in (*files, 'plan.json', 'resources.json', 'timings.json', *(s['path'] for s in states.values()))})
        write(output / 'receipt.json', receipt)
        return receipt
    except BaseException as error:
        write(output / 'failure.json', dict(status='failed', completed_events=len(timings), error=str(error)))
        raise
    finally:
        handle.remove()
        for stream in streams.values():
            stream.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for field in ('cache', 'cache-sha256', 'schedule', 'schedule-sha256', 'configuration', 'configuration-sha256', 'dataset', 'split', 'calibration', 'calibration-sha256',
                  'output'):
        parser.add_argument('--' + field, required=True)
    parser.add_argument('--checkpoint')
    parser.add_argument('--checkpoint-sha256')
    parser.add_argument('--fixture', action='store_true')
    args = parser.parse_args()
    protocol = PaperProtocol(args.dataset, args.split)
    for name in ('schedule', 'configuration'):
        if sha_file(getattr(args, name)) != getattr(args, name + '_sha256'):
            raise ValueError(name + ' changed')
    cache = (VerifiedForestCache if args.dataset == 'spd' else NativePaperCache)(args.cache, args.cache_sha256)
    configuration = json.loads(Path(args.configuration).read_bytes())
    events = [json.loads(line) for line in Path(args.schedule).read_bytes().splitlines() if line.strip()]
    model, calibration, binding = load_assets(
        args.checkpoint, args.checkpoint_sha256, args.calibration, args.calibration_sha256, cache=cache, protocol=protocol, fixture=args.fixture, method=configuration['method'])
    print(
        json.dumps(
            run(cache, events, args.output, protocol=protocol, configuration=configuration, model=model, calibration=calibration, binding=binding, fixture=args.fixture),
            sort_keys=True))


if __name__ == '__main__':
    main()
