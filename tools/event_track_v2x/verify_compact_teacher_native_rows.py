"""Native commit-chain checks over losslessly stored teacher members.

Derived from the frozen final-refit collector's per-sequence read checks.
No cohort files or original-identity SQLite copies are manufactured. Storage
readback, model provenance and the independent numerical oracles are separate.
"""
import hashlib
import itertools
import json
import math

SCHEMA = 'persistent_exclusive_completion_teacher_raw_probe_witness_v1'
RECIPE = 'exclusive_raw_search_before_after_probe_witness_v1'
INITIAL = 'exclusive_all_live_initial_search_and_catalog_v1'


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def require(value, message):
    if not value:
        raise ValueError(message)


def verify_native_rows(db, plan, receipt, events, open_member):
    """Read original decoded members and SQLite rows in lockstep, without writes.

    open_member(name) must return a context manager for that original member's
    decoded byte stream. Callers must first verify its registered storage chain.
    """
    sid = plan['expected_sequences'][0]
    require(plan['expected_sequences'] == receipt['completed_sequences'] == [sid], 'sequence coverage differs')
    require(len(events) == plan['expected_events'] == receipt['completed_events'] > 0, 'event count differs')
    require(all(e['sequence_id'] == sid for e in events), 'mixed original sequences')
    require(len({e['event_id'] for e in events}) == len(events), 'duplicate original event')
    refs = [e['reference_us'] for e in events]
    require(all(type(x) is int and x >= 0 for x in refs) and
            all(a < b for a,b in zip(refs,refs[1:])), 'original chronology differs')
    require(hashlib.sha256(canonical(events)).hexdigest() == plan['events_sha256'], 'original events SHA differs')
    require(db.execute('PRAGMA integrity_check').fetchone()[0] == 'ok', 'database integrity failure')
    meta = {k:json.loads(v) for k,v in db.execute('SELECT k,v FROM meta')}
    require(meta['schema'] == SCHEMA and meta['sequence_id'] == sid, 'native database metadata differs')
    config = dict(state=plan['configuration']['state'], **plan['configuration']['limits'])
    require(meta['config'] == config, 'native configuration differs')
    with open_member('timings.json') as stream:
        raw = stream.read()
    require(hashlib.sha256(raw).hexdigest() == receipt['files']['timings.json'], 'timings member SHA differs')
    timings = json.loads(raw)
    require(len(timings) == len(events), 'timing count differs')
    rows = db.execute('SELECT ordinal,event_id,prediction,audit FROM events ORDER BY ordinal')
    head = audit_head = '0'*64
    pred_hash = hashlib.sha256(); audit_hash = hashlib.sha256()
    count = labels = 0
    with open_member('predictions.jsonl') as preds, open_member('audit.jsonl') as audits:
        for event,row,pb,ab,timing in itertools.zip_longest(events,rows,preds,audits,timings):
            require(all(x is not None for x in (event,row,pb,ab,timing)), 'native event stream truncated or extended')
            ordinal,event_id,sql_prediction,sql_audit = row
            require(ordinal == count and event_id == event['event_id'], 'native event order differs')
            require(pb == sql_prediction+b'\n' and ab == sql_audit+b'\n', 'native JSONL differs from SQLite')
            pred_hash.update(pb); audit_hash.update(ab)
            prediction,audit = json.loads(pb),json.loads(ab)
            require(prediction['sequence_id'] == audit['sequence_id'] == sid, 'native sequence differs')
            require(prediction['frame_id'] == event['frame_id'] and audit['event_id'] == event_id, 'native frame differs')
            require(prediction['decision_timestamp_us'] == event['decision_us'] and
                    prediction['box_reference_timestamp_us'] == event['reference_us'], 'native timestamp differs')
            require(prediction['previous_commit_sha256'] == head and
                    audit['previous_audit_sha256'] == audit_head, 'native previous commit chain differs')
            head = hashlib.sha256(canonical({k:v for k,v in prediction.items() if k!='commit_sha256'})).hexdigest()
            require(head == prediction['commit_sha256'] == audit['prediction_sha256'], 'native prediction chain differs')
            audit_head = hashlib.sha256(canonical(audit)).hexdigest()
            require(audit['kind'] == SCHEMA and audit['training_trace_only'] is True, 'native audit kind differs')
            require(audit['raw_probe_witness_recipe'] == RECIPE and
                    audit['initial_search_witnesses']['recipe'] == INITIAL, 'original witness recipe differs')
            require(audit['explicit_residual_partition'] is True and audit['residual_partition_version'] == 1,
                    'residual partition differs')
            require(audit['configuration_sha256'] == hashlib.sha256(canonical(config)).hexdigest(), 'audit config SHA differs')
            ingestion = audit['cache_ingestion']
            deliveries = sorted(event['deliveries'], key=lambda d:(d['arrival_us'],d['side'],d['frame_id']))
            request = hashlib.sha256(canonical([event['frame_id'],event['reference_us'],event['decision_us'],event['deliveries']])).hexdigest()
            require(ingestion['request_sha256'] == request and ingestion['new_deliveries'] == deliveries,
                    'original causal deliveries differ')
            require(ingestion['duplicate_deliveries'] == [] and ingestion['gt_model_inputs'] is False,
                    'duplicate or GT model input differs')
            require(ingestion['candidate_protocol'] == 'rbf-all-class-top64-v1' and
                    ingestion['cache_manifest_sha256'] == plan['cache_sha256'], 'original cache binding differs')
            require(timing['event'] == count and math.isfinite(timing['seconds']) and timing['seconds'] >= 0,
                    'native event timing differs')
            labels += sum(len(x['allocation_training']['candidates']) for x in audit['allocation_trace'])
            count += 1
    require(count == len(events) == meta['state']['events'], 'native state count differs')
    require(head == meta['state']['prediction_sha256'] and audit_head == meta['state']['audit_sha256'], 'final native chain differs')
    require(pred_hash.hexdigest() == receipt['files']['predictions.jsonl'] and
            audit_hash.hexdigest() == receipt['files']['audit.jsonl'], 'original JSONL member SHA differs')
    return {'kind':'compact_teacher_native_commit_readback_v1','sequence_id':sid,'events':count,'labels':labels,
            'prediction_head':head,'audit_head':audit_head,'all_original_streams_and_commit_chains_verified':True,
            'numerical_teacher_targets_accepted':False,'full_teacher_target_acceptance':False,'scientific_acceptance':False}
