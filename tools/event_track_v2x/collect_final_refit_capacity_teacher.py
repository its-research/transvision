"""Collect final-model native teacher bytes without changing original schedules.

The assemble function is derived from the preserved capacity collector. Its
resource-file loop uses original_path so that the original experiment plan
remains available for sequence two and later. Old frozen sources are unchanged.
"""
import argparse
import datetime
import hashlib
import importlib.util
import itertools
import json
import math
from pathlib import Path
import shutil
import sqlite3
import tarfile
import time

from rbf_nested_seen_val_v2_common import R, new, register, sha
from rbf_final_refit_teacher_binding import canonical, source_gate, validate_registered

SCHEMA = 'persistent_exclusive_completion_teacher_raw_probe_witness_v1'
RECIPE = 'exclusive_raw_search_before_after_probe_witness_v1'
INITIAL = 'exclusive_all_live_initial_search_and_catalog_v1'
KIND = 'rbf_final_refit_teacher_native_cohort_collection_v1'
BYTE_KIND = 'rbf_final_refit_teacher_registered_bytes_full_event_cohort_v1'
PARENT = R/'source-freezes/rbf-capacity-undecided-teacher-cohort-collector-v2-20261002/collector.py'
PARENT_SHA = '53225e66965c90b54be9451ef0db3e402f02021aebd7d9230f94ae92b299d2d3'
TRANSPORT = R/'source-freezes/rbf-final-refit-Top1-independent-output-reader-v1-20261004'


def write(path,value):
    with Path(path).open('xb') as stream: stream.write(canonical(value)+b'\n')


def contained(root,relative):
    root,relative=Path(root),Path(relative)
    assert not relative.is_absolute() and '..' not in relative.parts
    path=root/relative
    assert path.is_file() and path.resolve().is_relative_to(root.resolve())
    assert not any(p.is_symlink() for p in (path,*path.parents))
    return path


def assemble(events, natives, output, *, expected_events, expected_sequences, fixture,
             binding, progress=lambda value: None):
    """Pure registered-byte consumer; no producer or solver imports.

    Expected scope is explicit so finite source controls cannot acquire the
    hard-coded real 46-sequence/7445-event admission made by the entry point.
    """
    output = Path(output).absolute()
    assert not output.exists() and not any(p.is_symlink() for p in (output, *output.parents))
    assert type(fixture) is bool and len(events) == expected_events > 0
    sequences = sorted({event['sequence_id'] for event in events})
    assert sequences == expected_sequences == sorted(natives)
    assert len({(e['sequence_id'], e['event_id']) for e in events}) == expected_events
    for sequence in sequences:
        refs = [e['reference_us'] for e in events if e['sequence_id'] == sequence]
        assert all(type(v) is int and v >= 0 for v in refs)
        assert all(a < b for a, b in zip(refs, refs[1:])), 'original per-sequence chronology required'
    if not fixture:
        original = binding['original_plan']
        assert binding['GT_read'] is binding['test_read'] is False
        assert binding['seed'] in (1337, 2027, 3407)
        assert binding['original_event_asset'] == original['events']
        assert binding['checkpoint_sha256'] == original['checkpoint']['sha256']
        assert binding['expected_runtime_sources']
    groups = {s: [e for e in events if e['sequence_id'] == s] for s in sequences}
    output.mkdir(parents=True)
    plans, files, databases, native_bindings = [], {}, {}, []
    selected_events = {}
    resources = []
    labels = 0
    started = time.monotonic()
    try:
        for index, sequence in enumerate(sequences):
            native = Path(natives[sequence])
            receipt_path = contained(native, 'receipt.json')
            receipt = json.loads(receipt_path.read_bytes())
            plan_path = contained(native, 'plan.json')
            plan = json.loads(plan_path.read_bytes())
            assert receipt['kind'] == 'rbf_paper_replay_receipt_v1' and plan['kind'] == 'rbf_paper_replay_v1'
            assert receipt['status'] == 'software_replay_completed'
            assert receipt['fixture'] is plan['fixture'] is fixture
            assert receipt['completed_sequences'] == plan['expected_sequences'] == [sequence]
            assert receipt['completed_events'] == plan['expected_events'] == len(groups[sequence])
            assert hashlib.sha256(canonical(groups[sequence])).hexdigest() == plan['events_sha256']
            assert plan['protocol']['split'] == plan['model_binding']['fit_split'] == 'train'
            assert plan['protocol']['dataset'] == plan['model_binding']['dataset'] == 'spd'
            assert plan['configuration']['allocation'] == 'teacher'
            assert plan['configuration']['method'] == 'rbf'
            assert plan['configuration']['backend'] == 'exclusive_root_partition_regions_v1'
            assert plan['configuration']['state']['candidate_protocol'] == 'rbf-all-class-top64-v1'
            assert plan['configuration']['limits']['residual_partition_version'] == 1
            if not fixture:
                assert plan['model_binding']['seed'] == binding['seed']
                assert plan['model_binding']['checkpoint_sha256'] == binding['checkpoint_sha256']
                assert plan['model_binding']['model_sha256'] == binding['final_model_sha256']
                assert plan['configuration'] == original['configuration']
                assert plan['cache_sha256'] == original['cache_manifest']['sha256']
                assert plan['source_sha256'] == binding['expected_runtime_sources'], 'source/model/config binding must precede collection receipt'
            required = {'plan.json', 'audit.jsonl', 'predictions.jsonl', 'timings.json', 'resources.json'}
            assert required <= set(receipt['files'])
            for name, digest in receipt['files'].items():
                assert sha(contained(native, name)) == digest
            assert receipt['files']['plan.json'] == sha(plan_path)
            assert set(receipt['databases']) == {sequence}
            info = receipt['databases'][sequence]
            source = contained(native, info['path'])
            assert sha(source) == info['sha256']
            db = sqlite3.connect(source.resolve().as_uri() + '?mode=ro', uri=True)
            try:
                assert db.execute('PRAGMA integrity_check').fetchone()[0] == 'ok'
                meta = {k: json.loads(v) for k, v in db.execute('SELECT k,v FROM meta')}
                assert meta['schema'] == SCHEMA and meta['sequence_id'] == sequence
                assert meta['config'] == dict(state=plan['configuration']['state'], **plan['configuration']['limits'])
                rows = db.execute('SELECT ordinal,event_id,prediction,audit FROM events ORDER BY ordinal')
                heads = audit_head = '0' * 64
                sequence_timings = json.loads(contained(native, 'timings.json').read_bytes())
                assert len(sequence_timings) == len(groups[sequence])
                with contained(native, 'predictions.jsonl').open('rb') as preds, contained(native, 'audit.jsonl').open('rb') as audits:
                    count = 0
                    for event, row, pb, ab, timing in itertools.zip_longest(groups[sequence], rows, preds, audits, sequence_timings):
                        assert all(x is not None for x in (event, row, pb, ab, timing)), 'native event stream truncated or extended'
                        ordinal, event_id, sql_prediction, sql_audit = row
                        assert ordinal == count and event_id == event['event_id']
                        assert pb == sql_prediction + b'\n' and ab == sql_audit + b'\n'
                        prediction, audit = json.loads(pb), json.loads(ab)
                        assert prediction['sequence_id'] == audit['sequence_id'] == sequence
                        assert prediction['frame_id'] == event['frame_id'] and audit['event_id'] == event_id
                        assert prediction['decision_timestamp_us'] == event['decision_us']
                        assert prediction['box_reference_timestamp_us'] == event['reference_us']
                        assert prediction['previous_commit_sha256'] == heads and audit['previous_audit_sha256'] == audit_head
                        heads = hashlib.sha256(canonical({k: v for k, v in prediction.items() if k != 'commit_sha256'})).hexdigest()
                        assert heads == prediction['commit_sha256'] == audit['prediction_sha256']
                        audit_head = hashlib.sha256(canonical(audit)).hexdigest()
                        assert audit['kind'] == SCHEMA and audit['training_trace_only'] is True
                        assert audit['raw_probe_witness_recipe'] == RECIPE and audit['initial_search_witnesses']['recipe'] == INITIAL
                        assert audit['explicit_residual_partition'] is True and audit['residual_partition_version'] == 1
                        assert audit['configuration_sha256'] == hashlib.sha256(canonical(meta['config'])).hexdigest()
                        if not fixture:
                            ingestion = audit['cache_ingestion']
                            expected = sorted(event['deliveries'], key=lambda d: (d['arrival_us'], d['side'], d['frame_id']))
                            request = hashlib.sha256(canonical([event['frame_id'], event['reference_us'], event['decision_us'], event['deliveries']])).hexdigest()
                            assert ingestion['request_sha256'] == request and ingestion['new_deliveries'] == expected
                            assert ingestion['duplicate_deliveries'] == [] and ingestion['gt_model_inputs'] is False
                            assert ingestion['candidate_protocol'] == 'rbf-all-class-top64-v1'
                            assert ingestion['cache_manifest_sha256'] == plan['cache_sha256']
                        assert timing['event'] == count and math.isfinite(timing['seconds']) and timing['seconds'] >= 0
                        # Keep only row references, never the entire cohort's
                        # potentially large raw probe/audit bytes in RAM.
                        selected_events[sequence, event_id] = (index, count,
                            dict(timing, sequence_id=sequence, native_event=count))
                        labels += sum(len(x['allocation_training']['candidates']) for x in audit['allocation_trace'])
                        count += 1
                    assert count == len(groups[sequence]) == meta['state']['events']
                assert heads == meta['state']['prediction_sha256'] and audit_head == meta['state']['audit_sha256']
            finally:
                db.close()
            name = f'sequence-{index:02d}.sqlite'
            with source.open('rb') as src, (output / name).open('xb') as dest:
                shutil.copyfileobj(src, dest, 8 * 1024**2)
            assert sha(output / name) == info['sha256']
            files[name] = info['sha256']
            databases[sequence] = dict(path=name, sha256=info['sha256'])
            for label, original_path in [('native-receipt', receipt_path), ('native-resources', contained(native, 'resources.json'))]:
                name = f'{label}-{index:02d}.json'
                with original_path.open('rb') as src, (output / name).open('xb') as dest:
                    shutil.copyfileobj(src, dest)
                files[name] = sha(output / name)
            native_bindings.append(dict(sequence_id=sequence, receipt_sha256=sha(receipt_path),
                                        database_sha256=info['sha256'], original_files=receipt['files']))
            resources.append(dict(sequence_id=sequence, original_resource_file=f'native-resources-{index:02d}.json',
                                  peak_rss_native_units=receipt['peak_rss_native_units'], peak_rss_scope=receipt['peak_rss_scope']))
            common = {k: v for k, v in plan.items() if k not in ('events_sha256', 'expected_events', 'expected_sequences')}
            assert not plans or common == plans[0], 'native sequences differ in source/model/config/cache/protocol'
            plans.append(common)
            progress(dict(stage='teacher_native_cohort_collection', completed_sequences=index + 1, total_sequences=len(sequences),
                          ETA_seconds=(time.monotonic() - started) * (len(sequences) - index - 1) / (index + 1),
                          ETA_scope='byte/commit collection only; independent numerics excluded'))
        assert len(selected_events) == expected_events
        plan = dict(plans[0], expected_events=expected_events, expected_sequences=sequences,
                    events_sha256=hashlib.sha256(canonical(events)).hexdigest())
        write(output / 'plan.json', plan)
        write(output / 'events.json', events)
        timings = []
        with (output / 'predictions.jsonl').open('xb') as preds, (output / 'audit.jsonl').open('xb') as audits:
            reader, current = None, None
            try:
                for ordinal, event in enumerate(events):
                    index, native_ordinal, timing = selected_events[event['sequence_id'], event['event_id']]
                    if index != current:
                        if reader is not None:
                            reader.close()
                        reader = sqlite3.connect((output / f'sequence-{index:02d}.sqlite').as_uri() + '?mode=ro', uri=True)
                        current = index
                    rows = reader.execute('SELECT event_id,prediction,audit FROM events WHERE ordinal=?', (native_ordinal,)).fetchall()
                    assert len(rows) == 1 and rows[0][0] == event['event_id']
                    preds.write(rows[0][1] + b'\n')
                    audits.write(rows[0][2] + b'\n')
                    timings.append(dict(timing, event=ordinal))
            finally:
                if reader is not None:
                    reader.close()
        write(output / 'timings.json', timings)
        write(output / 'resources.json', dict(kind='rbf_teacher_native_resource_records_without_isolated_cost_admission',
              native_sequences=resources, process_lifetime_peaks_are_not_summed=True,
              same_resource_performance_accepted=False, total_teacher_resource_cost_admitted=False))
        write(output / 'collection-binding.json', dict(binding, native_sequences=native_bindings,
              original_native_receipts_preserved=True, collector_sha256=sha(__file__),
              new_events_inferred_or_completed=False, labels=labels, fixture=fixture))
        for name in ('plan.json', 'events.json', 'predictions.jsonl', 'audit.jsonl', 'timings.json', 'resources.json', 'collection-binding.json'):
            files[name] = sha(output / name)
        receipt = dict(kind='rbf_paper_replay_receipt_v1', status='software_replay_completed',
                       completed_events=expected_events, completed_sequences=sequences, databases=databases,
                       files=files, fixture=fixture, full_dataset_verified=False, paper_results_verified=False,
                       collection_kind=KIND, actual_dataset_and_full_causal_trajectory_accepted=False,
                       full_real_teacher_target_admission=False, total_teacher_resource_cost_admitted=False)
        write(output / 'receipt.json', receipt)
        return dict(kind=KIND, collection_receipt_sha256=sha(output / 'receipt.json'), labels=labels,
                    completed_events=expected_events, completed_sequences=sequences,
                    full_real_teacher_target_admission=False, total_teacher_resource_cost_admitted=False)
    except BaseException as error:
        write(output / 'collection-failure.json', dict(kind='rbf_teacher_collection_failure', type=type(error).__name__,
              message=str(error), experiment_accepted=False, native_outputs_modified=False))
        raise


def load(path, checksum, name):
    assert sha(path) == checksum
    spec = importlib.util.spec_from_file_location(name,path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def artifact_keys(world):
    assert type(world) is int and world in (4,8)
    return {'receipt','exclusive-source-manifest',*(f'replay-rank{i}' for i in range(world))}


def runtime_sources(plan):
    """Reconstruct source identity from frozen archive and exact replacements."""
    source = R/'artifacts/rbf-original-joint-coupled-forest-GPU4-v3-source-20261001/independent-cloud-readback/source.bytes'
    assert sha(source) == plan['source']['sha256'] and source.stat().st_size == plan['source']['bytes']
    with tarfile.open(source) as archive:
        expected = {m.name:hashlib.sha256(archive.extractfile(m).read()).hexdigest()
            for m in archive.getmembers() if m.isfile() and (
                (m.name.startswith('transvision/models/event_track_v2x/')
                 and len(Path(m.name).parts)==4 and m.name.endswith('.py'))
                or m.name=='tools/event_track_v2x/persistent_mht_tracking.py')}
    old = load(PARENT,PARENT_SHA,'teacher_exact_core_replacement_bindings')
    expected = old.apply_core_replacements(expected,plan)
    expected.update(plan['exclusive_patches'])
    return expected


def validate_rank_report(report, task_id, plan, sequences, groups):
    assert report['kind'] == 'rbf_final_refit_capacity_witness_full_train_teacher_candidate_v1'
    actual = dict(report['plan']); actual.pop('cache_relative_root',None)
    assert actual == plan and report['task_id'] == task_id
    assert report['failure'] is None and report['all_46_sequences_7445_events_completed'] is True
    world = plan['world_size']
    assert sorted(r['rank'] for r in report['ranks']) == list(range(world))
    assert len({r['gpu_uuid'] for r in report['ranks']}) == world
    for rank in report['ranks']:
        assert rank['seed'] == plan['seed'] and rank['method'] == 'rbf'
        assert rank['world_size'] == world and rank['all_sequences_completed'] is True
        assert rank['TF32_matmul'] is rank['TF32_cudnn'] is False
        expected = sequences[rank['rank']::world]
        assert [r['sequence_id'] for r in rank['sequences']] == expected
        assert rank['events_committed'] == sum(len(groups[s]) for s in expected)
        for item in rank['sequences']:
            assert item['completed'] is True and item['failure'] is None
            assert item['committed_events'] == item['expected_events'] == len(groups[item['sequence_id']])


def verify_factors(database, expected, origin):
    """Independently bind stored factors to already accepted final-model logits."""
    connection = sqlite3.connect(database.resolve().as_uri()+'?mode=ro',uri=True)
    count = 0; maximum = 0.
    try:
        for i,node,raw,digest in connection.execute('SELECT i,node_id,raw,sha FROM observations ORDER BY i'):
            assert i == count and hashlib.sha256(raw).hexdigest() == digest
            assert i < len(expected)
            reference = expected[i]; observation = json.loads(raw)
            assert reference['row'] == i and reference['node_id'] == node
            assert observation['features'][141] == (observation['state_us']-origin)/1e8
            values = [float(v) for v in reference['logits']]
            assert values and all(math.isfinite(v) for v in values)
            largest = max(values); denominator = largest+math.log(sum(math.exp(v-largest) for v in values))
            expected_values = [v-denominator for v in values]
            potentials = connection.execute('SELECT p,w FROM potentials WHERE i=? ORDER BY p',(i,)).fetchall()
            assert [p for p,w in potentials] == [-1]+reference['context_indices'][:-1]
            assert len(potentials) == len(values)
            for (_,actual),value in zip(potentials,expected_values,strict=True):
                assert math.isfinite(actual) and abs(actual-value) <= 1e-4+1e-4*abs(value)
                maximum = max(maximum,abs(actual-value))
            count += 1
        assert count == len(expected)
        return dict(rows=count,max_abs_error=maximum,atol=1e-4,rtol=1e-4)
    finally: connection.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed',type=int,choices=(1337,2027,3407),required=True)
    for name in ('main-admission','main-byte-admission','published-prerequisite','job-journal','readback','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    args = parser.parse_args()
    own = Path(__file__).resolve().parent
    frozen = json.loads((own/'source-freeze.json').read_bytes())
    for name,record in frozen['sources'].items(): assert sha(own/name) == record['sha256']
    for record in frozen['references']: assert sha(record['path']) == record['sha256']
    assert args.readback.resolve().is_relative_to(R/'artifacts') and args.output.resolve().is_relative_to(R/'artifacts')
    assert not args.readback.exists() and not args.output.exists(), 'preserve any previous collection or active reader'
    jobs = [j for j in json.loads(args.job_journal.read_bytes())['jobs'] if j['seed'] == args.seed]
    assert len(jobs) == 1
    job = jobs[0]; plan = job['plan']
    task,proof = validate_registered(job,args.main_admission,args.main_byte_admission,args.published_prerequisite)
    assert set(task.artifacts) == artifact_keys(plan['world_size'])
    args.readback.mkdir(parents=True,exist_ok=False)
    start = args.readback/'read-started.json'
    new(start,dict(task_id=task.id,recipe_sha256=job['recipe_sha256'],source_sha256=sha(__file__),
        job_journal_sha256=sha(args.job_journal),main_prerequisite=proof,ETA='unknown; transfer progress will follow'))
    register(start,'rbf-final-refit-teacher-read-started')
    try:
        helper = TRANSPORT/'read_rbf_final_refit_top1_outputs.py'
        pinned = next(x['sha256'] for x in frozen['references'] if x['path'] == str(helper))
        reader = load(helper,pinned,'teacher_exact_cloud_byte_reader')
        inventory = {}
        for key in sorted(task.artifacts):
            path = args.readback/(key+('.tar.gz' if key.startswith('replay-rank') else '.json'))
            inventory[key] = reader.read_artifact(task,key,path)
        report = json.loads((args.readback/'receipt.json').read_bytes())
        event_path = R/f'artifacts/rbf-original-cache-CPU-metadata-export-v1-20261001/seed{args.seed}/events.json'
        assert sha(event_path) == plan['events']['sha256'] and event_path.stat().st_size == plan['events']['bytes']
        envelope = json.loads(event_path.read_bytes()); events = envelope['events']
        sequences = sorted(envelope['origin_us_by_sequence'])
        assert len(events) == 7445 and len(sequences) == 46
        groups = {s:[e for e in events if e['sequence_id']==s] for s in sequences}
        validate_rank_report(report,task.id,plan,sequences,groups)
        manifest = json.loads((args.readback/'exclusive-source-manifest.json').read_bytes())
        for a,b in (('patches','exclusive_patches'),('configuration','configuration'),('original_source','source'),
                    ('bootstrap_sha256','bootstrap_sha256'),('source_replacements','source_replacements'),
                    ('CPU_capacity_candidate_admission','CPU_capacity_candidate_admission')):
            assert manifest[a] == plan[b]
        expected_sources = runtime_sources(plan)
        index_path = R/'receipts/rbf-final-refit-three-seed-all-row-full-independent-numeric-acceptance-20261004.json'
        assert sha(index_path) == plan['numeric_reference_admission_sha256']
        entry = next(v for v in json.loads(index_path.read_bytes())['seeds'] if v['seed']==args.seed)
        forward_path = Path(entry['prediction_byte_proof'])
        assert sha(forward_path) == entry['prediction_byte_proof_sha256']
        forward = json.loads(forward_path.read_bytes()); expected = {}
        for spec in plan['forward_outputs']:
            assert spec == forward['artifacts'][spec['key']]
            path = forward_path.parent/(spec['key']+'.jsonl'); assert sha(path) == spec['sha256']
            for line in path.open('rb'):
                row = json.loads(line); values = expected.setdefault(row['sequence_id'],[])
                assert row['row'] == len(values); values.append(row)
        assert set(expected) == set(sequences) and sum(map(len,expected.values())) == entry['rows']
        natives = {}; factors = {}
        for rank in range(plan['world_size']):
            dest = args.readback/f'rank{rank}-unpack'
            reader.unpack(args.readback/f'replay-rank{rank}.tar.gz',dest)
            for sequence in sequences[rank::plan['world_size']]:
                assert sequence and '/' not in sequence and '\\' not in sequence and sequence not in ('.','..')
                native = dest/f'rank-{rank}'/sequence; natives[sequence] = native
                receipt = json.loads(contained(native,'receipt.json').read_bytes())
                db = receipt['databases'][sequence]; database = contained(native,db['path'])
                assert sha(database) == db['sha256']
                factors[sequence] = verify_factors(database,expected[sequence],envelope['origin_us_by_sequence'][sequence])
                print(json.dumps(dict(stage='final_teacher_independent_final_model_factors',completed_sequences=len(factors),
                    total_sequences=46,ETA='unknown; heterogeneous SQL histories')),flush=True)
        binding = dict(task_id=task.id,seed=args.seed,recipe_sha256=job['recipe_sha256'],registered_artifacts=inventory,
            original_event_asset=plan['events'],original_event_asset_sha256=sha(event_path),original_plan=plan,
            main_task_id=proof['main_task_id'],main_replay_admission_sha256=sha(args.main_admission),
            main_byte_admission_sha256=sha(args.main_byte_admission),published_prerequisite_sha256=sha(args.published_prerequisite),
            job_journal_sha256=sha(args.job_journal),checkpoint_sha256=plan['checkpoint']['sha256'],
            final_model_sha256=plan['final_refit_model_sha256'],expected_runtime_sources=expected_sources,GT_read=False,test_read=False)
        result = assemble(events,natives,args.output,expected_events=7445,expected_sequences=sequences,fixture=False,
            binding=binding,progress=lambda value:print(json.dumps(value),flush=True))
        path = args.output/'independent-byte-cohort-collection.json'
        new(path,dict(result,kind=BYTE_KIND,task_id=task.id,seed=args.seed,recipe_sha256=job['recipe_sha256'],
            world_size=plan['world_size'],registered_artifacts=inventory,all_registered_bytes_verified=True,
            all_7445_events_and_46_native_sequences_byte_bound=True,original_events_inferred_or_completed=False,
            full_teacher_factor_state_action_target_admission=False,final_model_sha256=plan['final_refit_model_sha256'],
            final_checkpoint_sha256=plan['checkpoint']['sha256'],main_replay_admission_sha256=sha(args.main_admission),
            original_event_asset_sha256=sha(event_path),collector_sha256=sha(__file__),
            all_final_model_factor_rows_independently_bound=True,final_model_factor_reports=factors,
            max_factor_difference=max(v['max_abs_error'] for v in factors.values()),NN_atol=1e-4,NN_rtol=1e-4,
            full_Stage2_complete=False,paper_performance_complete=False))
        register(path,'rbf-final-refit-teacher-full-native-byte-factor-cohort')
        print(json.dumps(dict(receipt=str(path))),flush=True)
    except BaseException as error:
        failure = args.readback/'failure.json'
        new(failure,dict(task_id=task.id,exception_type=type(error).__name__,message=str(error),accepted=False,
            partials_preserved=True,automatic_producer_retry=False))
        register(failure,'rbf-final-refit-teacher-cohort-failure'); raise


if __name__ == '__main__': main()
