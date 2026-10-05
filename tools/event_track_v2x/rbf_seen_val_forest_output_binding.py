"""Exact lineage and declared scope for complete exploratory seen-val output."""
import hashlib
import json
from pathlib import Path

from rbf_nested_seen_val_v2_common import R, sha
import publish_rbf_seen_val_forest_inputs as publication
import submit_rbf_seen_val_bound_forest as dispatch

DISPATCH = R/'source-freezes/rbf-seen-val-forest-publication-dispatch-v1-20261005'
DISPATCH_SHA = 'a2645f27e1bf516aa54552fd35e817901004bc836107f1f3dc29d0eeb0916651'
ORIGINAL_READER = R/'source-freezes/rbf-final-refit-main-topK-independent-output-reader-v1-20261004/read_rbf_final_refit_forest_outputs.py'
ORIGINAL_READER_SHA = '53747489a446c8990cfc6d9310eb686a69b39d99bde226c8945665a974d83f61'
INDEX = R/'receipts/rbf-matching-seen-val-three-seed-full-independent-GPU-NN-output-acceptance-20261004.json'
INDEX_SHA = '3e2d70039922ae44568ae302101b6fcd1a8092d88ddaf973cb15b89a74df1191'
KIND = 'rbf_seen_val_complete_bound_forest_independent_bytes_events_factors_v1'
canonical = publication.canonical


def source_gate():
    own = Path(__file__).resolve().parent
    freeze = json.loads((own/'source-freeze.json').read_bytes())
    for name, item in freeze['sources'].items():assert sha(own/name) == item['sha256']
    for item in freeze['references']:assert sha(item['path']) == item['sha256']
    assert sha(DISPATCH/'source-freeze.json') == DISPATCH_SHA
    prior = json.loads((DISPATCH/'source-freeze.json').read_bytes())
    for name, item in prior['sources'].items():assert sha(DISPATCH/name) == item['sha256']
    for module in (publication,dispatch):
        path = Path(module.__file__).resolve()
        assert path.parent == own and sha(path) == prior['sources'][path.name]['sha256']
    assert sha(ORIGINAL_READER) == ORIGINAL_READER_SHA and sha(INDEX) == INDEX_SHA
    publication.source_gate()
    return dict(reader_freeze_sha256=sha(own/'source-freeze.json'),input_dispatch_freeze_sha256=DISPATCH_SHA)


def arguments(parser):
    parser.add_argument('--seed',type=int,choices=(1337,2027,3407),required=True)
    for name in ('main-admission','main-byte-admission','publication'):
        parser.add_argument('--'+name,type=Path,required=True)


def qualify_job(args, Task, job, task):
    assert job['seed'] == args.seed == job['plan']['seed']
    for key in ('main_admission','main_byte_admission','publication'):
        assert job[key] == str(getattr(args,key))
    assert sha(args.publication) == job['publication_sha256']
    proof, main, _, _ = publication.prerequisites(args.main_admission,args.main_byte_admission,args.seed)
    records, binding = publication.local_inputs(args.seed,args.main_admission,proof,main)
    manifest = publication.manifest(args.seed,records,proof,main,Task)
    published = publication.publication(args.publication,manifest,Task)
    base = publication.expected_plan(main,binding,published,4)
    dispatch.verify_existing(task,job,base,dispatch.semantic(base))
    publication.remote_gate(job['plan'],args.publication,Task)
    return published


def artifact_keys(world):
    assert type(world) is int and world in (4,8)
    return {'receipt','exclusive-source-manifest','seen-val-input-binding'} | {f'replay-rank{i}' for i in range(world)}


def contained(directory, relative):
    directory, relative = Path(directory), Path(relative)
    assert not relative.is_absolute() and '..' not in relative.parts
    path = directory/relative
    assert path.is_file() and path.resolve().is_relative_to(directory.resolve())
    assert not any(p.is_symlink() for p in (path,*path.parents))
    return path


def verify_task_unchanged(task, job, inventory, status):
    task.reload()
    assert str(task.status) == status
    assert hashlib.sha256(task.data.script.diff.encode()).hexdigest() == job['plan']['bootstrap_sha256']
    assert json.loads(task.get_parameters()['General/plan']) == job['plan']
    assert task.get_parameters()['General/recipe_sha256'] == job['recipe_sha256']
    assert set(task.artifacts) == set(inventory)
    for key, item in inventory.items():
        assert task.artifacts[key].hash == item['sha256'] and task.artifacts[key].size == item['bytes']


def reference_inputs(seed, plan):
    """Reuse full admitted val NN outputs and the original complete schedule."""
    assert sha(INDEX) == INDEX_SHA
    index = json.loads(INDEX.read_bytes())
    assert index['kind'] == 'rbf_matching_seen_val_three_seed_all_rows_full_independent_GPU_NN_output_admission_index_v1'
    assert index['all_class_score_005_top64_preserved'] is True and index['GT_read'] is False
    assert index['atol'] == index['rtol'] == 1e-4 and index['total_rows'] == 345925
    entry = next(v for v in index['seeds'] if v['seed'] == seed)
    assert entry['status'] == 'completed' and entry['numeric_failed_rows'] == 0
    assert entry['original_events'] == 3316 and entry['sequences'] == 21
    paths = {k:Path(entry[k]) for k in ('output_byte_receipt','full_numeric_receipt')}
    for key,path in paths.items():assert sha(path) == entry[key+'_sha256']
    forward,numeric = (json.loads(paths[k].read_bytes()) for k in ('output_byte_receipt','full_numeric_receipt'))
    assert forward['kind'] == 'rbf_matching_seen_val_full_independent_output_bytes_and_row_coverage_v1'
    assert forward['all_cloud_output_bytes_independently_read'] is forward['all_original_seen_val_input_rows_exactly_once'] is True
    assert numeric['kind'] == 'rbf_matching_seen_val_full_independent_float64_joint_NN_numeric_v1'
    assert numeric['full_independent_numeric_pass'] is True and numeric['numeric_failed_rows'] == 0
    assert numeric['atol'] == numeric['rtol'] == 1e-4 and numeric['all_rows_without_GT_selection'] is True
    assert numeric['prediction_byte_proof_sha256'] == entry['output_byte_receipt_sha256']
    assert numeric['row_input_admission_sha256'] == forward['row_input_admission_sha256']
    assert forward['seed'] == numeric['seed'] == seed and forward['task_id'] == entry['task_id']
    assert forward['rows'] == numeric['rows'] == entry['rows']
    assert len(numeric['sequences']) == len(forward['sequences']) == 21
    assert {v['sequence_id']:v['rows'] for v in numeric['sequences']} == forward['sequences']
    assert all(v['numeric_failed_rows'] == 0 for v in numeric['sequences'])
    ck = R/f'artifacts/rbf-all-class-final-refit-independent-byte-freeze-v1-20261004/seed{seed}/checkpoint'
    assert sha(ck) == plan['checkpoint']['sha256']
    checkpoint = json.loads(ck.read_bytes())
    assert checkpoint['seed'] == seed and checkpoint['model_sha256'] == plan['final_refit_model_sha256']
    assert checkpoint['weights']['sha256'] == forward['weights_sha256'] == numeric['weights_sha256']
    expected_specs = [v for k,v in sorted(forward['artifacts'].items()) if k.startswith('predictions-rank')]
    assert plan['forward_outputs'] == expected_specs
    expected = {}
    for spec in expected_specs:
        path = paths['output_byte_receipt'].parent/spec['key']
        assert path.stat().st_size == spec['bytes'] and sha(path) == spec['sha256']
        with path.open('rb') as stream:
            for line in stream:
                row = json.loads(line); rows = expected.setdefault(row['sequence_id'],[])
                assert row['row'] == len(rows)
                rows.append(row)
    assert {sid:len(rows) for sid,rows in expected.items()} == forward['sequences']
    root = R/f'artifacts/rbf-seen-val-forest-input-bridge-v1-20261005/seed{seed}'
    event_path = root/'events.json'
    assert sha(event_path) == plan['events']['sha256']
    event_document = json.loads(event_path.read_bytes())
    assert event_document['kind'] == 'rbf_matching_seen_val_complete_forest_events_v1'
    assert event_document['seed'] == seed and event_document['split'] == 'val'
    assert event_document['measured_network_arrival_history_verified'] is False
    assert event_document['cache_manifest_sha256'] == plan['cache_manifest']['sha256']
    groups = {sid:[] for sid in event_document['origin_us_by_sequence']}
    for event in event_document['events']:groups[event['sequence_id']].append(event)
    assert len(groups) == 21 and sum(map(len,groups.values())) == 3316
    assert set(groups) == set(expected) and sum(map(len,expected.values())) == entry['rows']
    return entry,checkpoint,expected,event_document,groups


def verify_report(plan, report, binding, published):
    assert report['kind'] == 'rbf_final_refit_SPD_seen_val_bound_forest_candidate_v1'
    assert report['all_21_sequences_3316_events_completed'] is True
    assert binding['input_publication'] == published == plan['seen_val_input_publication']
    assert binding['seed'] == plan['seed'] and binding['full_train_interface_required'] is True
    for flag in ('validation_or_test_selection','measured_network_arrival_history_verified','learned_Stage2_complete',
                 'same_resource_performance_accepted','paper_performance_complete'):
        assert binding[flag] is False
    assert len(report['ranks']) == plan['world_size']
    assert sorted(v['rank'] for v in report['ranks']) == list(range(plan['world_size']))
    for rank in report['ranks']:
        assert rank['seed'] == plan['seed'] and rank['method'] == 'rbf'
        assert rank['world_size'] == plan['world_size'] and rank['all_sequences_completed'] is True
        assert rank['TF32_matmul'] is rank['TF32_cudnn'] is False
        assert rank['gpu_uuid'] not in ('unavailable','None',None,'')
        assert 'sm_'+''.join(map(str,rank['capability'])) in rank['native_architectures']
    assert len({v['gpu_uuid'] for v in report['ranks']}) == plan['world_size']


def verify_sequence(plan, per_plan, receipt, sequence, events):
    assert per_plan['kind'] == 'rbf_paper_replay_v1' and receipt['kind'] == 'rbf_paper_replay_receipt_v1'
    assert per_plan['expected_sequences'] == receipt['completed_sequences'] == [sequence]
    assert per_plan['expected_events'] == receipt['completed_events'] == len(events)
    assert per_plan['protocol'] == dict(dataset='spd',split='val',candidates='rbf-all-class-top64-v1',
        evaluation_class='car',maximum_detections=64,minimum_raw_score=.05)
    assert per_plan['configuration'] == plan['configuration'] and per_plan['fixture'] is False
    assert per_plan['model_binding']['checkpoint_sha256'] == plan['checkpoint']['sha256']
    assert per_plan['model_binding']['model_sha256'] == plan['final_refit_model_sha256']
    assert per_plan['events_sha256'] == hashlib.sha256(canonical(events)).hexdigest()
    assert per_plan['cache_sha256'] == plan['cache_manifest']['sha256']
    assert receipt['status'] == 'software_replay_completed'
    for name,checksum in plan['exclusive_patches'].items():assert per_plan['source_sha256'][name] == checksum
    for name,item in plan['source_replacements'].items():assert per_plan['source_sha256'][name] == item['sha256']
