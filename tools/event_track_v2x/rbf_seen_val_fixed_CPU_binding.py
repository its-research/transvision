"""Complete seen-val fixed-baseline CPU admission, with no inherited train pass."""
import hashlib
import importlib
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

from rbf_nested_seen_val_v2_common import R,sha

READER=R/'source-freezes/rbf-seen-val-fixed-K1-K4-independent-output-reader-v1-20261005'
READER_SHA='d5751810ffb9d8e2507b1361b3d63fb7bfdac4366a9c941dcb620348d5c69737'
VAL_CPU=R/'source-freezes/rbf-seen-val-bound-forest-full-independent-CPU-v1-20261005'
VAL_CPU_SHA='6ed98ff9f34599f8d3e4190470dc5febb8bcc9d8f6c31e06158051b4a30dd0ce'
MODEL_CPU=R/'source-freezes/rbf-final-refit-full-forest-independent-CPU-v4-receipt-rows-20261004'
MODEL_CPU_SHA='f36f644f5f18fad3e5ab489edbc949e93bcab4efce22749dcb90211c8c8b27f7'
ORIGINALS={
    1:(R/'source-freezes/rbf-final-refit-fixed-Top1-full-independent-CPU-v2-receipt-rows-20261004','6ebb95230c82d5564000e911f3c4c45939783954f5937b27e4d553d03a171e35'),
    4:(R/'source-freezes/rbf-final-refit-fixed-topK-full-independent-CPU-v2-receipt-rows-20261004','7dbc44e80ac6b13c232a59004fb3baaaa67ebde9b6e95c4890ce57a74f2f47f4')}


def scope(width):
    assert width in (1,4)
    return f'SPD seen-val exploratory scheduled snapshots; fixed K{width} baseline'


def frozen_modules(width):
    original,digest=ORIGINALS[width]
    for directory,checksum in ((original,digest),(READER,READER_SHA),(MODEL_CPU,MODEL_CPU_SHA)):
        assert sha(directory/'source-freeze.json')==checksum
        freeze=json.loads((directory/'source-freeze.json').read_bytes())
        for name,item in freeze['sources'].items():assert sha(directory/name)==(item['sha256'] if isinstance(item,dict) else item)
    modules=[]
    for directory,name in ((MODEL_CPU,'rbf_final_refit_forest_binding'),(READER,'rbf_seen_val_fixed_output_binding'),
                           (READER,'read_rbf_seen_val_fixed_outputs')):
        sys.path.insert(0,str(directory));module=importlib.import_module(name)
        assert Path(module.__file__).resolve()==directory/(name+'.py')
        modules.append(module)
    name=f'unchanged_fixed_K{width}_cache203_receipt_oracle'
    path=original/'final_cache203_receipt_v2.py'
    if name not in sys.modules:
        spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module);sys.modules[name]=module
    features=sys.modules[name];assert Path(features.__file__).resolve()==path
    modules[0].source_gate();modules[1].source_gate()
    return modules[0],features,modules[1],modules[2]


def source_gate(width):
    directory=Path(__file__).resolve().parent
    freeze=json.loads((directory/'source-freeze.json').read_bytes())
    assert freeze['kind']=='rbf_seen_val_fixed_K1_K4_full_CPU_source_v1'
    for name,item in freeze['sources'].items():assert sha(directory/name)==item['sha256']
    for item in freeze['references']:assert sha(item['path'])==item['sha256']
    frozen_modules(width)
    return dict(seen_val_CPU_source_freeze_sha256=sha(directory/'source-freeze.json'),K=width,
        original_baseline_CPU_source_freeze_sha256=ORIGINALS[width][1],seen_val_byte_reader_source_freeze_sha256=READER_SHA,
        scope=scope(width),measured_network_arrival_history_verified=False,prior_train_or_other_seed_forest_acceptance_inherited=False)


def registered(path):
    path=Path(path);assert path.resolve().is_relative_to(R)
    assert not any(p.is_symlink() for p in (path,*path.parents))
    digest=sha(path);entries=json.loads((R/'receipts/20260928-execution-ledger.json').read_bytes())['entries']
    assert any(e.get('receipt')==str(path) and e.get('receipt_sha256')==digest for e in entries)
    return json.loads(path.read_bytes())


def validate_local(value,job,output,width):
    plan=job['plan'];assert width in (1,4) and value['K']==job['K']==plan['baseline_K']==width
    assert value['kind']==output.KIND and value['seed']==job['seed']==plan['seed'] in (1337,2027,3407)
    assert value['task_id']==job['task_id']
    assert value['recipe_sha256']==job['recipe_sha256']==hashlib.sha256(output.canonical(plan)).hexdigest()
    assert value['method']==plan['method']==plan['configuration']['method']=='topk'
    assert plan['configuration']['allocation']=='bound' and plan['configuration']['history_features'] is True
    assert plan['configuration']['state']['active_limit']==width and plan['configuration']['state']['decision_mode']=='retained'
    assert value['world_size']==plan['world_size'] in (4,8)
    assert value['NN_atol']==value['NN_rtol']==plan['scoring_atol']==plan['scoring_rtol']==1e-4
    assert value['all_registered_bytes_verified'] is value['all_21_sequences_3316_events_and_final_model_factor_nodes_verified'] is True
    for flag in ('measured_network_arrival_history_verified','full_baseline_semantics_or_fresh_state_independently_accepted',
                 'main_or_train_full_forest_acceptance_inherited','learned_Stage2_complete','same_resource_performance_accepted','paper_performance_complete'):
        assert value[flag] is False
    assert value['scope']==plan['scope']==plan['evaluation_scope']==scope(width)
    assert value['source_package_sha256']==output.SOURCE_SHA==plan['source']['sha256']
    assert value['input_numeric_index_sha256']==output.shared.INDEX_SHA
    assert value['source_sha256']==sha(READER/'read_rbf_seen_val_fixed_outputs.py')
    assert value['publication_sha256']==job['input_receipt_sha256']['publication']
    assert value['final_checkpoint_sha256']==plan['checkpoint']['sha256']
    assert value['final_model_sha256']==plan['final_refit_model_sha256']
    assert set(value['artifacts'])==output.artifact_keys(plan['world_size'])
    sequences=value['sequences'];assert len(sequences)==len({s['sequence_id'] for s in sequences})==21
    assert sum(s['events'] for s in sequences)==3316 and sum(s['nodes'] for s in sequences)==value['total_nodes']
    for item in sequences:
        sid=item['sequence_id'];assert isinstance(sid,str) and sid and '/' not in sid and '\\' not in sid and sid not in ('.','..')
        assert type(item['events']) is int and item['events']>0 and type(item['nodes']) is int and item['nodes']>=0
        digest=item['database_sha256'];assert len(digest)==64 and all(c in '0123456789abcdef' for c in digest)
    return plan


def model_binding(seed,checkpoint,job,model_module,output):
    entry,ck,_,_,_=output.reference_inputs(seed,job['plan'])
    assert sha(checkpoint)==job['plan']['checkpoint']['sha256'] and json.loads(Path(checkpoint).read_bytes())==ck
    train=model_module.validate_final_model(seed,checkpoint)
    return dict(final_checkpoint_sha256=sha(checkpoint),final_model_sha256=ck['model_sha256'],inherited_train_NN_admission=train,
        seen_val_NN_numeric_admission_sha256=entry['full_numeric_receipt_sha256'],
        seen_val_NN_byte_admission_sha256=entry['output_byte_receipt_sha256'],
        old_model_full_forest_acceptance_inherited=False,full_forest_accepted=False)


def validate_output(path,*,width,Task=None):
    binding=source_gate(width);model,_,output,reader=frozen_modules(width)
    value=registered(path)
    jobs=json.loads(output.dispatch.JOURNAL.read_bytes())['jobs']
    matches=[j for j in jobs if j['K']==width and j['seed']==value['seed']];assert len(matches)==1
    job=matches[0];plan=validate_local(value,job,output,width)
    if Task is None:
        from clearml import Task
    task=Task.get_task(task_id=job['task_id']);assert str(task.status)=='completed'
    args=SimpleNamespace(K=width,seed=job['seed'],**{k:Path(v) for k,v in job['input_paths'].items()})
    published=output.qualify_job(args,Task,job,task)
    output.verify_task_unchanged(task,job,value['artifacts'],'completed')
    root=Path(path).parent
    for key,item in value['artifacts'].items():
        local=reader.local_artifact(root,key);assert local.stat().st_size==item['bytes'] and sha(local)==item['sha256']
    report=json.loads((root/'receipt.json').read_bytes())
    output.verify_report(plan,report,json.loads((root/'seen-val-input-binding.json').read_bytes()),
                         json.loads((root/'seen-val-baseline-binding.json').read_bytes()),published)
    entry,_,expected,_,groups=output.reference_inputs(job['seed'],plan)
    assert value['total_nodes']==entry['rows']
    assert {s['sequence_id']:s['events'] for s in value['sequences']}=={s:len(e) for s,e in groups.items()}
    assert {s['sequence_id']:s['nodes'] for s in value['sequences']}=={s:len(e) for s,e in expected.items()}
    sources=output.source_map(plan)
    for seq in value['sequences']:
        sid=seq['sequence_id'];found=list(root.glob(f'rank*-unpack/rank-*/{sid}/receipt.json'));assert len(found)==1
        directory=found[0].parent
        receipt=json.loads(output.contained(directory,'receipt.json').read_bytes())
        per_plan=json.loads(output.contained(directory,'plan.json').read_bytes())
        output.verify_sequence(plan,per_plan,receipt,sid,groups[sid],sources)
        db=receipt['databases'][sid]
        assert db['sha256']==seq['database_sha256'] and sha(output.contained(directory,db['path']))==db['sha256']
    checkpoint=R/f'artifacts/rbf-all-class-final-refit-independent-byte-freeze-v1-20261004/seed{value["seed"]}/checkpoint'
    return job,checkpoint,model_binding(value['seed'],checkpoint,job,model,output),binding
