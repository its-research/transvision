"""Prepare a distinct four/eight-GPU prefix profiler, without dispatching it."""
import ast
import base64
import datetime
import hashlib
import json
from pathlib import Path
import zlib

from rbf_nested_seen_val_v2_common import R, new, register, sha


def main():
    original = R/'source-freezes/rbf-final-refit-full-train-exclusive-forest-GPU-v1-20261004/bootstrap.py'
    observer_root = R/'source-freezes/rbf-original-frozen-replay-external-function-profile-v1-20261004'
    observer = observer_root/'rbf_frozen_replay_cost_profile.py'
    freeze = json.loads((observer_root/'source-freeze.json').read_bytes())
    for name, spec in freeze['sources'].items():
        assert sha(observer_root/name) == spec['sha256']
    assert sha(original) == 'e363579c19831420b9782eb7aef4d96e50f3ab303239942ed78923ee481929dd'
    root = R/'source-freezes/rbf-original-real-prefix-GPU-function-profile-executor-v1-20261004'
    assert not root.exists(), 'exclusive new preparation; do not overwrite'
    old = original.read_text()
    encoded = base64.b64encode(zlib.compress(observer.read_bytes(), 9)).decode()
    candidate = old + '\n'
    edits = [
        ("FILES_SERVER=os.environ", "PROFILE_OBSERVER="+repr(encoded)+"\nFILES_SERVER=os.environ"),
        ("base=Path(base_raw);packages(base/'source');", "base=Path(base_raw);sys.path.insert(0,str(base/'observer'));from rbf_frozen_replay_cost_profile import profile_replay;packages(base/'source');"),
        ("seqs=sorted(event_asset['origin_us_by_sequence'])[rank::plan['world_size']];expected={}",
         "seqs=sorted(event_asset['origin_us_by_sequence'])[rank::plan['world_size']][:1];expected={}"),
        ("selected=[e for e in events if e['sequence_id']==sequence];destination=out/sequence;error=None;receipt=None",
         "selected_all=[e for e in events if e['sequence_id']==sequence];assert plan['profile_prefix_events']==32 and len(selected_all)>32;selected=selected_all[:32];assert selected_all[32]['decision_us']>selected[-1]['decision_us'];expected[sequence]=[r for r in expected[sequence] if r['decision_us']<=selected[-1]['decision_us']];destination=out/sequence;error=None;receipt=None\n"
         "   runtime_identity={'events_from_independently_admitted_real_schedule':True,'causal_event_bytes_unchanged':True,'read_only_model_loaded_from_frozen_checkpoint':True,'actual_device':'cuda:'+str(rank),'GPU_uuid':str(props.uuid),'TF32_matmul':False,'TF32_cudnn':False,'torch_version':torch.__version__,'numpy_version':np.__version__,'python_version':platform.python_version(),'CUDA_version':torch.version.cuda,'platform':platform.platform(),'world_size':plan['world_size'],'rank':rank}\n"
         "   source_files={str(p.relative_to(base/'source')):sha(p) for p in (base/'source/transvision/models/event_track_v2x').glob('*.py')}"),
        ("try:receipt=replay(cache,selected,destination,protocol=PaperProtocol('spd','train'),configuration=config,scorer=scorer,model_binding=bound,fixture=False)",
         "try:receipt=profile_replay(replay,source_root=base/'source',source_manifest=source_files,cache=cache,events=selected,output=destination,protocol=PaperProtocol('spd','train'),configuration=config,scorer=scorer,model_binding=bound,runtime_identity=runtime_identity,schedule_path=base/'events.json',schedule_sha256=plan['events']['sha256'],selected_sequences=[sequence],per_sequence_prefix_count=32,synchronize=lambda:torch.cuda.synchronize(rank))"),
        ("databases=list(destination.glob('*.sqlite'));", "databases=list((destination/'original-replay').glob('*.sqlite'));"),
        ("task_name='final-refit exclusive full-train forest candidate'", "task_name='source-bound original real-prefix function profile debugging candidate'"),
        ("base=Path('rbf-original-joint-exclusive-replay').absolute();base.mkdir();", "base=Path('rbf-original-real-prefix-function-profile').absolute();base.mkdir();observer=base/'observer';observer.mkdir();raw=zlib.decompress(base64.b64decode(PROFILE_OBSERVER));assert hashlib.sha256(raw).hexdigest()==plan['profile_observer_sha256'];(observer/'rbf_frozen_replay_cost_profile.py').write_bytes(raw);"),
        ("'kind':'rbf_final_refit_exclusive_full_train_forest_candidate_v1'", "'kind':'rbf_original_frozen_real_prefix_GPU_function_profile_candidate_v1'"),
        ("'all_46_sequences_7445_events_completed':complete", "'all_selected_real_prefixes_profiled':complete,'full_46_sequence_replay_completed':False,'profiler_overhead_quantified':False,'paper_cost_admission':False"),
        ("'scope':'full paired train, distinct frozen final-refit checkpoint, unchanged exclusive kernel and limits; bound allocator only; full independent forest/learned Stage2/MHT/metrics pending'",
         "'scope':'original causal 32-event prefix of first assigned sequence per rank; unchanged learned scorer, forest kernel/config/limits and final-refit checkpoint; external cProfile only; no full replay, latency or paper cost claim'"),
    ]
    for before, after in edits:
        assert candidate.count(before) == 1, ('unexpected source edit', before)
        candidate = candidate.replace(before, after, 1)
    compile(candidate, str(root/'bootstrap.py'), 'exec', dont_inherit=True)
    left, right = ast.parse(old), ast.parse(candidate)
    def constants(tree):
        return {n.targets[0].id: ast.dump(n.value, include_attributes=False) for n in tree.body
                if isinstance(n, ast.Assign) and len(n.targets)==1 and isinstance(n.targets[0], ast.Name)
                and n.targets[0].id in ('PATCHES','REPLACEMENTS')}
    assert constants(left) == constants(right)
    def functions(tree):
        return {n.name:ast.dump(n, include_attributes=False) for n in tree.body if isinstance(n,ast.FunctionDef)}
    lf, rf = functions(left), functions(right)
    assert {k:v for k,v in lf.items() if k not in ('work','main')} == {k:v for k,v in rf.items() if k not in ('work','main')}
    preparation = json.loads((original.parent/'preparation.json').read_bytes())
    parent = next(v['plan'] for v in preparation['seeds'] if v['seed']==2027)
    schedule = R/'artifacts/rbf-original-joint-coupled-forest-GPU4-v1-source-20261001/independent-cloud-readback/events-seed2027.bytes'
    assert sha(schedule) == parent['events']['sha256']
    events = json.loads(schedule.read_bytes())
    selections = []
    for world in (4,8):
        seqs = sorted(events['origin_us_by_sequence'])[:world]
        for sequence in seqs:
            seq = [e for e in events['events'] if e['sequence_id']==sequence]
            assert len(seq)>32 and seq[32]['decision_us']>seq[31]['decision_us']
        selections.append(dict(world_size=world, sequences=seqs, prefix_events_per_rank=32,
                               total_measured_events=32*world, full_dataset_coverage=False))
    plan = dict(parent)
    plan.update(bootstrap_sha256=hashlib.sha256(candidate.encode()).hexdigest(), world_size=None,
        profile_observer_sha256=sha(observer), profile_prefix_events=32,
        profile_library_source_freeze_sha256=sha(observer_root/'source-freeze.json'),
        frozen_parent_bootstrap_sha256=sha(original), scope='separately named source-bound real-prefix cost debugging; no model/kernel changes',
        profiling_debug_contract=True, full_dataset_or_latency_claim=False)
    for key in ('source','events','cache_archive','cache_manifest','checkpoint','weights_archive','configuration','scoring_atol','scoring_rtol'):
        assert plan[key] == parent[key]
    root.mkdir()
    (root/'bootstrap.py').write_text(candidate)
    for path in (Path(__file__).resolve(), Path(__file__).with_name('rbf_nested_seen_val_v2_common.py')):
        with (root/path.name).open('xb') as stream:
            stream.write(path.read_bytes())
    receipt = root/'preparation.json'
    new(receipt, dict(kind='rbf_original_real_prefix_GPU_function_profile_executor_preparation_v1',
        checked_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), plan=plan,
        sources={p.name:dict(sha256=sha(p),bytes=p.stat().st_size) for p in root.glob('*.py')},
        unchanged_effective_kernel_PATCHES_and_REPLACEMENTS_AST=True,
        unchanged_fetch_unpack_and_input_validation_helpers=True,
        original_checkpoint_and_inference_source_bytes_unchanged=True,
        original_all_class_candidate_and_legal_context_and_capacity_limits_unchanged=True,
        selections=selections, independent_prefix_factor_state_and_profile_reader_pending=True,
        upload_or_actual_GPU_execution_started=False, profile_experiment_admitted=False,
        scalar_function_wall_times_are_not_GPU_kernel_occupancy_or_isolated_latency=True))
    register(receipt, 'rbf-original-real-prefix-GPU-function-profile-executor-preparation')
    print(json.dumps(dict(preparation=str(receipt),sha256=sha(receipt),bootstrap_sha256=sha(root/'bootstrap.py'),
        original_kernel_unchanged=True,actual_GPU_profile=False)),flush=True)


if __name__ == '__main__':
    main()
