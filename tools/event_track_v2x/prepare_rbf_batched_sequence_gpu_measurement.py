"""Freeze a distinct real-sequence GPU candidate; never dispatch or promote it."""
import ast
import base64
import hashlib
import json
from pathlib import Path
import zlib

from rbf_nested_seen_val_v2_common import R, new, register, sha


def main():
    parent = R/'source-freezes/rbf-final-refit-full-train-exclusive-forest-GPU-v1-20261004'
    original = parent/'bootstrap.py'
    assert sha(original) == 'e363579c19831420b9782eb7aef4d96e50f3ab303239942ed78923ee481929dd'
    accepted = R/'artifacts/rbf-batched-full-sequence-independent-acceptance-v2-20261004'
    admission = json.loads((accepted/'independent-sequence-receipt.json').read_bytes())
    assert admission['events'] == 195 and admission['all_discrete_outputs_identical'] and admission['all_continuous_outputs_match']
    for role, checksum in admission['reports'].items():
        assert sha(accepted/(role+'-independent.json')) == checksum
    assert sha(accepted/'output-comparison.json') == admission['output_comparison_sha256']
    pair = R/'artifacts/rbf-batched-full-real-sequence-CPU-pair-v1-20261004'
    binding = json.loads((pair/'input-binding.json').read_bytes())
    old = original.read_text()
    tree = ast.parse(old)
    patch_node = next(n for n in tree.body if isinstance(n, ast.Assign)
        and isinstance(n.targets[0], ast.Name) and n.targets[0].id == 'PATCHES')
    patches = ast.literal_eval(patch_node.value)
    added = {}
    prefix = 'transvision/models/event_track_v2x/'
    for name in ('exclusive_paper_runtime.py', 'batched_row_context_scoring.py', 'batched_persistent_cache_stream.py'):
        relative = prefix+name
        p = pair/'batched-source'/relative
        assert sha(p) == binding['source_hashes']['batched'][relative]
        added[relative] = dict(sha256=sha(p), data=base64.b64encode(zlib.compress(p.read_bytes(),9)).decode())
    observer = Path(__file__).with_name('rbf_gpu_replay_measurement.py')
    added[prefix+observer.name] = dict(sha256=sha(observer), data=base64.b64encode(zlib.compress(observer.read_bytes(),9)).decode())
    patches.update(added)
    lines = old.splitlines(keepends=True)
    lines[patch_node.lineno-1:patch_node.end_lineno] = ['PATCHES='+repr(patches)+'\n']
    candidate = ''.join(lines)
    edits = [
        ('from transvision.models.event_track_v2x.exclusive_paper_runtime import replay,default_configuration',
         'from transvision.models.event_track_v2x.exclusive_paper_runtime import replay,default_configuration\n from transvision.models.event_track_v2x.rbf_gpu_replay_measurement import measure_replay'),
        ("seqs=sorted(event_asset['origin_us_by_sequence'])[rank::plan['world_size']];expected={}",
         "seqs=sorted(event_asset['origin_us_by_sequence'])[rank::plan['world_size']][:1];expected={}"),
        ("try:receipt=replay(cache,selected,destination,protocol=PaperProtocol('spd','train'),configuration=config,scorer=scorer,model_binding=bound,fixture=False)",
         "try:receipt=measure_replay(replay,device='cuda:'+str(rank),cache=cache,events=selected,output=destination,protocol=PaperProtocol('spd','train'),configuration=config,scorer=scorer,model_binding=bound,fixture=False)"),
        ("task_name='final-refit exclusive full-train forest candidate'", "task_name='batched complete real-sequence GPU measurement candidate'"),
        ("Path('rbf-original-joint-exclusive-replay')", "Path('rbf-batched-complete-sequence-GPU-measurement')"),
        ("'kind':'rbf_final_refit_exclusive_full_train_forest_candidate_v1'", "'kind':'rbf_batched_complete_sequence_GPU_measurement_candidate_v1'"),
        ("'all_46_sequences_7445_events_completed':complete", "'all_selected_complete_sequences_completed':complete,'all_46_sequences_7445_events_completed':False,'GPU_memory_target_admitted':False,'production_promotion_allowed':False"),
        ("same exclusive factory with capacity-undecided action handling; unchanged model, limits and full real inputs", "distinct CPU-qualified batched row factory; unchanged model and forest limits; complete selected real sequences"),
        ("'scope':'full paired train, distinct frozen final-refit checkpoint, unchanged exclusive kernel and limits; bound allocator only; full independent forest/learned Stage2/MHT/metrics pending'",
         "'scope':'first full real sequence assigned to each rank; distinct CPU-qualified batched row execution; original final checkpoint and forest limits; independent GPU numeric, per-card memory and throughput validation pending'")]
    for before, after in edits:
        assert candidate.count(before) == 1, before
        candidate = candidate.replace(before, after, 1)
    root = R/'source-freezes/rbf-batched-complete-sequence-GPU-measurement-v1-20261004'
    compile(candidate, str(root/'bootstrap.py'), 'exec')
    plan = dict(next(s['plan'] for s in json.loads((parent/'preparation.json').read_bytes())['seeds'] if s['seed']==2027))
    plan.update(bootstrap_sha256=hashlib.sha256(candidate.encode()).hexdigest(),
        exclusive_patches={k:v['sha256'] for k,v in patches.items()}, world_size=None,
        execution_variant='batched-same-arrival-max64-complete-sequence-v1',
        CPU_single_sequence_admission_sha256=sha(accepted/'independent-sequence-receipt.json'),
        GPU_memory_target_percent=[75,80], production_promotion_allowed=False)
    assert plan['configuration'] == binding['configuration']
    root.mkdir(exist_ok=False)
    (root/'bootstrap.py').write_text(candidate)
    for p in (Path(__file__).resolve(), observer, Path(__file__).with_name('rbf_nested_seen_val_v2_common.py')):
        with (root/p.name).open('xb') as f:f.write(p.read_bytes())
    receipt=root/'preparation.json'
    new(receipt,dict(kind='rbf_batched_complete_sequence_GPU_measurement_preparation_v1',plan=plan,
        sources={p.name:dict(sha256=sha(p),bytes=p.stat().st_size) for p in root.glob('*.py')},
        CPU_admission_path=str(accepted/'independent-sequence-receipt.json'),
        actual_GPU_execution_started=False, upload_started=False,
        independent_GPU_reader_pending=True, GPU_memory_target_admitted=False))
    register(receipt,'rbf-batched-complete-sequence-GPU-measurement-preparation')
    print(json.dumps(dict(preparation=str(receipt),sha256=sha(receipt))))


if __name__=='__main__':main()
