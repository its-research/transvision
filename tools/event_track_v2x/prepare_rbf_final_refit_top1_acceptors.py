"""Prepare separately named K1 byte and causal/cache/state gates.

The numerical oracles remain unchanged. The K4 binding is never relaxed: a
new K1 contract refuses K4 and old-model evidence before inspecting outputs.
"""
import ast
import datetime
import json
from pathlib import Path
import shutil

from rbf_nested_seen_val_v2_common import R, new, register, sha

READER_PARENT = R/'source-freezes/rbf-final-refit-main-topK-independent-output-reader-v1-20261004'
CPU_PARENT = R/'source-freezes/rbf-final-refit-fixed-topK-full-independent-CPU-v1-20261004'
PRODUCER = R/'source-freezes/rbf-final-refit-full-train-fixed-Top1-GPU-v1-20261004'
READER = R/'source-freezes/rbf-final-refit-Top1-independent-output-reader-v1-20261004'
CPU = R/'source-freezes/rbf-final-refit-fixed-Top1-full-independent-CPU-v1-20261004'
BYTE_KIND = 'rbf_final_refit_full_train_fixed_Top1_independent_bytes_events_factor_admission_v1'


def replace(source, before, after, count=1):
    assert source.count(before) == count, (before, source.count(before), count)
    return source.replace(before, after)


def verify_sources(root):
    frozen = json.loads((root/'source-freeze.json').read_bytes())
    for name, spec in frozen['sources'].items():
        assert sha(root/name) == spec['sha256'] and (root/name).stat().st_size == spec['bytes']
    return frozen


def main():
    assert not READER.exists() and not CPU.exists(), 'create-once preparation; preserve and inspect existing files'
    verify_sources(READER_PARENT)
    verify_sources(CPU_PARENT)
    prep = json.loads((PRODUCER/'preparation.json').read_bytes())
    for name, spec in prep['execution_sources'].items():
        assert sha(PRODUCER/name) == spec['sha256']
    reader = (READER_PARENT/'read_rbf_final_refit_forest_outputs.py').read_text()
    reader = replace(reader, "parser.add_argument('--method',required=True,choices=('rbf','topk'))",
        "parser.add_argument('--method',choices=('topk',),default='topk')")
    reader = replace(reader, "variant='exclusive-forest' if args.method=='rbf' else 'fixed-topK'", "variant='fixed-Top1'")
    reader = replace(reader, "assert plan['method']==args.method and world in (4,8)",
        "assert plan['method']==args.method=='topk' and world in (4,8)\n    assert plan['configuration']['state']['active_limit']==1\n    assert plan['baseline_variant']=='fixed_Top1_final_refit_full_train_v1'\n    assert plan['old_nested_Top1_acceptance_inherited'] is False")
    reader = replace(reader,
        "root=R/f'artifacts/rbf-final-refit-full-train-forest-independent-byte-factor-v1-20261004/{args.method}/seed{args.seed}'",
        "root=R/f'artifacts/rbf-final-refit-full-train-fixed-Top1-independent-byte-factor-v1-20261004/seed{args.seed}'")
    reader = reader.replace('rbf_final_refit_full_train_fixed_topK_candidate_v1', 'rbf_final_refit_full_train_fixed_Top1_candidate_v1')
    reader = reader.replace('rbf_final_refit_full_train_forest_independent_bytes_events_factor_admission_v1', BYTE_KIND)
    reader = reader.replace('rbf_final_refit_forest_failed_candidate_independent_bytes_v1', 'rbf_final_refit_fixed_Top1_failed_candidate_independent_bytes_v1')
    source_guard = """directory=Path(__file__).resolve().parent
    freeze=json.loads((directory/'source-freeze.json').read_bytes())
    for name,spec in freeze['sources'].items():
        assert sha(directory/name)==spec['sha256'] and (directory/name).stat().st_size==spec['bytes']
    for item in freeze['references']:
        assert sha(item['path'])==item['sha256']
    from clearml import Task"""
    reader = replace(reader, '    from clearml import Task\n    variant=', '    '+source_guard+'\n    variant=')
    compile(reader, 'read_rbf_final_refit_top1_outputs.py', 'exec')
    functions = lambda s: {n.name: ast.dump(n, include_attributes=False) for n in ast.parse(s).body if isinstance(n, ast.FunctionDef)}
    old_functions, new_functions = functions((READER_PARENT/'read_rbf_final_refit_forest_outputs.py').read_text()), functions(reader)
    assert {k:v for k,v in old_functions.items() if k!='main'} == {k:v for k,v in new_functions.items() if k!='main'}
    READER.mkdir()
    (READER/'read_rbf_final_refit_top1_outputs.py').write_text(reader)
    shutil.copyfile(READER_PARENT/'rbf_nested_seen_val_v2_common.py', READER/'rbf_nested_seen_val_v2_common.py')
    shutil.copyfile(__file__, READER/Path(__file__).name)
    references = [READER_PARENT/'source-freeze.json', CPU_PARENT/'source-freeze.json', PRODUCER/'preparation.json']
    new(READER/'source-freeze.json', dict(kind='rbf_final_refit_Top1_independent_byte_factor_reader_source_v1',
        checked_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        sources={p.name:dict(sha256=sha(p),bytes=p.stat().st_size) for p in READER.iterdir()},
        references=[dict(path=str(p),sha256=sha(p)) for p in references],
        artifact_range_read_safe_unpack_and_factor_normalization_functions_AST_identical=True,
        K1_contract_does_not_admit_K4_or_old_checkpoint=True, whole_forest_or_performance_accepted=False))
    workspace = Path(__file__).resolve().parent
    mapping = {
        'rbf_final_refit_topk_binding.py':'rbf_final_refit_top1_binding.py',
        'accept_final_refit_topk_cohort.py':'accept_final_refit_top1_cohort.py',
        'prepare_rbf_final_refit_topk_CPU.py':'prepare_rbf_final_refit_top1_cpu.py',
    }
    generated = []
    for old_name, new_name in mapping.items():
        source = (CPU_PARENT/old_name).read_text()
        source = source.replace('rbf_final_refit_topk_binding', 'rbf_final_refit_top1_binding')
        source = source.replace('accept_final_refit_topk_cohort', 'accept_final_refit_top1_cohort')
        source = source.replace('rbf-final-refit-fixed-topK-full-independent-CPU', 'rbf-final-refit-fixed-Top1-full-independent-CPU')
        source = source.replace('rbf-final-refit-full-train-fixed-topK-GPU', 'rbf-final-refit-full-train-fixed-Top1-GPU')
        source = source.replace('rbf-original-joint-coupled-topk-GPU4-v5-native34-dispatch-20261001.json', 'rbf-original-joint-fixed-top1-GPU4-v1-dispatch-20261002.json')
        source = source.replace('rbf-final-refit-fixed-topK-full-independent-CPU-preparation', 'rbf-final-refit-fixed-Top1-full-independent-CPU-preparation')
        source = source.replace('rbf_final_refit_full_train_forest_independent_bytes_events_factor_admission_v1', BYTE_KIND)
        source = source.replace('rbf_final_refit_fixed_topK', 'rbf_final_refit_fixed_Top1')
        source = source.replace('K=4', 'K=1').replace("['active_limit'] == 4", "['active_limit'] == 1")
        source = source.replace('TopK', 'Top1').replace('topK', 'Top1')
        if old_name.startswith('rbf_final'):
            source = replace(source, "assert value['method'] == plan['method'] == 'topk'",
                "assert value['method'] == plan['method'] == 'topk'\n    assert plan['baseline_variant']=='fixed_Top1_final_refit_full_train_v1'\n    assert plan['old_nested_Top1_acceptance_inherited'] is False\n    reader=R/'source-freezes/rbf-final-refit-Top1-independent-output-reader-v1-20261004/read_rbf_final_refit_top1_outputs.py'\n    assert value['source_sha256']==sha(reader)")
        if old_name.startswith('prepare_'):
            source = replace(source, "INDEX, PRODUCER/'preparation.json', OLD_JOURNAL]",
                "INDEX, PRODUCER/'preparation.json', OLD_JOURNAL,\n        R/'source-freezes/rbf-final-refit-Top1-independent-output-reader-v1-20261004/source-freeze.json',\n        R/'source-freezes/rbf-final-refit-Top1-independent-output-reader-v1-20261004/read_rbf_final_refit_top1_outputs.py']")
            source = replace(source, "total_nodes=entry['rows'])",
                "total_nodes=entry['rows'], source_sha256=sha(R/'source-freezes/rbf-final-refit-Top1-independent-output-reader-v1-20261004/read_rbf_final_refit_top1_outputs.py'))", 2)
            source = replace(source, "    mutations = {", "    mutations = {\n        'K4_byte_receipt': lambda v,j:v.update(kind='rbf_final_refit_full_train_forest_independent_bytes_events_factor_admission_v1'),\n        'foreign_reader_source': lambda v,j:v.update(source_sha256='b'*64),")
        compile(source, new_name, 'exec')
        destination = workspace/new_name
        with destination.open('x') as stream:
            stream.write(source)
        generated.append(dict(path=str(destination),sha256=sha(destination)))
    receipt = R/'receipts/rbf-final-refit-Top1-byte-reader-and-CPU-source-derivation-20261004.json'
    value = dict(kind='rbf_final_refit_Top1_reader_and_CPU_source_derivation_v1',
        reader_source_freeze_sha256=sha(READER/'source-freeze.json'),
        parent_K4_CPU_source_freeze_sha256=sha(CPU_PARENT/'source-freeze.json'),generated_workspace_sources=generated,
        numerical_oracles_unchanged=True, separate_K1_identity_and_kind=True,
        completed_real_output_admission=False, full_forest_or_same_resource_performance_accepted=False)
    new(receipt,value);register(receipt,value['kind'])
    print(json.dumps(dict(reader=str(READER),CPU_preparer=str(workspace/'prepare_rbf_final_refit_top1_cpu.py'),receipt=str(receipt))))


if __name__=='__main__':
    main()
