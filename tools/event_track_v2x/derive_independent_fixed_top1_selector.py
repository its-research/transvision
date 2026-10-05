"""Derive strict K1 bindings without changing the loss or SHA arithmetic."""
import ast
import json
from pathlib import Path

from rbf_nested_seen_val_v2_common import R, new, register, sha

PARENT = R/'source-freezes/rbf-independent-fixed-topK-conditional-selector-v1-20261004'


def replace(source, before, after, count=1):
    assert source.count(before) == count, (before, source.count(before))
    return source.replace(before, after)


def main():
    freeze = json.loads((PARENT/'source-freeze.json').read_bytes())
    for name, spec in freeze['sources'].items():
        assert sha(PARENT/name) == spec['sha256']
    oracle = (PARENT/'independent_fixed_topk_selector.py').read_text()
    oracle = oracle.replace('rbf_independent_fixed_topK', 'rbf_independent_fixed_Top1')
    oracle = oracle.replace('all_node_local_legal_root_class_topK_choices_verified', 'all_node_local_legal_root_class_Top1_choices_verified')
    oracle = replace(oracle, "proof['beam_width'] == 4", "proof['beam_width'] == 1")
    oracle = replace(oracle, "meta['config']['state']['active_limit'] == 4", "meta['config']['state']['active_limit'] == 1")
    oracle = replace(oracle, '1 <= len(handles) <= 4', 'len(handles) == 1')
    oracle = oracle.replace('independent_fixed_topK_conditional_output_selector', 'independent_fixed_Top1_conditional_output_selector')
    def functions(source):
        return {n.name:ast.dump(n,include_attributes=False) for n in ast.parse(source).body if isinstance(n,ast.FunctionDef)}
    left, right = functions((PARENT/'independent_fixed_topk_selector.py').read_text()), functions(oracle)
    for name in ('sha', 'pairwise_risks', 'normative_float64'):
        assert left[name] == right[name], 'unchanged numerical function: '+name
    qualifier = (PARENT/'qualify_independent_fixed_topk_selector.py').read_text()
    qualifier = qualifier.replace('independent_fixed_topk_selector', 'independent_fixed_top1_selector')
    qualifier = qualifier.replace('rbf-original-joint-coupled-topk-GPU4-v5-native34-readback-20261001', 'rbf-original-joint-fixed-top1-GPU4-v1-readback-20261002')
    qualifier = qualifier.replace('3940c73ff8d80941c1bdbc5e3e68a5e46276073588c4346e25b3dd6097e9aaf1', 'f43a3a003c238b7fa5a2d44331bf4ffe851577fd89eed420528f4341615fc521')
    qualifier = qualifier.replace('rbf-independent-fixed-topK-search-pruning-qualification-v3', 'rbf-independent-fixed-Top1-search-pruning-qualification-v1')
    qualifier = qualifier.replace('rbf_independent_fixed_topK', 'rbf_independent_fixed_Top1')
    qualifier = qualifier.replace('all_node_local_legal_root_class_topK_choices_verified', 'all_node_local_legal_root_class_Top1_choices_verified')
    qualifier = replace(qualifier, '    return dict(MAP_not_Bayes_example=True,',
        "    chosen, single, risks = check([(0,1)], [1.], ['a'], False, .2)\n    assert chosen==0 and risks==(0.,) and single['conditional_risk']==single['optimization_gap']==0. and single['risk_bound']==.2\n    _, complete_single, _ = check([(0,1)], [1.], ['a'], True)\n    assert complete_single['risk_bound']==0.\n    return dict(K1_single_supported_action_case=True, generic_multi_action_MAP_not_Bayes_example=True,")
    start, stop = qualifier.index('def mutate_selected'), qualifier.index('\n\ndef main():')
    mutation = '''def mutate_selected(db, name):
    for ordinal, blob in db.execute('SELECT ordinal,audit FROM events ORDER BY ordinal'):
        value = json.loads(blob)
        for s in value['components']:
            if len(s['active']) != 1 or len(s['decision_indices']) < 2:
                continue
            if name == 'nonretained_output':
                digest = db.execute(f"SELECT sha FROM pc{s['component']}_prefixes WHERE h=0").fetchone()[0]
                s['output_handle'], s['output_sha256'] = 0, digest
            elif name == 'output_SHA': s['output_sha256'] = '0'*64
            elif name in ('conditional_risk','relaxed_bayes_lower','optimization_gap','risk_bound'):
                s['decision'][name] += .001
            elif name == 'wrong_complete_lower_kind': s['decision']['conditional_bayes_lower_kind'] = 'full_posterior_certified'
            elif name == 'empty_scope_flag': s['decision']['empty_loss_scope'] = not s['decision']['empty_loss_scope']
            elif name == 'global_model_regret': value['model_regret_upper'] += .001
            elif name == 'regret_fallback': value['global_fallback_used'] = True
            elif name == 'MAP_policy_label': value['output_policy'] = 'retained_posterior_MAP'
            elif name == 'query_scope': s['decision_indices'].reverse()
            elif name == 'multiple_active_classes': s['active'].append(dict(s['active'][0]))
            elif name == 'bad_conditional_normalizer': s['log_retained'] += .001
            else: raise AssertionError(name)
            db.execute('UPDATE events SET audit=? WHERE ordinal=?', (json.dumps(value,sort_keys=True).encode(), ordinal))
            return
    raise AssertionError('no applicable K1 decision mutation: '+name)
'''
    qualifier = qualifier[:start]+mutation+qualifier[stop:]
    qualifier = qualifier.replace("baseline['posterior_MAP_and_output_differ'] == 17", "baseline['posterior_MAP_and_output_differ'] == 0 and baseline['scoped_identity_queries'] == 4794")
    qualifier = qualifier.replace("('wrong_retained_output','MAP_substitution','output_SHA','conditional_risk',", "('nonretained_output','output_SHA','conditional_risk',")
    qualifier = qualifier.replace("'query_scope','active_class_order','bad_conditional_normalizer'", "'query_scope','multiple_active_classes','bad_conditional_normalizer'")
    qualifier = replace(qualifier, '        edits = {', "        edits = {\n            'foreign_K4_search_kind': lambda v:v.update(kind='rbf_independent_fixed_topK_raw_graph_search_pruning_audit_v1'),\n            'foreign_K4_beam_width': lambda v:v.update(beam_width=4),")
    qualifier = qualifier.replace('len(negatives) == 21', 'len(negatives) == 22')
    qualifier = qualifier.replace('existing_MAP_and_Bayes_output_differences=17', 'existing_MAP_and_Bayes_output_differences=0, sole_K1_supported_action_verified=True')
    driver = (PARENT/'accept_final_refit_topk_selector.py').read_text()
    preparer = (PARENT/'prepare_independent_fixed_topk_selector.py').read_text()
    for name in ('driver', 'preparer'):
        source = driver if name=='driver' else preparer
        source = source.replace('independent_fixed_topk_selector', 'independent_fixed_top1_selector')
        source = source.replace('accept_final_refit_topk_selector', 'accept_final_refit_top1_selector')
        source = source.replace('accept_final_refit_topk_search', 'accept_final_refit_top1_search')
        source = source.replace('rbf_final_refit_topk_binding', 'rbf_final_refit_top1_binding')
        source = source.replace('rbf-independent-fixed-topK', 'rbf-independent-fixed-Top1')
        source = source.replace('rbf-final-refit-fixed-topK-full-independent-CPU', 'rbf-final-refit-fixed-Top1-full-independent-CPU')
        source = source.replace('rbf_independent_fixed_topK', 'rbf_independent_fixed_Top1')
        source = source.replace('rbf_final_refit_fixed_topK', 'rbf_final_refit_fixed_Top1')
        source = source.replace("['K'] == 4", "['K'] == 1").replace('K=4', 'K=1')
        source = source.replace("['negative_controls_rejected']==21", "['negative_controls_rejected']==22")
        source = source.replace("['negative_controls_rejected'] == 21", "['negative_controls_rejected'] == 22")
        source = source.replace("['existing_MAP_and_Bayes_output_differences'] == 17", "['existing_MAP_and_Bayes_output_differences'] == 0")
        source = source.replace('MAP_and_Bayes_differences=17', 'MAP_and_Bayes_differences=0')
        source = source.replace('semantic_negative_controls_rejected=21', 'semantic_negative_controls_rejected=22')
        source = source.replace('completed new TopK full byte/factor readback', 'completed new Top-1 full byte/factor readback')
        if name=='preparer':
            source = replace(source, '        controls={', "        controls={\n            'foreign_K4_search_kind':lambda v:v.update(kind='rbf_final_refit_fixed_topK_full_search_pruning_admission_v1'),\n            'foreign_K4_width':lambda v:v.update(K=4),")
            source = replace(source, "references=[SEARCH_SOURCE/'source-freeze.json',", "references=[SEARCH_SOURCE/'source-freeze.json',\n        R/'source-freezes/rbf-independent-fixed-topK-conditional-selector-v1-20261004/source-freeze.json',\n        R/'receipts/rbf-independent-fixed-Top1-conditional-selector-source-derivation-20261004.json',")
            preparer = source
        else:
            driver = source
    sources = {'independent_fixed_top1_selector.py':oracle, 'qualify_independent_fixed_top1_selector.py':qualifier,
        'accept_final_refit_top1_selector.py':driver, 'prepare_independent_fixed_top1_selector.py':preparer}
    directory = Path(__file__).resolve().parent
    for filename, source in sources.items():
        ast.parse(source); compile(source, filename, 'exec')
        with (directory/filename).open('x') as stream:stream.write(source)
    p = R/'receipts/rbf-independent-fixed-Top1-conditional-selector-source-derivation-20261004.json'
    value = dict(kind='rbf_independent_fixed_Top1_selector_source_derivation_v1',parent_source_freeze_sha256=sha(PARENT/'source-freeze.json'),
        sources={filename:dict(path=str(directory/filename),sha256=sha(directory/filename)) for filename in sources},
        derivation_source_sha256=sha(__file__),pairwise_Decimal_and_normative_float64_functions_AST_unchanged=True,
        single_K1_admitted_action_required=True,K4_original_sources_preserved=True,
        independent_K1_qualification_pending=True,new_final_model_experiment_accepted=False,paper_performance_complete=False)
    new(p,value);register(p,value['kind']);print(json.dumps(dict(derivation_receipt=str(p))))


if __name__=='__main__':main()
