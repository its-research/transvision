"""Derive a separately bound Top-1 search gate; preserve the K4 gate.

Only width admission, scope labels and the legacy qualification input change
in the oracle. The raw graph, root-class masses and pruning algorithm remain
unchanged. Qualification must run on K1 evidence before a source freeze.
"""
import ast
import datetime
import json
from pathlib import Path

from rbf_nested_seen_val_v2_common import R, new, register, sha

PARENT = R/'source-freezes/rbf-independent-fixed-topK-search-pruning-v1-20261004'
DB_SHA = 'f43a3a003c238b7fa5a2d44331bf4ffe851577fd89eed420528f4341615fc521'
QUAL = R/'artifacts/rbf-independent-fixed-Top1-search-pruning-qualification-v1-20261004/qualification.json'


def replace(source, before, after, count=1):
    assert source.count(before) == count, (before, source.count(before))
    return source.replace(before, after)


def main():
    freeze = json.loads((PARENT/'source-freeze.json').read_bytes())
    for name, spec in freeze['sources'].items():
        assert sha(PARENT/name) == spec['sha256']
    old = (PARENT/'independent_fixed_topk_search.py').read_text()
    oracle = replace(old, 'self.width == 4', 'self.width == 1')
    oracle = oracle.replace('rbf_independent_fixed_topK_raw_graph_search_pruning_audit_v1', 'rbf_independent_fixed_Top1_raw_graph_search_pruning_audit_v1')
    oracle = oracle.replace('independent_fixed_topK_raw_graph_search_and_pruning', 'independent_fixed_Top1_raw_graph_search_and_pruning')
    oracle = oracle.replace('all_node_local_legal_root_class_topK_choices_verified', 'all_node_local_legal_root_class_Top1_choices_verified')
    # The algorithms are byte-identical after reversing the named edits.
    recovered = oracle.replace('self.width == 1', 'self.width == 4')
    recovered = recovered.replace('rbf_independent_fixed_Top1_raw_graph_search_pruning_audit_v1', 'rbf_independent_fixed_topK_raw_graph_search_pruning_audit_v1')
    recovered = recovered.replace('independent_fixed_Top1_raw_graph_search_and_pruning', 'independent_fixed_topK_raw_graph_search_and_pruning')
    recovered = recovered.replace('all_node_local_legal_root_class_Top1_choices_verified', 'all_node_local_legal_root_class_topK_choices_verified')
    assert recovered == old
    qualifier = (PARENT/'qualify_independent_fixed_topk_search.py').read_text()
    qualifier = qualifier.replace('independent_fixed_topk_search', 'independent_fixed_top1_search')
    qualifier = qualifier.replace('rbf-original-joint-coupled-topk-GPU4-v5-native34-readback-20261001', 'rbf-original-joint-fixed-top1-GPU4-v1-readback-20261002')
    qualifier = qualifier.replace('3940c73ff8d80941c1bdbc5e3e68a5e46276073588c4346e25b3dd6097e9aaf1', DB_SHA)
    qualifier = qualifier.replace('2d3a4abbf4404f18bc6c362456e0b5bc', '879429b762b74e3b8be42ddcdef34efe')
    qualifier = qualifier.replace('rbf_independent_fixed_topK_search_oracle_qualification_v1', 'rbf_independent_fixed_Top1_search_oracle_qualification_v1')
    qualifier = replace(qualifier, "component = next(s for s in value['components'] if len(s['active']) > 1)",
        "component = next(s for s in value['components'] if len(s['active']) == 1 and len(s['predecessors']) > 1 and s['pruning'] and s['merge_pruning_stages'])")
    qualifier = replace(qualifier, "retained_class_rank=audit_mutation(lambda a, s: s['active'].reverse()),",
        "multiple_retained_classes_in_K1=audit_mutation(lambda a, s: s['active'].append(dict(s['active'][0]))),")
    helper = '''def promote_nonselected_raw_root_class(db):
    meta = {k: json.loads(v) for k, v in db.execute('SELECT k,v FROM meta')}
    audit = oracle.DatabaseAudit(db, oracle.load_root_oracle(), meta)
    for component, in db.execute('SELECT component FROM component_catalog ORDER BY component'):
        tree = audit.tree(component)
        for handle, row in sorted(tree.p.items()):
            if not handle:
                continue
            parent, depth, chosen = row[1], row[2]-1, row[3]
            alternatives = [p for p in tree.allowed[parent] if p != chosen]
            if not alternatives:
                continue
            other = alternatives[0]
            global_i = tree.full_members[depth]
            global_p = -1 if other < 0 else tree.full_members[other]
            # This alters an unchosen root class, leaving the chosen class's
            # raw mass untouched. Its new score must beat the stored K1 path.
            weight = max(w for _, w in tree.factors[depth])+20.
            cursor = db.execute('UPDATE potentials SET w=? WHERE i=? AND p=?', (weight, global_i, global_p))
            assert cursor.rowcount == 1
            return
    raise AssertionError('fixture has no competing legal root class')


def change_meta_width(db):
    value = json.loads(db.execute("SELECT v FROM meta WHERE k='config'").fetchone()[0])
    value['state']['active_limit'] = 4
    db.execute("UPDATE meta SET v=? WHERE k='config'", (json.dumps(value).encode(),))


'''
    qualifier = replace(qualifier, 'def controls():\n', helper+'def controls():\n')
    qualifier = replace(qualifier, '    return dict(\n        multiple_retained_classes_in_K1=',
        '    return dict(\n        promoted_unchosen_legal_root_class=promote_nonselected_raw_root_class,\n        foreign_K4_width=change_meta_width,\n        multiple_retained_classes_in_K1=')
    qualifier = replace(qualifier, "baseline = oracle.verify_database(DB, DB_SHA)",
        "baseline = oracle.verify_database(DB, DB_SHA)\n        assert baseline['beam_width']==1")
    driver = (PARENT/'accept_final_refit_topk_search.py').read_text()
    preparer = (PARENT/'prepare_independent_fixed_topk_search.py').read_text()
    for name in ('driver', 'preparer'):
        source = driver if name == 'driver' else preparer
        source = source.replace('independent_fixed_topk_search', 'independent_fixed_top1_search')
        source = source.replace('rbf_final_refit_topk_binding', 'rbf_final_refit_top1_binding')
        source = source.replace('accept_final_refit_topk_search', 'accept_final_refit_top1_search')
        source = source.replace('rbf-final-refit-fixed-topK-full-independent-CPU', 'rbf-final-refit-fixed-Top1-full-independent-CPU')
        source = source.replace('rbf-independent-fixed-topK-search-pruning', 'rbf-independent-fixed-Top1-search-pruning')
        source = source.replace('qualification-v3-20261004', 'qualification-v1-20261004')
        source = source.replace('rbf_independent_fixed_topK', 'rbf_independent_fixed_Top1')
        source = source.replace('rbf_final_refit_fixed_topK', 'rbf_final_refit_fixed_Top1')
        source = source.replace('K=4', 'K=1')
        source = source.replace("['controls_rejected'] == 18", "['controls_rejected'] == 20")
        source = source.replace('negative_controls_rejected=18', 'negative_controls_rejected=20')
        if name == 'driver':
            driver = source
        else:
            preparer = source
    preparer = replace(preparer, "TCPU/'rbf_final_refit_top1_binding.py',",
        "TCPU/'rbf_final_refit_top1_binding.py',\n        R/'source-freezes/rbf-independent-fixed-topK-search-pruning-v1-20261004/source-freeze.json',\n        R/'receipts/rbf-independent-fixed-Top1-search-pruning-source-derivation-20261004.json',")
    sources = {
        'independent_fixed_top1_search.py': oracle,
        'qualify_independent_fixed_top1_search.py': qualifier,
        'accept_final_refit_top1_search.py': driver,
        'prepare_independent_fixed_top1_search.py': preparer,
    }
    directory = Path(__file__).resolve().parent
    for filename, source in sources.items():
        ast.parse(source); compile(source, filename, 'exec')
        with (directory/filename).open('x') as stream:
            stream.write(source)
    receipt = R/'receipts/rbf-independent-fixed-Top1-search-pruning-source-derivation-20261004.json'
    value = dict(kind='rbf_independent_fixed_Top1_search_source_derivation_v1',
        parent_source_freeze_sha256=sha(PARENT/'source-freeze.json'),
        parent_oracle_sha256=sha(PARENT/'independent_fixed_topk_search.py'),
        derived_sources={filename: dict(path=str(directory/filename),sha256=sha(directory/filename)) for filename in sources},
        derivation_source_sha256=sha(__file__),
        oracle_only_width_and_scope_metadata_changes=True,
        raw_graph_product_root_mass_pruning_work_counter_algorithms_unchanged=True,
        K4_original_sources_preserved=True, independent_K1_qualification_pending=True,
        qualifier_refuses_K4_width_and_changed_K1_retained_classes=True,
        new_final_model_experiment_accepted=False, full_online_method_accepted=False,paper_performance_complete=False)
    new(receipt,value);register(receipt,value['kind'])
    print(json.dumps(dict(derivation_receipt=str(receipt),qualification_output=str(QUAL.parent))))


if __name__=='__main__':
    main()
