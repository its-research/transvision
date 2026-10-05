"""Create-once source freeze for the independently qualified TopK search gate."""
import ast
import datetime
import json
from pathlib import Path
import shutil

from rbf_nested_seen_val_v2_common import R, new, register, sha

D = R/'source-freezes/rbf-independent-fixed-topK-search-pruning-v1-20261004'
QUAL = R/'artifacts/rbf-independent-fixed-topK-search-pruning-qualification-v3-20261004/qualification.json'
TCPU = R/'source-freezes/rbf-final-refit-fixed-topK-full-independent-CPU-v1-20261004'


def main():
    assert not D.exists(), 'immutable source freeze already exists'
    directory = Path(__file__).resolve().parent
    qualification = json.loads(QUAL.read_bytes())
    assert qualification['controls_rejected'] == 18
    assert qualification['software_qualification_only'] is True
    assert qualification['new_final_model_experiment_accepted'] is False
    assert qualification['oracle_sha256'] == sha(directory/'independent_fixed_topk_search.py')
    assert qualification['qualification_source_sha256'] == sha(directory/'qualify_independent_fixed_topk_search.py')
    names = ('independent_fixed_topk_search.py', 'qualify_independent_fixed_topk_search.py',
             'accept_final_refit_topk_search.py', 'prepare_independent_fixed_topk_search.py',
             'rbf_nested_seen_val_v2_common.py')
    for name in names: ast.parse((directory/name).read_bytes())
    references = [Path(qualification['production_archive_path']),
        Path(qualification['original_admission_path']), Path(qualification['original_database']),
        R/'source-freezes/rbf-independent-identity-forest-audit-v1-20261001/oracle.py',
        TCPU/'source-freeze.json', TCPU/'rbf_final_refit_topk_binding.py',
        R/'source-freezes/rbf-final-refit-full-forest-independent-CPU-v3-20261004/source-freeze.json']
    D.mkdir()
    for name in names: shutil.copyfile(directory/name, D/name)
    freeze = dict(kind='rbf_independent_fixed_topK_search_pruning_source_v1',
        checked_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        sources={name: dict(sha256=sha(D/name), bytes=(D/name).stat().st_size) for name in names},
        unchanged_references=[dict(path=str(p), sha256=sha(p)) for p in references],
        qualification=dict(path=str(QUAL), sha256=sha(QUAL)), method='topk', K=4,
        independent_connected_components_by_graph_traversal=True,
        independent_product_search_by_small_Cartesian_enumeration=True,
        independent_heap_work_by_selected_grid_neighbour_union=True,
        independent_root_class_children_by_raw_parent_grouping=True,
        floating_point_tolerance_unchanged=1e-8, full_online_method_accepted=False,
        new_final_model_experiment_accepted=False, same_resource_performance_accepted=False,
        conditional_output_selector_independently_accepted=False, paper_performance_complete=False)
    new(D/'source-freeze.json', freeze)
    receipt = R/'receipts/rbf-independent-fixed-topK-search-pruning-source-preparation-20261004.json'
    final = dict(kind='rbf_independent_fixed_topK_search_pruning_preparation_v1',
        source_freeze_path=str(D/'source-freeze.json'), source_freeze_sha256=sha(D/'source-freeze.json'),
        qualification_path=str(QUAL), qualification_sha256=sha(QUAL),
        exhaustive_product_controls=60, negative_controls_rejected=18,
        existing_small_sequence_events=20, existing_small_sequence_observations=414,
        max_abs_error=qualification['max_abs_error'], local_software_qualification_only=True,
        new_final_model_experiment_accepted=False, full_online_method_accepted=False,
        next_dependency='new final-refit TopK completed task and independent full byte/event/factor admission',
        next_remaining_method_gate='conditional Bayes output selector plus separate full causal/cache203/fresh-state gate',
        checked_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
    new(receipt, final); register(receipt, final['kind'])
    print(json.dumps(final), flush=True)


if __name__ == '__main__': main()
