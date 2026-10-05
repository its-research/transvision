"""Freeze the qualified conditional selector and strict new-model cohort gate."""
import ast
import copy
import datetime
import importlib.util
import json
from pathlib import Path
import shutil
import sys

from rbf_nested_seen_val_v2_common import R, new, register, sha

D = R/'source-freezes/rbf-independent-fixed-Top1-conditional-selector-v1-20261004'
QUAL = R/'artifacts/rbf-independent-fixed-Top1-selector-qualification-v1-20261004/qualification.json'
SEARCH_SOURCE = R/'source-freezes/rbf-independent-fixed-Top1-search-pruning-v1-20261004'


def main():
    assert not D.exists(), 'immutable source freeze already exists'
    directory=Path(__file__).resolve().parent
    qualification=json.loads(QUAL.read_bytes())
    assert qualification['negative_controls_rejected']==22
    assert qualification['software_qualification_only'] is True and qualification['new_final_model_experiment_accepted'] is False
    assert qualification['oracle_sha256']==sha(directory/'independent_fixed_top1_selector.py')
    assert qualification['qualification_source_sha256']==sha(directory/'qualify_independent_fixed_top1_selector.py')
    names=('independent_fixed_top1_selector.py','qualify_independent_fixed_top1_selector.py',
           'accept_final_refit_top1_selector.py','prepare_independent_fixed_top1_selector.py',
           'rbf_nested_seen_val_v2_common.py')
    for name in names: ast.parse((directory/name).read_bytes())
    spec=importlib.util.spec_from_file_location('candidate_final_topK_selector_driver',directory/'accept_final_refit_top1_selector.py')
    driver=importlib.util.module_from_spec(spec);spec.loader.exec_module(driver)
    sys.path.insert(0,str(SEARCH_SOURCE))
    from rbf_final_refit_forest_binding import INDEX,validate_final_model
    source_binding=dict(unchanged_search_source_binding=driver.search_source_gate())
    entries=json.loads(INDEX.read_bytes())['seeds'];fixtures=[];rejected=[]
    for entry in entries:
        seed=entry['seed'];checkpoint=Path(entry['training_byte_proof']).parent/'checkpoint'
        model=validate_final_model(seed,checkpoint)
        value=dict(task_id='d'*32,seed=seed,recipe_sha256='a'*64,total_nodes=model['rows'])
        search=dict(kind='rbf_final_refit_fixed_Top1_full_search_pruning_admission_v1',method='topk',K=1,
            task_id=value['task_id'],seed=seed,recipe_sha256=value['recipe_sha256'],
            final_model_binding=model,source_binding=source_binding['unchanged_search_source_binding'],
            completed_sequences=46,completed_events=7445,observations=model['rows'],node_pruning_steps=model['rows'],
            all_final_model_cohort_search_pruning_independently_verified=True,atol=1e-8,rtol=1e-8,
            old_model_search_or_forest_acceptance_inherited=False,conditional_output_selector_independently_accepted=False,
            complete_online_method_accepted=False,same_resource_performance_accepted=False,paper_performance_complete=False)
        driver.validate_local_search(search,value,model,source_binding)
        fixtures.append(dict(seed=seed,positive_shape_fixture_passed=True,real_search_or_selector_admission=False))
        controls={
            'foreign_K4_search_kind':lambda v:v.update(kind='rbf_final_refit_fixed_topK_full_search_pruning_admission_v1'),
            'foreign_K4_width':lambda v:v.update(K=4),
            'software_sequence_result':lambda v:v.update(kind='rbf_independent_fixed_Top1_raw_graph_search_pruning_audit_v1'),
            'old_checkpoint_binding':lambda v:v['final_model_binding'].update(final_model_sha256=model['original_model_sha256']),
            'wrong_task':lambda v:v.update(task_id='e'*32),
            'wrong_recipe':lambda v:v.update(recipe_sha256='b'*64),
            'partial_sequences':lambda v:v.update(completed_sequences=45),
            'partial_events':lambda v:v.update(completed_events=7444),
            'partial_nodes':lambda v:v.update(observations=model['rows']-1),
            'missing_search_success':lambda v:v.update(all_final_model_cohort_search_pruning_independently_verified=False),
            'foreign_source_binding':lambda v:v.update(source_binding={}),
            'widened_tolerance':lambda v:v.update(atol=1e-6),
            'old_search_acceptance_inherited':lambda v:v.update(old_model_search_or_forest_acceptance_inherited=True),
            'premature_selector':lambda v:v.update(conditional_output_selector_independently_accepted=True),
            'premature_whole_method':lambda v:v.update(complete_online_method_accepted=True),
            'premature_resource_metric':lambda v:v.update(same_resource_performance_accepted=True),
            'premature_paper_metric':lambda v:v.update(paper_performance_complete=True),
        }
        for name,edit in controls.items():
            altered=copy.deepcopy(search);edit(altered)
            try:driver.validate_local_search(altered,value,model,source_binding)
            except AssertionError:rejected.append(dict(seed=seed,control=name))
            else:raise AssertionError('selector cohort fixture admitted: '+name)
    references=[SEARCH_SOURCE/'source-freeze.json',
        R/'source-freezes/rbf-independent-fixed-topK-conditional-selector-v1-20261004/source-freeze.json',
        R/'receipts/rbf-independent-fixed-Top1-conditional-selector-source-derivation-20261004.json',
        R/'source-freezes/rbf-final-refit-fixed-Top1-full-independent-CPU-v1-20261004/source-freeze.json',INDEX,
        Path(qualification['database_path']),Path(qualification['search_proof_path'])]
    D.mkdir()
    for name in names:shutil.copyfile(directory/name,D/name)
    freeze=dict(kind='rbf_independent_fixed_Top1_conditional_selector_source_v1',
        sources={name:dict(sha256=sha(D/name),bytes=(D/name).stat().st_size) for name in names},
        unchanged_references=[dict(path=str(p),sha256=sha(p)) for p in references],
        qualification=dict(path=str(QUAL),sha256=sha(QUAL)),
        checked_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        model_metadata_shape_fixtures=fixtures,rejected_cohort_shape_controls=rejected,
        actual_completed_search_and_same_byte_domain_required=True,search_pruning_rerun=False,
        fixed_float64_comparison_tolerance=1e-8,independent_pairwise_Decimal_precision=70,
        new_final_model_experiment_accepted=False,full_online_method_accepted=False,paper_performance_complete=False)
    new(D/'source-freeze.json',freeze)
    path=R/'receipts/rbf-independent-fixed-Top1-conditional-selector-preparation-20261004.json'
    final=dict(kind='rbf_independent_fixed_Top1_conditional_selector_preparation_v1',
        source_freeze_path=str(D/'source-freeze.json'),source_freeze_sha256=sha(D/'source-freeze.json'),
        qualification_path=str(QUAL),qualification_sha256=sha(QUAL),
        decisions_qualified=229,queries_qualified=4794,MAP_and_Bayes_differences=0,
        semantic_negative_controls_rejected=22,model_cohort_negative_shape_controls_rejected=len(rejected),
        max_abs_error=qualification['max_abs_error'],atol=1e-8,rtol=1e-8,
        software_qualification_only=True,new_final_model_experiment_accepted=False,
        next_dependency='completed new Top-1 full byte/factor readback and independently successful full search/pruning gate',
        complementary_gate='existing separately frozen causal/cache203/fresh continuous state CPU admission',
        full_online_method_accepted=False,same_resource_performance_accepted=False,paper_performance_complete=False,
        checked_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
    new(path,final);register(path,final['kind']);print(json.dumps(final),flush=True)


if __name__=='__main__':main()
