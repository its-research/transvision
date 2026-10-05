"""Fail closed before contacting the task service for a new ablation."""
import json
from rbf_nested_seen_val_v2_common import sha


def require_qualified_source(root, prepared):
    proof = json.loads((root/'source-qualification.json').read_bytes())
    assert proof['kind'] == 'rbf_recovery_off_GPU_source_qualification_v1'
    assert proof['preparation_sha256'] == sha(root/'preparation.json')
    assert proof['execution_sources'] == prepared['execution_sources']
    for name, record in proof['execution_sources'].items():
        assert sha(root/name) == record['sha256']
    assert proof['actual_deployed_sources_match_qualified_candidate'] is True
    assert proof['producer_math_AST_unchanged_except_runtime_import'] is True
    assert proof['all_three_exact_input_and_configuration_bindings'] is True
    assert proof['original_experiment_acceptance_inherited'] is False
    assert proof['full_forest_independently_accepted'] is False
    for record in proof['evidence']:
        assert sha(record['path']) == record['sha256']
    return proof
