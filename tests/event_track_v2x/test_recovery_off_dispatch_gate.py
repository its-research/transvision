"""Do not launch an expensive replay when admission is absent or stale."""
import json
from pathlib import Path
import sys
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]/'tools/event_track_v2x'))
from recovery_off_dispatch_gate import require_qualified_source
from rbf_nested_seen_val_v2_common import sha


def setup(root):
    (root/'bootstrap.py').write_text('original source')
    evidence = root/'input-proof.json'
    evidence.write_text('real input binding')
    prepared = dict(execution_sources={'bootstrap.py': dict(sha256=sha(root/'bootstrap.py'))})
    (root/'preparation.json').write_text(json.dumps(prepared))
    proof = dict(kind='rbf_recovery_off_GPU_source_qualification_v1',
        preparation_sha256=sha(root/'preparation.json'), execution_sources=prepared['execution_sources'],
        actual_deployed_sources_match_qualified_candidate=True,
        producer_math_AST_unchanged_except_runtime_import=True,
        all_three_exact_input_and_configuration_bindings=True,
        original_experiment_acceptance_inherited=False, full_forest_independently_accepted=False,
        evidence=[dict(path=str(evidence),sha256=sha(evidence))])
    (root/'source-qualification.json').write_text(json.dumps(proof))
    return prepared, proof


def test_qualified_candidate_passes_without_claiming_full_acceptance(tmp_path):
    prepared, _ = setup(tmp_path)
    assert require_qualified_source(tmp_path,prepared)['full_forest_independently_accepted'] is False


@pytest.mark.parametrize('name', ['bootstrap.py','input-proof.json','preparation.json'])
def test_post_qualification_mutation_is_rejected(tmp_path, name):
    prepared, _ = setup(tmp_path)
    with (tmp_path/name).open('a') as stream: stream.write('changed')
    with pytest.raises(AssertionError): require_qualified_source(tmp_path,prepared)


def test_missing_qualification_is_not_permission_to_run(tmp_path):
    prepared, _ = setup(tmp_path)
    (tmp_path/'source-qualification.json').unlink()
    with pytest.raises(FileNotFoundError): require_qualified_source(tmp_path,prepared)


def test_parent_forest_acceptance_cannot_transfer(tmp_path):
    prepared, proof = setup(tmp_path)
    proof['original_experiment_acceptance_inherited'] = True
    (tmp_path/'source-qualification.json').write_text(json.dumps(proof))
    with pytest.raises(AssertionError): require_qualified_source(tmp_path,prepared)
