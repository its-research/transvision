import copy
import json

import pytest

from tools.event_track_v2x import audit_beam_teacher_equivalence as tool
from tools.event_track_v2x.run_persistent_forest_v2 import replay_rows
from tools.event_track_v2x.run_train_inference_diagnostic import diagnostic_sources
from transvision.models.event_track_v2x.detection_cache_v2 import canonical, sha_file
from transvision.models.event_track_v2x.forest_training_checkpoint import load_identity_checkpoint
from transvision.models.event_track_v2x.resource_sweep import configuration
from test_assemble_allocation_teachers import leaves
from test_forest_training_data import prepared_rows


@pytest.fixture
def completed_pair(leaves, prepared_rows, tmp_path):
    replays, metadata = leaves
    teacher = replays[0]
    left = json.loads((teacher[0] / 'plan.json').read_bytes())
    _, _, cache, rows = prepared_rows
    rows = [r for r in rows if r['sequence_id'] == left['development_sequence']]
    config = configuration(dict(backend='beam_recovery', configuration=left['configuration']))
    checkpoint = next(p.parent for p in (tmp_path / 'trained').rglob('checkpoint.json')
                      if sha_file(p) == left['identity_checkpoint_sha256'])
    scorer, _ = load_identity_checkpoint(checkpoint, left['identity_checkpoint_sha256'], config=config.state)
    plan = {k: left.get(k) for k in tool.COMMON}
    plan.update(kind='train_sequence_inference_diagnostic_plan_v1', backend='beam_recovery',
        cohort_mode='fixture-only', selected_sequence=left['development_sequence'],
        selected_schedule=rows, source_sha256=diagnostic_sources(), scorer_signature=scorer.signature)
    output = tmp_path / 'ordinary'
    replay_rows(cache, rows, output, config, plan=plan, learned_scorer=scorer)
    outer = dict(kind='train_sequence_inference_diagnostic_v1', status='complete',
        replay_receipt_sha256=sha_file(output / 'receipt.json'), plan_sha256=sha_file(output / 'plan.json'),
        backend='beam_recovery', cohort_mode='fixture-only', complete_selected_sequence_verified=False)
    (output / 'development-inference-receipt.json').write_bytes(canonical(outer))
    return teacher, (output, sha_file(output / 'development-inference-receipt.json')), metadata


def test_complete_pair_output_equivalence_and_no_fixture_promotion(completed_pair, tmp_path):
    result = tool.compare(*completed_pair, tmp_path / 'result.json', allow_fixture=True)
    assert result['frames'] == 1 and result['byte_identical_predictions']
    assert result['actual_factors_identical'] and result['output_payload_changed_frames'] == []
    assert not result['paper_eligible'] and not result['universal_probe_purity_proved']
    assert not result['all_unchosen_states_verified'] and not result['complete_selected_sequence_verified']
    with pytest.raises(ValueError, match='official full train'):
        tool.compare(*completed_pair, tmp_path / 'not-real.json')
    with pytest.raises(ValueError, match='fresh output'):
        tool.compare(*completed_pair, tmp_path / 'result.json', allow_fixture=True)
    with pytest.raises(ValueError, match='immutable runs'):
        tool.compare(*completed_pair, completed_pair[0][0] / 'inside.json', allow_fixture=True)


@pytest.mark.parametrize('bad', ['config', 'poses', 'model', 'sequence', 'signature', 'sources', 'events', 'factors'])
def test_mixed_intervention_is_rejected(completed_pair, tmp_path, monkeypatch, bad):
    teacher, inference, metadata = completed_pair
    report = copy.deepcopy(tool.inspect(*inference, allow_fixture=True))
    if bad == 'config': report['plan']['configuration']['recovery_budget'] += 1
    elif bad == 'poses': report['plan']['ego_pose_table_sha256'] = 'a' * 64
    elif bad == 'model': report['plan']['identity_checkpoint_sha256'] = 'b' * 64
    elif bad == 'sequence': report['plan']['selected_sequence'] = 'wrong'
    elif bad == 'signature': report['plan']['scorer_signature'] = 'wrong'
    elif bad == 'sources': report['plan']['source_sha256'] = {}
    elif bad == 'events': report['events'][0]['decision_us'] += 1
    else: report['factor_stream_sha256'] = 'different'
    monkeypatch.setattr(tool, 'inspect', lambda *args, **kwargs: report)
    with pytest.raises(ValueError):
        tool.compare(teacher, inference, metadata, tmp_path / 'invalid.json', allow_fixture=True)
    assert not (tmp_path / 'invalid.json').exists()


def test_negative_output_finding_is_saved_not_silently_discarded(completed_pair, tmp_path, monkeypatch):
    teacher, inference, metadata = completed_pair
    report = copy.deepcopy(tool.inspect(*inference, allow_fixture=True))
    report['predictions_sha256'] = 'a' * 64
    report['events'][0]['output_payload_sha256'] = 'b' * 64
    report['events'][0]['output_ids_sha256'] = 'c' * 64
    monkeypatch.setattr(tool, 'inspect', lambda *args, **kwargs: report)
    result = tool.compare(teacher, inference, metadata, tmp_path / 'negative.json', allow_fixture=True)
    assert not result['byte_identical_predictions']
    assert result['output_id_changed_frames'] == result['output_payload_changed_frames']
    assert len(result['output_payload_changed_frames']) == 1
    assert (tmp_path / 'negative.json').is_file()


def test_new_unconsumed_source_does_not_invalidate_sealed_past_run(completed_pair,tmp_path,monkeypatch):
    current=tool.diagnostic_sources()
    current['transvision/models/event_track_v2x/new_unconsumed_module.py']='a'*64
    monkeypatch.setattr(tool,'diagnostic_sources',lambda:current)
    result=tool.compare(*completed_pair,tmp_path/'added-module.json',allow_fixture=True)
    assert result['byte_identical_predictions']
