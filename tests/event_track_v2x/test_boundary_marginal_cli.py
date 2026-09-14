import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


def test_independent_process_boundary_replay_and_create_once(tmp_path):
    root = Path(__file__).resolve().parents[2]
    command = [sys.executable, str(root/'tools/event_track_v2x/verify_boundary_marginal.py'),
               '--trials', '4', '--seeds', '1337', '2027']
    environment = {**os.environ, 'PYTHONDONTWRITEBYTECODE': '1', 'OPENBLAS_NUM_THREADS': '1',
                   'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1', 'CUDA_VISIBLE_DEVICES': ''}
    outputs = []
    for name in ('first.json', 'second.json'):
        output = tmp_path/name
        completed = subprocess.run(command+['--output', str(output)], cwd=root, env=environment,
                                   capture_output=True, text=True, timeout=60)
        assert completed.returncode == 0, completed.stderr
        outputs.append(output.read_bytes())
    assert outputs[0] == outputs[1]
    report = json.loads(outputs[0])
    assert report['case_count'] == 8 and report['status'] == 'verified'
    assert report['maximum_probability_error'] < 2e-12
    assert report['maximum_identity_risk_error'] < 2e-12
    scope = report['scope']
    assert scope['synthetic_only'] and scope['class_scope'] == ['car']
    assert not scope['bounded_memory_long_sequence_inference_completed']
    assert not scope['real_data_or_test_read'] and not scope['paper_tracking_performance_evidence']
    event = report['future_amplification_and_recovery']
    assert event['old_omitted_mass'] == .001 and event['updated_omitted_mass_decimal_reference'] > .999
    assert event['normalized_root_identity_regret'] > .4
    assert event['same_legal_action_space'] and event['restored_outside_old_active']
    assert event['historical_output_unchanged'] and event['conditional_state_replay_equal']
    for path, expected in report['source_hashes'].items():
        assert hashlib.sha256((root/path).read_bytes()).hexdigest() == expected
    repeat = subprocess.run(command+['--output', str(tmp_path/'first.json')], cwd=root, env=environment,
                            capture_output=True, text=True, timeout=60)
    assert repeat.returncode != 0 and 'create-once' in repeat.stderr
    assert (tmp_path/'first.json').read_bytes() == outputs[0]
