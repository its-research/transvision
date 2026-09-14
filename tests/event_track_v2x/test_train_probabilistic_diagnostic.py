"""Real fresh-process V2 fixtures; no fixture result is a paper experiment."""
from dataclasses import asdict
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from tools.event_track_v2x import run_train_probabilistic_diagnostic as tool
from tools.event_track_v2x.audit_train_inference_comparison import inspect
from tools.event_track_v2x.train_forest_identity import FitConfig, fit_dataset
from transvision.models.event_track_v2x.detection_cache_v2 import sha_file
from transvision.models.event_track_v2x.resource_sweep import THREAD_ENV, configuration
from test_forest_training_data import prepared_rows
from test_train_inference_diagnostic import pair_file

ROOT = Path(__file__).resolve().parents[2]


def test_three_fixed_updaters_in_fresh_processes_share_actual_node_beam_factors(prepared_rows, tmp_path):
    data, _, cache, rows = prepared_rows
    pair, _ = pair_file(tmp_path, rows)
    fitted = fit_dataset(data, sha_file(data/'manifest.json'), tmp_path/'fit',
        config=FitConfig(epochs=1, batch_size=4, hidden=8, heads=2, dropout=0.), require_full_train=False)
    cp = fitted['seeds'][0]
    checkpoint = (tmp_path/'fit'/cp['checkpoint_manifest']).parent
    common = [str(cache.root), cache.manifest_sha256, str(pair), sha_file(pair),
              str(checkpoint), cp['checkpoint_sha256'], rows[0]['sequence_id']]
    code = '''import sys
from transvision.models.event_track_v2x.forest_cache_stream import VerifiedForestCache
from tools.event_track_v2x.run_train_probabilistic_diagnostic import run
c,h,m,mh,p,ph,s,b,o=sys.argv[1:]
run(VerifiedForestCache(c,h),m,mh,p,ph,o,sequence=s,backend=b,allow_fixture=True)
'''
    reference = None
    for backend in ('node_beam', *tool.BACKENDS):
        source = code.replace('run_train_probabilistic_diagnostic', 'run_train_inference_diagnostic') if backend=='node_beam' else code
        output = tmp_path/backend
        process = subprocess.run([sys.executable, '-c', source, *common, backend, str(output)],
            cwd=ROOT, env=dict(os.environ, **THREAD_ENV), capture_output=True, text=True, timeout=60)
        assert process.returncode == 0, process.stdout+process.stderr
        report = inspect(output, sha_file(output/'development-inference-receipt.json'), allow_fixture=True)
        if reference is None:
            reference = report
            continue
        assert report['factor_stream_sha256'] == reference['factor_stream_sha256']
        assert report['plan']['selected_schedule'] == reference['plan']['selected_schedule']
        assert report['plan']['producer'] == tool.PRODUCER
        assert report['plan']['configuration'] == asdict(configuration(dict(backend=backend)))
        assert report['plan']['source_sha256']['tools/event_track_v2x/run_train_probabilistic_diagnostic.py'] == sha_file(tool.__file__)
        assert not report['plan']['same_state_time_protocol_as_recoverable']
        assert not report['plan']['recovery_only_ablation']
        receipt = json.loads((output/'receipt.json').read_bytes())
        assert receipt['probabilistic_single_history_enabled'] and receipt['learned_identity_enabled']
        assert receipt['probabilistic_association_algorithm'] == 'lbp'
        assert receipt['probabilistic_anchor_decoder'] == 'joint-map'
        assert receipt['probabilistic_update_rule'] == configuration(dict(backend=backend)).update_rule
        assert not receipt['allocation_teacher'] and not receipt['paper_eligible']
        with pytest.raises(ValueError, match='provenance'):
            inspect(output, report['receipt_sha256'])


@pytest.mark.parametrize('options', [dict(backend='joint_beam'), dict(allow_fixture='yes'),
    dict(sequence=''), dict(metadata_sha256='wrong')])
def test_preflight_refuses_unknown_backend_bad_mode_or_unsealed_cohort(tmp_path, options):
    args = dict(cache=None, cooperative_metadata='missing', metadata_sha256=tool.PAIR_SHA256,
        checkpoint='missing', checkpoint_sha256='x', output=tmp_path/'absent', sequence='0001', backend='jpda_ci')
    args.update(options)
    with pytest.raises(ValueError):
        tool.run(**args)
    assert not (tmp_path/'absent').exists()


def test_three_backend_configs_differ_only_in_declared_state_updater():
    values = []
    for backend in tool.BACKENDS:
        config = asdict(configuration(dict(backend=backend)))
        config.pop('update_rule')
        assert config['association_algorithm']=='lbp' and config['anchor_decoder']=='joint-map'
        values.append(config)
    assert values[0] == values[1] == values[2]


def test_new_producer_does_not_modify_the_older_recovery_source_binding():
    base = tool.diagnostic_sources()
    added = tool.sources()
    own = 'tools/event_track_v2x/run_train_probabilistic_diagnostic.py'
    assert added.pop(own) == sha_file(tool.__file__)
    assert added == base and own not in base
