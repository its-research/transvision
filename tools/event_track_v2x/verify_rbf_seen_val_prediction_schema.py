"""Software rejection checks for the target-free GPU consumer's input schema."""
import argparse
import copy
import importlib.util
import json
from pathlib import Path
import tempfile

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--producer', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    spec = importlib.util.spec_from_file_location('frozen_prediction_shard_software_probe', args.producer)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    means = np.zeros((2, 9))
    means[:, 3:6] = 1
    features = np.zeros((2, 203))
    features[:, 3:6] = means[:, 3:6] / [20, 20, 10]
    features[:, 7] = 1
    features[0, 138] = features[1, 139] = 1
    features[1, 144] = 1
    features[:, 196] = 1
    features[:, 197] = np.log(2) / 8
    features[:, 200:202] = 1
    arrays = dict(features=features, mean=means,
        covariance=np.broadcast_to(np.eye(9), (2, 9, 9)).copy(), score=np.ones(2),
        source=np.array([0, 1], np.int64), node_id=np.array(['a', 'b']), frame_id=np.array(['1', '2']),
        cache_sha256=np.array(['a'*64]*2), information_us=np.array([0, 10], np.int64),
        arrival_us=np.array([10, 20], np.int64), state_us=np.array([0, 10], np.int64),
        detection_index=np.array([0, 1], np.int64), contexts=np.array([[0, -1], [0, 1]], np.int64),
        lengths=np.array([1, 2], np.int64), decision_us=np.array([10, 20], np.int64))
    mutations = {}
    bad = copy.deepcopy(arrays)
    bad['positive'] = np.ones((2, 2), bool)
    mutations['train_supervision_fields'] = bad
    bad = copy.deepcopy(arrays)
    bad['arrival_us'][1] = 21
    mutations['future_arrival'] = bad
    bad = copy.deepcopy(arrays)
    bad['contexts'][0] = [1, -1]
    mutations['future_query_row'] = bad
    bad = copy.deepcopy(arrays)
    bad['features'][0, 0] = np.nan
    mutations['nonfinite_features'] = bad
    bad = copy.deepcopy(arrays)
    bad['contexts'][0, 1] = 0
    mutations['nonempty_context_padding'] = bad
    with tempfile.TemporaryDirectory(prefix='rbf-prediction-schema-gate-') as temporary:
        root = Path(temporary)

        def check(name, value):
            path = root / (name + '.npz')
            np.savez_compressed(path, **value)
            record = dict(bytes=path.stat().st_size, sha256=module.sha(path), nodes=2, sequence_id='software-fixture')
            return module.PredictionShard(path, record, 1)

        check('legal_target_free_arrays', arrays)
        for name, mutation in mutations.items():
            try:
                check(name, mutation)
            except (AssertionError, ValueError):
                continue
            raise AssertionError('illegal prediction input was accepted: ' + name)
    receipt = dict(kind='rbf_target_free_prediction_GPU_consumer_schema_software_rejection_gate_v1',
        producer_sha256=module.sha(args.producer), verifier_sha256=module.sha(__file__),
        software_legal_cases=1, rejected_mutations=list(mutations),
        numerical_feature_and_context_math_test=False, real_val_input_admitted=False,
        original_model_GPU_forward_executed=False, full_online_RBF_accepted=False,
        paper_performance_complete=False)
    with args.output.open('x') as stream:
        json.dump(receipt, stream, indent=2)
        stream.write('\n')
    print(json.dumps(receipt), flush=True)


if __name__ == '__main__':
    main()
