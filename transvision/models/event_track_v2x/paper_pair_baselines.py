"""Paper-protocol adapters for the actual frozen learned+CI and M0--M4 code.

These are clean-link paired-frame controls, not recoverable forest aliases. The original numerical steps are reused via instance-local feature hooks.
"""
import copy
import json
from dataclasses import asdict
from pathlib import Path
from types import FunctionType, MethodType

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

from .detection_cache_v2 import contained_file, sha_file
from .forest_training_data import frozen_cache_identity
from .paper_protocol import PAPER, SEEDS, PaperProtocol
from .predicted_association_v2 import PredictedAssociation, association_hypotheses, k_best_assignments
from .prediction_features import encode_features
from .tracking_birth_score_v2 import BirthScoreDiagnosticTrackerV2
from .tracking_mechanisms_v2 import MechanismDiagnosticTrackerV2
from .tracking_v2 import LearnedPairTrackerV2, TrackingConfigV2

METHODS = ('learned-ci', 'M0', 'M1', 'M2', 'M3', 'M4')
FEATURE_RECIPE = 'rbf-all-class-pair-203-v1'


class NoPairing(torch.nn.Module):
    """M1 is defined to never evaluate an association model."""

    def __init__(self):
        super().__init__()
        self.eval()

    def forward(self, *inputs):
        raise AssertionError('M1 must use the explicit all-unmatched hypothesis')


def load_assets(checkpoint, checkpoint_sha256, calibration_path, calibration_sha256, *, cache, protocol, fixture, method):
    if sha_file(calibration_path) != calibration_sha256:
        raise ValueError('pair calibration changed')
    calibration = json.loads(Path(calibration_path).read_bytes())
    producer = frozen_cache_identity(cache)
    if (type(fixture) is not bool or type(calibration.get('fixture')) is not bool or calibration.get('kind') not in {'rbf_pair_calibration_v1', 'eventtrack_train_calibration_v1'}
            or calibration.get('fit_split') != 'train' or calibration.get('dataset') != protocol.dataset or calibration.get('candidate_protocol') != PAPER
            or calibration.get('fixture') != fixture or producer['calibration_sha256'] != calibration_sha256):
        raise ValueError('pair calibration dataset/train/protocol/cache binding differs')
    for side in ('vehicle-side', 'infrastructure-side'):
        for category in ('car', 'bicycle', 'pedestrian'):
            score = calibration['sides'][side][category]['score']
            if (not all(np.isfinite(score[k]) for k in ('slope', 'intercept', 'logit_clip')) or score['slope'] < 0 or not 0 < score['logit_clip'] < .5):
                raise ValueError('invalid frozen class score calibration')
    binding = dict(
        dataset=protocol.dataset,
        candidate_protocol=PAPER,
        feature_recipe=FEATURE_RECIPE,
        fit_split='train',
        calibration_sha256=calibration_sha256,
        frozen_cache_identity=producer,
        seed=None,
        deterministic=method == 'M1',
        fixture=fixture)
    if method == 'M1':
        if checkpoint is not None or checkpoint_sha256 is not None:
            raise ValueError('M1 has no learned model or training-seed selection')
        return NoPairing(), calibration, binding
    if checkpoint is None or checkpoint_sha256 is None:
        raise ValueError('pair checkpoint required; identity-forest weights are not interchangeable')
    checkpoint = Path(checkpoint)
    if sha_file(checkpoint) != checkpoint_sha256:
        raise ValueError('pair checkpoint manifest changed')
    manifest = json.loads(checkpoint.read_bytes())
    if (manifest.get('kind') != 'rbf_pair_checkpoint_v1' or manifest.get('dataset') != protocol.dataset or manifest.get('fit_split') != 'train'
            or manifest.get('candidate_protocol') != PAPER or manifest.get('feature_recipe') != FEATURE_RECIPE or manifest.get('seed') not in SEEDS
            or manifest.get('calibration_sha256') != calibration_sha256 or manifest.get('frozen_cache_identity') != producer or type(manifest.get('fixture')) is not bool
            or manifest.get('fixture') != fixture):
        raise ValueError('pair checkpoint training/protocol/producer binding differs')
    training = manifest['training_receipt']
    training_path = contained_file(checkpoint.parent, training['path'])
    if sha_file(training_path) != training['sha256']:
        raise ValueError('pair training evidence changed')
    training_record = json.loads(training_path.read_bytes())
    if (training_record.get('fit_split') != 'train' or training_record.get('dataset') != protocol.dataset or training_record.get('candidate_protocol') != PAPER
            or training_record.get('fixture') != fixture or training_record.get('seed') != manifest['seed']
            or training_record.get('weights_sha256') != manifest['weights']['sha256']):
        raise ValueError('pair training receipt does not bind the frozen weights')
    weights = contained_file(checkpoint.parent, manifest['weights']['path'])
    if sha_file(weights) != manifest['weights']['sha256']:
        raise ValueError('pair weights changed')
    state = torch.load(weights, map_location='cpu', weights_only=True)
    if not isinstance(state, dict) or any(not isinstance(v, torch.Tensor) or not torch.isfinite(v).all() for v in state.values()):
        raise ValueError('finite tensor-only pair checkpoint required')
    model = PredictedAssociation(**manifest['architecture'])
    model.load_state_dict(state, strict=True)
    model.requires_grad_(False).eval()
    if sha_file(weights) != manifest['weights']['sha256']:
        raise ValueError('pair weights changed during load')
    binding.update(seed=manifest['seed'], checkpoint_sha256=checkpoint_sha256, training_receipt_sha256=training['sha256'], weights_sha256=manifest['weights']['sha256'])
    return model, calibration, binding


def _instance_function(function, overrides):
    """Bind reviewed dependencies without editing source or module globals.

    Reuses the exact original code object. No source text, eval, user-defined code, or process-wide monkeypatch is involved; each tracker owns its map.
    """
    if not set(overrides) <= set(function.__globals__):
        raise ValueError('sealed function dependency names changed')
    namespace = dict(function.__globals__, **overrides)
    result = FunctionType(function.__code__, namespace, function.__name__, function.__defaults__, function.__closure__)
    result.__kwdefaults__ = copy.copy(function.__kwdefaults__)
    return result


def make_tracker(method, model, calibration, sequence_id, origin_us, *, protocol, binding, config=None, counters=None):
    if method not in METHODS or type(protocol) is not PaperProtocol or protocol.candidates != PAPER:
        raise ValueError('explicit paper pair method/protocol required')
    if (binding.get('dataset') != protocol.dataset or binding.get('candidate_protocol') != PAPER or binding.get('fit_split') != 'train'):
        raise ValueError('pair model is not bound to this dataset/protocol')
    config = config or TrackingConfigV2()
    if (type(config) is not TrackingConfigV2 or config.deadline_us != 100_000 or type(config.top_h) is not int or not 1 <= config.top_h <= 64
            or not all(np.isfinite(v) for v in asdict(config).values()) or not 0 <= config.prune_score <= config.birth_score <= 1 or not 0 < config.survival_per_second <= 1
            or not 0 <= config.missed_factor <= 1 or config.max_age_seconds < 0 or config.process_noise_per_second < 0 or not 0 < config.gate_probability < 1
            or config.unmatched_cost < 0 or not 0 < config.ci_weight < 1):
        raise ValueError('invalid frozen clean-link tracking configuration')
    costs = counters if counters is not None else {}

    def features(frame, reference, origin, delta, side_calibration):
        if (frame.metadata['dataset_split'] != protocol.split or frame.metadata['calibration_sha256'] != binding['calibration_sha256']):
            raise ValueError('frame split or pair calibration differs')
        indices = protocol.select(frame.raw_scores, frame.class_indices)
        arrays = dict(scores=frame.raw_scores, class_indices=frame.class_indices, appearance_128=frame.appearance, appearance_valid=frame.appearance_valid)
        values = encode_features(frame.states, frame.covariances, arrays, indices, frame.metadata, reference, delta, origin, side_calibration)
        costs['encoded_detections'] = costs.get('encoded_detections', 0) + len(indices)
        return indices, values

    def solve(matrix):
        costs['temporal_assignment_solves'] = costs.get('temporal_assignment_solves', 0) + 1
        return linear_sum_assignment(matrix)

    def cross_solve(matrix):
        costs['cross_assignment_solves'] = costs.get('cross_assignment_solves', 0) + 1
        return linear_sum_assignment(matrix)

    best = _instance_function(k_best_assignments, {'linear_sum_assignment': cross_solve})
    hypotheses = _instance_function(association_hypotheses, {'k_best_assignments': best})
    if method == 'learned-ci':
        tracker = LearnedPairTrackerV2(model, calibration, sequence_id, origin_us, config)
    elif method == 'M4':
        tracker = BirthScoreDiagnosticTrackerV2(model, calibration, sequence_id, origin_us, config)
    else:
        tracker = MechanismDiagnosticTrackerV2(model, calibration, sequence_id, origin_us, config, mode=method)
    tracker.step = MethodType(_instance_function(type(tracker).step, dict(frame_features=features, association_hypotheses=hypotheses, linear_sum_assignment=solve)), tracker)
    tracker.paper_binding = copy.deepcopy(dict(protocol=asdict(protocol), model=binding, method=method, config=asdict(config)))
    return tracker


def snapshot(tracker):
    tracks = [{k: v.tolist() if isinstance(v, np.ndarray) else copy.deepcopy(v) for k, v in t.items()} for _, t in sorted(tracker.tracks.items())]
    return dict(
        kind='rbf_pair_state_v1',
        binding=copy.deepcopy(tracker.paper_binding),
        sequence_id=tracker.sequence_id,
        origin_us=tracker.origin_us,
        tracks=tracks,
        next_id=tracker.next_id,
        last_reference_us=tracker.last_reference_us,
        commit_hash=tracker.commit_hash,
        diagnostic_commit_hash=getattr(tracker, 'diagnostic_commit_hash', None),
        seen=sorted(tracker.seen))


def restore(tracker, path, expected_sha256):
    if tracker.tracks or tracker.last_reference_us is not None:
        raise ValueError('restore only into an unused tracker')
    if sha_file(path) != expected_sha256:
        raise ValueError('pair state changed')
    state = json.loads(Path(path).read_bytes())
    if (state['kind'] != 'rbf_pair_state_v1' or state['binding'] != tracker.paper_binding or state['sequence_id'] != tracker.sequence_id
            or state['origin_us'] != tracker.origin_us):
        raise ValueError('pair state binding differs')
    tracks = {}
    for raw in state['tracks']:
        track = dict(raw, mean=np.asarray(raw['mean'], float), cov=np.asarray(raw['cov'], float))
        if (track['track_id'] in tracks or track['mean'].shape != (9, ) or track['cov'].shape != (9, 9) or not np.isfinite(track['mean']).all()
                or not np.isfinite(track['cov']).all()):
            raise ValueError('invalid persisted pair track')
        np.linalg.cholesky(track['cov'])
        tracks[track['track_id']] = track
    tracker.tracks, tracker.next_id = tracks, state['next_id']
    tracker.last_reference_us, tracker.commit_hash = state['last_reference_us'], state['commit_hash']
    tracker.seen = {tuple(x) for x in state['seen']}
    if state['diagnostic_commit_hash'] is not None:
        tracker.diagnostic_commit_hash = state['diagnostic_commit_hash']
    return tracker
