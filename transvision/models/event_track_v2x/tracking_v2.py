"""Causal clean-link V2 integration: frozen learned pairing, mixture CI, tracking.

This engineering route does not claim learned VoI, network C1-C9, or OOF/paper
qualification. State time is the official box time; commitment is 100 ms later.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import copy
import hashlib
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

from .association import chi_square_quantile
from .detection_cache_v2 import DetectionCacheV2, CLASSES, canonical
from .prediction_features import wrap_angle, transform_state_covariance
from .predicted_association_v2 import frame_features, association_hypotheses
from .fusion import covariance_intersection


@dataclass(frozen=True)
class TrackingConfigV2:
    deadline_us: int = 100000
    top_h: int = 3
    birth_score: float = .3
    prune_score: float = .05
    max_age_seconds: float = 2.
    survival_per_second: float = .95
    missed_factor: float = .75
    process_noise_per_second: float = .1
    gate_probability: float = .99
    unmatched_cost: float = 12.
    ci_weight: float = .5


def propagate(mean, covariance, delta, density=.1):
    f = np.eye(9); f[0, 7] = delta; f[1, 8] = delta
    return f @ mean, f @ covariance @ f.T + np.eye(9) * density * abs(delta)


def physical_world(frame, selected, target_us):
    """Invert the upstream MMDet legacy yaw/axes before world rotation.

Upstream output_to_nusc_box uses w,l,h and yaw=-legacy_yaw-pi/2.
The frozen association features retain their original legacy convention.
"""
    state = frame.states[selected].astype(float).copy()
    cov = frame.covariances[selected].astype(float).copy()
    state[:, [3, 4]] = state[:, [4, 3]]
    state[:, 6] = wrap_angle(-state[:, 6] - np.pi/2)
    j = np.eye(9); j[[3, 4]] = j[[4, 3]]; j[6, 6] = -1
    cov = j @ cov @ j.T
    world = {'lidar_to_world_row_rotation': np.eye(3), 'lidar_to_world_translation': np.zeros(3)}
    state, cov, _, _ = transform_state_covariance(state, cov, frame.metadata, world)
    delta = (target_us - frame.metadata['box_reference_timestamp_us']) / 1e6
    for i in range(len(state)):
        state[i], cov[i] = propagate(state[i], cov[i], delta)
    return state, cov


def ci(mean, cov, other, other_cov, weight=.5):
    other = np.array(other, dtype=float, copy=True)
    other[6] = mean[6] + float(wrap_angle(other[6] - mean[6]))
    result = covariance_intersection(mean, cov, other, other_cov, weight_first=weight)
    mean = result.mean.copy(); mean[6] = wrap_angle(mean[6])
    return mean, result.covariance.copy()


def moment_match(means, covariances, weights):
    weights = np.asarray(weights, float)
    if not len(weights) or not np.isclose(weights.sum(), 1) or np.any(weights < 0):
        raise ValueError('invalid mixture mass')
    means = np.asarray(means).copy()
    anchor = means[0, 6]
    means[:, 6] = anchor + wrap_angle(means[:, 6] - anchor)
    mean = weights @ means
    diff = means - mean
    covariance = np.einsum('k,kij->ij', weights, np.asarray(covariances)) + (diff.T * weights) @ diff
    mean[6] = wrap_angle(mean[6])
    return mean, (covariance + covariance.T)/2


class LearnedPairTrackerV2:
    """Single-sequence, append-only frame lifecycle consuming V2 directly."""
    def __init__(self, model, calibration, sequence_id, origin_us, config=None):
        if model.training or any(p.requires_grad for p in model.parameters()):
            raise ValueError('association model must be frozen and in eval mode')
        self.model, self.calibration = model, calibration
        self.sequence_id, self.origin_us = sequence_id, origin_us
        self.config = config or TrackingConfigV2()
        self.tracks, self.next_id = {}, 1
        self.last_reference_us = None
        self.commit_hash = '0'*64
        self.seen = set()

    def step(self, vehicle, infrastructure):
        if not all(isinstance(f, DetectionCacheV2) for f in [vehicle, infrastructure]):
            raise TypeError('V2 tracker cannot silently consume V1/raw-cache inputs')
        vm, im = vehicle.metadata, infrastructure.metadata
        if (vm['side'] != 'vehicle-side' or im['side'] != 'infrastructure-side'
                or vm['sequence_id'] != self.sequence_id or im['sequence_id'] != self.sequence_id
                or vm['dataset_sha256'] != im['dataset_sha256'] or vm['dataset_split'] != im['dataset_split']):
            raise ValueError('cooperative source/sequence/cohort mismatch')
        target = vm['box_reference_timestamp_us']
        if self.last_reference_us is not None and target <= self.last_reference_us:
            raise ValueError('reference frames must be strictly increasing; commits cannot be rewritten')
        identity = (vm['frame_id'], im['frame_id'])
        if identity in self.seen:
            raise ValueError('duplicate pair')
        deadline = target + self.config.deadline_us
        available = [f.available_at(deadline) for f in [vehicle, infrastructure]]
        pair_delta = (im['box_reference_timestamp_us'] - target)/1e6
        selections, features = [], []
        for f, allowed in zip([vehicle, infrastructure], available):
            if allowed:
                indices, values = frame_features(f, vm, self.origin_us, pair_delta,
                    self.calibration['sides'][f.metadata['side']])
            else:
                indices, values = np.zeros(0, dtype=np.int64), np.zeros((0, 203), np.float32)
            selections.append(indices); features.append(values)
        vi, ii = selections
        device = next(self.model.parameters()).device
        with torch.inference_mode():
            logits = self.model(*[torch.from_numpy(x[None]).to(device) for x in features])
        hypotheses = association_hypotheses(*[x[0].cpu().numpy() for x in logits],
            vehicle.class_indices[vi], infrastructure.class_indices[ii], self.config.top_h)
        vs, vc = physical_world(vehicle, vi, target)
        ins, inc = physical_world(infrastructure, ii, target)
        nodes = []
        # Marginalize the retained GLOBAL one-to-one alternatives; unmatched is
        # an explicit component, and between-hypothesis variance is retained.
        for i, raw in enumerate(vi):
            means, covs, weights, scores = [], [], [], []
            for h in hypotheses:
                partner = dict(h['pairs']).get(i)
                if partner is None:
                    mean, cov, score = vs[i], vc[i], vehicle.scores[raw]
                else:
                    mean, cov = ci(vs[i], vc[i], ins[partner], inc[partner], self.config.ci_weight)
                    score = max(vehicle.scores[raw], infrastructure.scores[ii[partner]])
                means.append(mean); covs.append(cov); weights.append(h['weight']); scores.append(score)
            mean, cov = moment_match(means, covs, weights)
            nodes.append((mean, cov, float(np.dot(weights, scores)), int(vehicle.class_indices[raw])))
        for j, raw in enumerate(ii):
            mass = sum(h['weight'] for h in hypotheses if j in h['unmatched_right'])
            if mass > 0:
                nodes.append((ins[j], inc[j], float(mass * infrastructure.scores[raw]), int(infrastructure.class_indices[raw])))
        dt = 0. if self.last_reference_us is None else (target-self.last_reference_us)/1e6
        tracks = []
        working_tracks = copy.deepcopy(self.tracks)
        next_id = self.next_id
        for tid in sorted(working_tracks):
            track = working_tracks[tid]
            track['mean'], track['cov'] = propagate(track['mean'], track['cov'], dt, self.config.process_noise_per_second)
            track['score'] *= self.config.survival_per_second**dt
            if (target-track['last_update_us'])/1e6 <= self.config.max_age_seconds and track['score'] >= self.config.prune_score:
                tracks.append(track)
        n, m = len(tracks), len(nodes)
        costs = np.full((n, m+n), np.inf)
        gate = chi_square_quantile(3, self.config.gate_probability)
        for i, tr in enumerate(tracks):
            for j, (mean, cov, score, cls) in enumerate(nodes):
                if tr['class_index'] != cls:
                    continue
                innovation = mean[:3]-tr['mean'][:3]
                # Unknown cross-correlation: 2(P+R) is a conservative innovation
                # covariance upper bound, rather than an independence assertion.
                uncertainty = 2*(cov[:3, :3]+tr['cov'][:3, :3])
                d2 = float(innovation @ np.linalg.solve(uncertainty, innovation))
                if d2 <= gate:
                    costs[i, j] = .5*d2 - np.log(max(score, 1e-12))
            costs[i, m+i] = self.config.unmatched_cost
        assigned = [] if n == 0 else list(zip(*linear_sum_assignment(costs)))
        matched = set()
        current = {}
        for i, j in assigned:
            tr = tracks[i]
            if j < m:
                mean, cov, score, cls = nodes[j]
                tr['mean'], tr['cov'] = ci(tr['mean'], tr['cov'], mean, cov, self.config.ci_weight)
                tr['score'] = max(tr['score'], score)  # no independent-Bernoulli product.
                tr['last_update_us'] = target
                eligible = [(tracks[k]['track_id'], costs[k, j]) for k in range(n) if np.isfinite(costs[k, j])]
                eligible.sort(key=lambda x: (x[1], x[0]))
                masses = np.exp(-np.asarray([x[1] for x in eligible]+[self.config.unmatched_cost]))
                masses /= masses.sum()
                keep = min(self.config.top_h, len(eligible))
                tr['identity_hypotheses'] = [{'track_id': eligible[k][0], 'probability': float(masses[k])} for k in range(keep)]
                tr['other_identity_probability'] = float(masses[keep:].sum())
                matched.add(j)
            else:
                tr['score'] *= self.config.missed_factor
            if tr['score'] >= self.config.prune_score:
                current[tr['track_id']] = tr
        for j, (mean, cov, score, cls) in enumerate(nodes):
            if j in matched or score < self.config.birth_score:
                continue
            tid = f'{self.sequence_id}:{next_id:06d}'; next_id += 1
            current[tid] = {'track_id': tid, 'class_index': cls, 'mean': mean.copy(), 'cov': cov.copy(),
                'score': score, 'last_update_us': target,
                'identity_hypotheses': [{'track_id': tid, 'probability': 1.}], 'other_identity_probability': 0.}
        predictions = []
        for tid, tr in sorted(current.items()):
            if not np.isfinite(tr['mean']).all() or not np.isfinite(tr['cov']).all():
                raise FloatingPointError('nonfinite tracker state')
            np.linalg.cholesky(tr['cov'])
            predictions.append({'track_id': tid, 'class_label': CLASSES[tr['class_index']],
                'mean': tr['mean'].tolist(), 'covariance': tr['cov'].tolist(), 'score': float(tr['score']),
                'identity_hypotheses': copy.deepcopy(tr['identity_hypotheses']), 'other_identity_probability': tr['other_identity_probability']})
        result = {'sequence_id': self.sequence_id, 'frame_id': vm['frame_id'],
            'box_reference_timestamp_us': target, 'decision_timestamp_us': deadline,
            'coordinate_frame': 'world', 'state_layout': 'gravity_xyz_length_width_height_yaw_vxy',
            'source_cache_sha256': [vehicle.digest(), infrastructure.digest()],
            'source_available': available,
            'source_information_timestamp_us': [vehicle.information_timestamp_us, infrastructure.information_timestamp_us],
            'selected_detections': [len(vi), len(ii)], 'predictions': predictions,
            'previous_commit_sha256': self.commit_hash}
        result['commit_sha256'] = hashlib.sha256(canonical(result)).hexdigest()
        self.commit_hash = result['commit_sha256']; self.last_reference_us = target
        self.seen.add(identity); self.tracks = current; self.next_id = next_id
        audit = {'sequence_id': self.sequence_id, 'frame_id': vm['frame_id'], 'hypotheses': hypotheses,
                 'posterior_interpretation': 'normalized_truncated_joint_energy_with_all_unmatched',
                 'full_posterior_omitted_mass': None, 'moment_matching': True}
        return result, audit
