"""Strict M4 birth-score-only supplement to the frozen M0 diagnostic path.

The parent step is monolithic. Its arithmetic is retained here, with a separate
birth-only score list; neither temporal node scores nor existing-track update
expressions consume that list. Future free-running states can differ after a
birth, so this is not a claim of temporal-output parity across whole sequences.
"""
from __future__ import annotations

import copy
import hashlib
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

from .association import chi_square_quantile
from .detection_cache_v2 import DetectionCacheV2, CLASSES, canonical
from .predicted_association_v2 import frame_features, association_hypotheses
from .tracking_v2 import propagate, physical_world, ci, moment_match
from .tracking_mechanisms_v2 import MechanismDiagnosticTrackerV2, state_record


class BirthScoreDiagnosticTrackerV2(MechanismDiagnosticTrackerV2):
    """M0 common-score semantics with original road score only at birth."""
    def __init__(self, model, calibration, sequence_id, origin_us, config=None, *, plan_sha256=None):
        super().__init__(model, calibration, sequence_id, origin_us, config,
                         mode='M0', plan_sha256=plan_sha256)

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
        if self.mode == 'M1':
            hypotheses = [{'pairs': [], 'unmatched_left': list(range(len(vi))),
                'unmatched_right': list(range(len(ii))), 'weight': 1., 'energy': 0.}]
        else:
            device = next(self.model.parameters()).device
            with torch.inference_mode():
                logits = self.model(*[torch.from_numpy(x[None]).to(device) for x in features])
            hypotheses = association_hypotheses(*[x[0].cpu().numpy() for x in logits],
                vehicle.class_indices[vi], infrastructure.class_indices[ii], self.config.top_h)
        vs, vc = physical_world(vehicle, vi, target)
        ins, inc = physical_world(infrastructure, ii, target)
        nodes = []
        birth_scores = []
        node_records = []
        events = []
        temporal = []
        source_records = []
        for side, source, selected, means, covs in [
                ('vehicle-side', vehicle, vi, vs, vc),
                ('infrastructure-side', infrastructure, ii, ins, inc)]:
            for local, raw in enumerate(selected):
                source_records.append({'side': side, 'frame_id': source.metadata['frame_id'],
                    'selected_index': local, 'raw_index': int(raw),
                    'raw_score': float(source.raw_scores[raw]), 'score': float(source.scores[raw]),
                    'class_index': int(source.class_indices[raw]), **state_record(means[local], covs[local])})
        # Marginalize the retained GLOBAL one-to-one alternatives; unmatched is
        # an explicit component, and between-hypothesis variance is retained.
        for i, raw in enumerate(vi):
            means, covs, weights, scores = [], [], [], []
            components = []
            for h in hypotheses:
                partner = dict(h['pairs']).get(i)
                if partner is None:
                    mean, cov, score = vs[i], vc[i], vehicle.scores[raw]
                else:
                    if self.mode == 'M2':
                        mean, cov = ins[partner], inc[partner]
                    else:
                        mean, cov = ci(vs[i], vc[i], ins[partner], inc[partner], self.config.ci_weight)
                    score = max(vehicle.scores[raw], infrastructure.scores[ii[partner]])
                components.append({'hypothesis_index': len(components), 'weight': h['weight'],
                    'partner_selected_index': partner,
                    'partner_raw_index': None if partner is None else int(ii[partner]),
                    'kind': 'vehicle_unmatched' if partner is None else
                            ('road_state_replacement' if self.mode == 'M2' else 'cross_source_ci'),
                    'score': float(score), **state_record(mean, cov)})
                means.append(mean); covs.append(cov); weights.append(h['weight']); scores.append(score)
            mean, cov = moment_match(means, covs, weights)
            nodes.append((mean, cov, float(np.dot(weights, scores)), int(vehicle.class_indices[raw])))
            birth_scores.append(nodes[-1][2])
            unmatched_mass = sum(h['weight'] for h in hypotheses if i in h['unmatched_left'])
            node_records.append({'node_index': len(nodes)-1, 'kind': 'vehicle_anchored_mixture',
                'source_side': 'vehicle-side', 'source_frame_id': vm['frame_id'],
                'source_selected_index': i, 'source_raw_index': int(raw),
                'class_index': int(vehicle.class_indices[raw]), 'score': nodes[-1][2],
                'birth_effective_score': birth_scores[-1],
                'matched_mass': sum(h['weight'] for h in hypotheses if i not in h['unmatched_left']),
                'unmatched_mass': unmatched_mass, 'components': components,
                **state_record(mean, cov)})
        for j, raw in enumerate(ii):
            mass = sum(h['weight'] for h in hypotheses if j in h['unmatched_right'])
            if mass > 0:
                score = float(infrastructure.scores[raw]) if self.mode == 'M3' else float(mass * infrastructure.scores[raw])
                nodes.append((ins[j], inc[j], score, int(infrastructure.class_indices[raw])))
                birth_scores.append(float(infrastructure.scores[raw]))
                node_records.append({'node_index': len(nodes)-1, 'kind': 'road_residual',
                    'source_side': 'infrastructure-side', 'source_frame_id': im['frame_id'],
                    'source_selected_index': j, 'source_raw_index': int(raw),
                    'class_index': int(infrastructure.class_indices[raw]),
                    'original_score': float(infrastructure.scores[raw]), 'score': score,
                    'birth_effective_score': birth_scores[-1],
                    'matched_mass': sum(h['weight'] for h in hypotheses if j not in h['unmatched_right']),
                    'unmatched_mass': mass, 'components': [], **state_record(ins[j], inc[j])})
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
            else:
                events.append({'event': 'kill_before_assignment', 'track_id': tid,
                    'score': float(track['score']),
                    'age_seconds': (target-track['last_update_us'])/1e6,
                    'max_age_exceeded': (target-track['last_update_us'])/1e6 > self.config.max_age_seconds,
                    'below_prune_score': track['score'] < self.config.prune_score})
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
                innovation = mean[:3] - tr['mean'][:3]
                uncertainty = 2*(cov[:3, :3]+tr['cov'][:3, :3])
                temporal.append({'event': 'matched', 'track_id': tr['track_id'],
                    'node_index': int(j), 'assigned_cost': float(costs[i, j]),
                    'innovation_xyz': innovation.tolist(),
                    'mahalanobis_squared': float(innovation @ np.linalg.solve(uncertainty, innovation)),
                    'prior_score': float(tr['score']), 'node_score': float(score),
                    'prior_state': state_record(tr['mean'], tr['cov']),
                    'eligible_track_costs': [[tracks[k]['track_id'], float(costs[k, j])]
                        for k in range(n) if np.isfinite(costs[k, j])]})
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
                events.append({'event': 'miss', 'track_id': tr['track_id'],
                    'assigned_cost': float(costs[i, j]), 'prior_score': float(tr['score']),
                    'feasible_nodes': int(np.isfinite(costs[i, :m]).sum())})
                tr['score'] *= self.config.missed_factor
            if tr['score'] >= self.config.prune_score:
                current[tr['track_id']] = tr
            else:
                events.append({'event': 'kill_after_miss', 'track_id': tr['track_id'],
                    'score': float(tr['score'])})
        for j, (mean, cov, score, cls) in enumerate(nodes):
            birth_score = birth_scores[j]
            if j in matched or birth_score < self.config.birth_score:
                if j not in matched:
                    events.append({'event': 'birth_rejected_score', 'node_index': j, 'score': float(birth_score),
                                   'node_score': float(score), 'birth_effective_score': float(birth_score)})
                continue
            tid = f'{self.sequence_id}:{next_id:06d}'; next_id += 1
            events.append({'event': 'birth', 'track_id': tid, 'node_index': j, 'score': float(birth_score),
                           'node_score': float(score), 'birth_effective_score': float(birth_score)})
            current[tid] = {'track_id': tid, 'class_index': cls, 'mean': mean.copy(), 'cov': cov.copy(),
                'score': birth_score, 'last_update_us': target,
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
        audit = {'sequence_id': self.sequence_id, 'frame_id': vm['frame_id'], 'hypotheses': hypotheses,
                 'posterior_interpretation': 'normalized_truncated_joint_energy_with_all_unmatched',
                 'full_posterior_omitted_mass': None, 'moment_matching': True}
        if self.mode == 'M1':
            audit['posterior_interpretation'] = 'forced_all_unmatched_control'
        diagnostic = {'kind': 'tracking_birth_score_v2_frame', 'mode': 'M4',
            'intervention': 'road residual birth threshold and initial score only; common node score unchanged',
            'plan_sha256': self.plan_sha256,
            'sequence_id': self.sequence_id, 'frame_id': vm['frame_id'],
            'box_reference_timestamp_us': target, 'decision_timestamp_us': deadline,
            'source_frame_ids': [vm['frame_id'], im['frame_id']],
            'source_cache_sha256': result['source_cache_sha256'],
            'source_available': available, 'selected_detections': result['selected_detections'],
            'source_records': source_records, 'nodes': node_records,
            'temporal_assignments': temporal, 'events': events,
            'tracks_before': len(self.tracks), 'tracks_eligible': n, 'tracks_after': len(current),
            'gate_mahalanobis_squared': float(gate),
            'prediction_commit_sha256': result['commit_sha256'],
            'association_frame_sha256': hashlib.sha256(canonical(audit)).hexdigest(),
            'previous_diagnostic_commit_sha256': self.diagnostic_commit_hash}
        diagnostic['diagnostic_commit_sha256'] = hashlib.sha256(canonical(diagnostic)).hexdigest()
        self.commit_hash = result['commit_sha256']; self.last_reference_us = target
        self.seen.add(identity); self.tracks = current; self.next_id = next_id
        self.diagnostic_commit_hash = diagnostic['diagnostic_commit_sha256']
        return result, audit, diagnostic

