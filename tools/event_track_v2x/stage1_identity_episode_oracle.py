"""GT-grounded identity ownership and observed, censored error intervals.

All global/future GT use is evaluator-only after the prediction stream is sealed.
No arbitrary first-observation anchor and no hidden-time continuity assumption.
"""
import numpy as np
from scipy.optimize import linear_sum_assignment


def global_owners(frame_matches, gt_ids, prediction_ids):
    """Reconstruct official Identity objective; certify each positive owner edge."""
    ng, npred = len(gt_ids), len(prediction_ids)
    gm = {k: i for i, k in enumerate(gt_ids)}; pm = {k: i for i, k in enumerate(prediction_ids)}
    if len(gm) != ng or len(pm) != npred: raise ValueError('unique sequence identities required')
    overlap = np.zeros((ng, npred), dtype=np.int64); gc = np.zeros(ng, dtype=np.int64); pc = np.zeros(npred, dtype=np.int64)
    for gs, ps, similarities in frame_matches:
        gi = np.asarray([gm[x] for x in gs], dtype=int); pi = np.asarray([pm[x] for x in ps], dtype=int)
        if len(gs) != len(set(gs)) or len(ps) != len(set(ps)): raise ValueError('duplicate original frame identity')
        if similarities.shape != (len(gs), len(ps)) or not np.isfinite(similarities).all(): raise ValueError('native similarity coverage differs')
        gc[gi] += 1; pc[pi] += 1
        left, right = np.nonzero(similarities >= .5)
        overlap[gi[left], pi[right]] += 1
    # Dummy rows/columns match the official sequence-level ID assignment.
    fn = np.zeros((ng + npred, ng + npred), dtype=np.int64); fp = np.zeros_like(fn)
    fp[ng:, :npred] = 10**10; fn[:ng, npred:] = 10**10
    for i in range(ng): fn[i, :npred] = gc[i]; fn[i, npred + i] = gc[i]
    for j in range(npred): fp[:ng, j] = pc[j]; fp[ng + j, j] = pc[j]
    fn[:ng, :npred] -= overlap; fp[:ng, :npred] -= overlap
    cost = fn + fp; rows, cols = linear_sum_assignment(cost); minimum = int(cost[rows, cols].sum())
    idfn, idfp = int(fn[rows, cols].sum()), int(fp[rows, cols].sum())
    owners = {k: dict(owner=None, status='unassigned_or_zero_overlap', positive_overlap=0, uniqueness_cost_gap=None) for k in prediction_ids}
    for i, j in zip(rows, cols):
        if i >= ng or j >= npred or overlap[i, j] == 0: continue
        other = cost.copy(); other[i, j] = 10**10
        ar, ac = linear_sum_assignment(other); gap = int(other[ar, ac].sum()) - minimum
        if gap < 0: raise ValueError('optimal ownership cost is inconsistent')
        owners[prediction_ids[j]] = dict(owner=gt_ids[i] if gap > 0 else None,
            status='unique_positive_optimal_owner' if gap > 0 else 'ambiguous_optimal_owner',
            positive_overlap=int(overlap[i, j]), uniqueness_cost_gap=gap,
            one_optimal_owner_not_used_when_ambiguous=gt_ids[i])
    return dict(owners=owners, IDTP=int(gc.sum()) - idfn, IDFN=idfn, IDFP=idfp,
                gt_detections=int(gc.sum()), predicted_detections=int(pc.sum()), assignment_minimum_cost=minimum)


def unique_correspondences(similarities, left_ids, right_ids):
    """Qualify only mutually unique >=.5 BEV-IoU edges; leave all others unknown."""
    matrix = np.asarray(similarities)
    if matrix.shape != (len(left_ids), len(right_ids)): raise ValueError('correspondence frame coverage differs')
    if not np.isfinite(matrix).all() or np.any(matrix < 0) or np.any(matrix > 1 + 1e-12): raise ValueError('invalid native similarities')
    mask = matrix >= .5; counts_l, counts_r = mask.sum(1), mask.sum(0)
    result = {}
    for i, lid in enumerate(left_ids):
        candidates = np.flatnonzero(mask[i])
        if len(candidates) == 1 and counts_r[candidates[0]] == 1:
            j = int(candidates[0]); result[lid] = dict(status='unique', other=right_ids[j], IoU=float(matrix[i, j]))
        else:
            result[lid] = dict(status='no_geometric_correspondence' if len(candidates) == 0 else 'ambiguous_geometric_correspondence', other=None, candidate_count=len(candidates))
    return result


def identity_label(correspondence, owners, gt_id):
    if correspondence['status'] != 'unique': return dict(label='unknown', reason=correspondence['status'])
    prediction_id = correspondence['other']; owner = owners[prediction_id]
    if owner['status'] != 'unique_positive_optimal_owner':
        return dict(label='unknown', reason=owner['status'], prediction_id=prediction_id)
    return dict(label='correct' if owner['owner'] == gt_id else 'wrong_under_unique_global_ID_assignment',
                prediction_id=prediction_id, global_GT_owner=owner['owner'], ownership_cost_gap=owner['uniqueness_cost_gap'], IoU=correspondence['IoU'])


class Episodes:
    """Observed consecutive-native-frame errors, with explicit censoring."""
    def __init__(self): self.active = {}; self.finished = []

    def close(self, gt_id, time, reason, fully_observed_boundary=False):
        item = self.active.pop(gt_id)
        item.update(end_reason=reason, boundary_timestamp_us=time,
                    observed_span_us=item['last_timestamp_us'] - item['first_timestamp_us'],
                    right_censored=not fully_observed_boundary,
                    possible_end_time_upper_us=time if fully_observed_boundary else None,
                    hidden_time_continuity_claimed=False)
        self.finished.append(item)

    def step(self, sid, frame_index, timestamp, labels, links=None):
        links = links or {}
        for gid in list(self.active):
            current = labels.get(gid); previous = self.active[gid]
            if current is None: self.close(gid, timestamp, 'GT_not_present_or_outside_ROI')
            elif current['label'] != 'wrong_under_unique_global_ID_assignment':
                self.close(gid, timestamp, current.get('reason', current['label']), current['label'] == 'correct')
            elif current['prediction_id'] != previous['prediction_id'] or current['global_GT_owner'] != previous['global_GT_owner']:
                self.close(gid, timestamp, 'wrong_prediction_or_global_owner_changed')
        for gid, label in labels.items():
            if label['label'] != 'wrong_under_unique_global_ID_assignment': continue
            if gid not in self.active:
                self.active[gid] = dict(sequence_id=sid, GT_identity=gid, prediction_id=label['prediction_id'],
                    global_GT_owner=label['global_GT_owner'], first_frame_index=frame_index, last_frame_index=frame_index,
                    first_timestamp_us=timestamp, last_timestamp_us=timestamp, observed_wrong_frames=1,
                    observed_native_intervals_us=[], event_links=[links.get(gid)], left_boundary_before_first_wrong_not_observed=True)
            else:
                item = self.active[gid]
                if frame_index != item['last_frame_index'] + 1 or timestamp <= item['last_timestamp_us']:
                    raise ValueError('native frame continuity/timestamp differs')
                item['observed_native_intervals_us'].append(timestamp - item['last_timestamp_us'])
                item['last_frame_index'], item['last_timestamp_us'] = frame_index, timestamp
                item['observed_wrong_frames'] += 1; item['event_links'].append(links.get(gid))

    def end_sequence(self):
        for gid in list(self.active): self.close(gid, None, 'sequence_ended')

