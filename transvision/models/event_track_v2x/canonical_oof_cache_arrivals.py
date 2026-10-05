"""Role-bound causal V2 source arrivals for canonical OOF replay.

This is an ingestion component, not a tracker or a paper evaluation. Sources
are admitted individually, including unpaired frames and the post-evaluation
tail. A vehicle-only mask is a view of the same sealed cache.
"""
import hashlib
import json
import copy
from pathlib import Path

from .detection_cache_v2 import DetectionCacheV2, SIDES, canonical, contained_file, sha_file
from .prediction_features import choose_candidates


class CanonicalOOFCacheArrivals:
    def __init__(self, root, expected_sha256, calibration, *, calibration_sha256, fold_id, role, agent_mask=3):
        if role not in {'fit', 'held_out'} or type(agent_mask) is not int or agent_mask not in {1, 2, 3}:
            raise ValueError('explicit fold role and source mask required')
        binding = calibration['canonical_oof_binding']
        if type(fold_id) is not int or binding['fold_id'] != fold_id:
            raise ValueError('fold identity differs')
        fit, held = calibration['fit_sequences'], binding['held_out_sequence_ids']
        if set(fit) & set(held) or len(set(fit) | set(held)) != 46:
            raise ValueError('fit/held-out boundary differs')
        self.root = Path(root)
        path = contained_file(self.root, 'manifest.json')
        if sha_file(path) != expected_sha256:
            raise ValueError('sealed cache identity differs')
        self.manifest = json.loads(path.read_bytes())
        m = self.manifest
        if (m['kind'] != 'detection_cache_v2_manifest' or m['schema_version'] != 2
                or m['split'] != 'train' or m['gt_in_cache'] is not False
                or m['test_payloads_read'] is not False or path.read_bytes() != canonical(m)
                or m['calibration_sha256'] != calibration_sha256):
            raise ValueError('canonical train prediction-only V2 required')
        allowed = fit if role == 'fit' else held
        if set(m['sequences']) != set(allowed):
            raise ValueError('cache is not the full explicit fold cohort')
        self.entries = {}
        self.metadata = {}
        for entry in m['frames']:
            p = contained_file(self.root, entry['metadata']['path'])
            if sha_file(p) != entry['metadata']['sha256']:
                raise ValueError('source metadata identity differs')
            meta = json.loads(p.read_bytes())
            key = (meta['sequence_id'], meta['side'], meta['frame_id'])
            if (key in self.entries or key[0] not in allowed or key[1] not in SIDES
                    or meta['dataset_split'] != 'train' or meta['calibration_fit_split'] != 'train'
                    or meta['calibration_sha256'] != m['calibration_sha256']
                    or meta['dataset_sha256'] != m['dataset_sha256']
                    or meta['raw_manifest_sha256'] not in m['source_manifests']):
                raise ValueError('source provenance or fold role differs')
            self.entries[key], self.metadata[key] = entry, meta
        if len(self.entries) != m['frame_count']:
            raise ValueError('source inventory count differs')
        self.binding = dict(cache_manifest_sha256=expected_sha256, fold_id=fold_id,
                            calibration_sha256=calibration_sha256,
                            role=role, agent_mask=agent_mask, class_scope='car',
                            candidate_policy='raw-score>=0.05/all-class-top64 then class0')
        self.agent_mask = agent_mask
        self.receipts = {}
        self.last_arrival_us = {}
        self.prefix_sha256 = '0' * 64

    def consume(self, key, *, arrival_us, decision_us, frame_sha256):
        key = tuple(key)
        if key not in self.entries or type(arrival_us) is not int or type(decision_us) is not int:
            raise ValueError('unknown source or noninteger arrival/decision')
        meta, entry = self.metadata[key], self.entries[key]
        source_complete = max(meta['source_image_timestamp_us'], meta['box_reference_timestamp_us'])
        if not source_complete <= arrival_us <= decision_us or frame_sha256 != entry['frame_sha256']:
            raise ValueError('future, incomplete or changed source receipt')
        if key in self.receipts:
            first = self.receipts[key]
            if arrival_us < first['arrival_us']:
                raise ValueError('duplicate cannot change first arrival')
            # No payload reload, no repeated neural factor, no new prefix entry.
            return copy.deepcopy(first), None
        if arrival_us < self.last_arrival_us.get(key[0], -1):
            raise ValueError('first receipts must follow causal arrival order')
        included = bool(SIDES[key[1]] & self.agent_mask)
        frame, selected = None, []
        if included:
            frame = DetectionCacheV2.load(self.root, entry)
            candidates = choose_candidates(frame.raw_scores, dict(minimum_raw_score=.05, maximum_per_side=64))
            selected = candidates[frame.class_indices[candidates] == 0].tolist()
        record = dict(sequence_id=key[0], side=key[1], frame_id=key[2],
                      arrival_us=arrival_us, decision_us=decision_us,
                      source_complete_at_us=source_complete, frame_sha256=frame_sha256,
                      included_by_same_root_mask=included, selected_car_queries=selected,
                      previous_prefix_sha256=self.prefix_sha256)
        record['prefix_sha256'] = hashlib.sha256(canonical(record)).hexdigest()
        self.receipts[key] = record
        self.prefix_sha256 = record['prefix_sha256']
        self.last_arrival_us[key[0]] = arrival_us
        return copy.deepcopy(record), frame

    def completion(self):
        if set(self.receipts) != set(self.entries):
            raise ValueError('not every source has been accounted for')
        return dict(**self.binding, source_frames_accounted_for=len(self.receipts),
                    payload_frames_consumed=sum(r['included_by_same_root_mask'] for r in self.receipts.values()),
                    selected_car_queries=sum(len(r['selected_car_queries']) for r in self.receipts.values()),
                    prefix_sha256=self.prefix_sha256, tracking_executed=False, paper_eligible=False)
