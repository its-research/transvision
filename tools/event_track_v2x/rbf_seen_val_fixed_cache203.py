"""Seen-val input adapter for unchanged independent float64 feature/context math."""
import collections
import hashlib
import json
from pathlib import Path
import tarfile

from rbf_nested_seen_val_v2_common import R, sha
from rbf_seen_val_fixed_CPU_binding import frozen_modules, registered, scope


def load_metadata(root, manifest, checkpoint, origins):
    """Bind every original metadata row; NPZ bytes are verified on consumption."""
    assert manifest['split'] == 'val' and manifest['gt_in_cache'] is False
    assert manifest['frame_count'] == len(manifest['frames']) == 7189
    assert len(manifest['sequences']) == len(set(manifest['sequences'])) == 21
    assert set(manifest['sequences']) == set(origins)
    frames, paths, minimum = {}, set(), {}
    identity = checkpoint['frozen_cache_identity']
    for entry in manifest['frames']:
        for role in ('arrays','metadata'):
            spec = entry[role]; relative = Path(spec['path'])
            assert not relative.is_absolute() and '..' not in relative.parts and str(relative) not in paths
            paths.add(str(relative)); path = root/relative
            assert path.is_file() and path.resolve().is_relative_to(root.resolve())
            assert not any(p.is_symlink() for p in (path,*path.parents))
            assert path.stat().st_size == spec['bytes']
        raw = (root/entry['metadata']['path']).read_bytes()
        assert hashlib.sha256(raw).hexdigest() == entry['metadata']['sha256']
        meta = json.loads(raw)
        assert meta['arrays_sha256'] == entry['arrays']['sha256']
        assert meta['dataset_split'] == 'val' and meta['calibration_fit_split'] == 'train'
        assert all(meta[k] == identity[k] for k in ('calibration_sha256','feature_method','feature_checkpoint_sha256'))
        assert meta['detector_checkpoint_sha256'] == identity['detector_checkpoint_sha256'][meta['side']]
        key = (meta['sequence_id'],meta['side'],meta['frame_id'])
        assert key not in frames and key[0] in origins
        frames[key] = dict(metadata=meta,manifest_entry=entry)
        minimum[key[0]] = min(minimum.get(key[0],meta['box_reference_timestamp_us']),meta['box_reference_timestamp_us'])
    assert minimum == origins, 'origins must include every original frame, including unavailable frames'
    return frames


class CacheAdmission:
    def __init__(self, seed, checkpoint, job):
        model_binding, original, output_binding, _ = frozen_modules(job['K'])
        assert seed == job['seed'] == job['plan']['seed']
        plan = job['plan']
        index_entry, ck, expected, events, groups = output_binding.reference_inputs(seed,plan)
        assert sha(checkpoint) == plan['checkpoint']['sha256']
        assert json.loads(Path(checkpoint).read_bytes()) == ck
        self.checkpoint, self.seed = ck, seed
        train = model_binding.validate_final_model(seed,checkpoint)
        self.final_model_binding = dict(final_checkpoint_sha256=sha(checkpoint),final_model_sha256=ck['model_sha256'],
            inherited_train_NN_admission=train,seen_val_NN_numeric_admission_sha256=index_entry['full_numeric_receipt_sha256'],
            seen_val_NN_byte_admission_sha256=index_entry['output_byte_receipt_sha256'],
            old_model_full_forest_acceptance_inherited=False,full_forest_accepted=False)
        bridge = R/f'artifacts/rbf-seen-val-forest-input-bridge-v1-20261005/seed{seed}'
        bound = registered(bridge/'input-binding.json')
        assert bound['seed'] == seed and bound['events_sha256'] == plan['events']['sha256']
        assert bound['rows'] == index_entry['rows'] and bound['full_original_schedule_to_forest_event_binding_passed'] is True
        assert bound['inventory_sha256'] == sha(bridge/'cache-inventory.json')
        for proof in bound['prerequisite_receipts']:
            assert sha(proof['path']) == proof['sha256']; registered(proof['path'])
        inventory = json.loads((bridge/'cache-inventory.json').read_bytes())
        self.root = R/f'artifacts/rbf-nested-seen-val-matching-V2-full-admission-v1-20261004/seed{seed}/cache'
        assert inventory['cache_root'] == str(self.root)
        assert sha(self.root/'manifest.json') == inventory['cache_manifest_sha256'] == plan['cache_manifest']['sha256']
        self.origins, self.events = events['origin_us_by_sequence'], groups
        assert events['arrival_policy'] == 'scheduled_pair_snapshot_at_reference_plus_100ms'
        assert events['measured_network_arrival_history_verified'] is False
        manifest = json.loads((self.root/'manifest.json').read_bytes())
        self.frames = load_metadata(self.root,manifest,ck,self.origins)
        source = R/'artifacts/rbf-original-joint-coupled-forest-GPU4-v3-source-20261001/independent-cloud-readback/source.bytes'
        source_sha = '038fa8118c9540d91073fbb8bf594fb6abefe8347f13bb69dfb006d27fcfda03'
        assert sha(source) == source_sha
        with tarfile.open(source,'r:gz') as archive:
            for filename in ('forest_tracking.py','forest_row_context.py','forest_potentials.py','prediction_features.py','tracking_v2.py'):
                name = 'transvision/models/event_track_v2x/'+filename
                assert hashlib.sha256(archive.extractfile(name).read()).hexdigest() == ck['source_sha256'][name]
        self.proof = dict(manifest_sha256=plan['cache_manifest']['sha256'],per_sequence_rows={s:len(v) for s,v in expected.items()})
        self.memo = collections.OrderedDict()
        self.binding = dict(seed=seed,cache_manifest_sha256=plan['cache_manifest']['sha256'],checkpoint_sha256=sha(checkpoint),
            schedule_sha256=plan['events']['sha256'],input_bridge_sha256=sha(bridge/'input-binding.json'),
            independently_admitted_forward_sha256=index_entry['full_numeric_receipt_sha256'],
            rows=index_entry['rows'],original_frozen_runtime_source_sha256=source_sha)
        self.binding.update(self.final_model_binding)
        self._original = original
        self.scope = scope(job['K'])

    def frame(self, delivery):
        return self._original.CacheAdmission.frame(self,delivery)


def verify_database(path, expected_sha, admission, *, allow_prefix=False, progress=None):
    assert allow_prefix is False, 'seen-val full admission never accepts prefixes'
    result = admission._original.verify_database(path,expected_sha,admission,allow_prefix=False,progress=progress)
    assert result['complete_original_sequence'] is True and result['atol'] == result['rtol'] == 1e-8
    result.update(scope=admission.scope,kind='rbf_seen_val_fixed_full_runtime_cache203_context_sequence_admission_v1',
        measured_network_arrival_history_verified=False,original_schedule_inferred=False,
        formal_independent_evaluation=False,NN_numeric_repeated=False)
    return result
