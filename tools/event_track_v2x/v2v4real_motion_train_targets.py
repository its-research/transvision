"""Offline train-only past identity velocity targets; no online GT or model fit.

Caller must separately admit input bytes and official train membership. This
primitive returns explicit validity and predecessor evidence; first observations
remain unavailable, never measured zero. Poses use column-vector source_to_world.
"""
import numpy as np


def targets(manifest, frames):
    if manifest.get('split') != 'train' or manifest.get('official_test_read') is not False:
        raise ValueError('train-only supervision required before reading frames')
    history, last_ordinal, out = {}, {}, []
    for frame in frames:
        sequence, side, ordinal = frame['sequence_id'], frame['side'], frame['frame_ordinal']
        if not isinstance(sequence, str) or not sequence or side not in ('vehicle-side', 'infrastructure-side'):
            raise ValueError('explicit sequence and source identity required')
        if type(ordinal) is not int or ordinal < 0:
            raise ValueError('nonnegative nominal frame ordinal required')
        key = sequence, side
        timestamp = ordinal * 100000  # frozen nominal 10 Hz, not measured time
        if ordinal <= last_ordinal.get(key, -1):
            raise ValueError('duplicate or noncausal frame order')
        last_ordinal[key] = ordinal
        state = np.asarray(frame['states_source'], dtype=np.float64)
        ids = frame['track_ids']
        if not isinstance(ids, list):
            raise ValueError('explicit native identity list required')
        pose = np.asarray(frame['source_to_world'], dtype=np.float64)
        if (state.shape != (len(ids), 7) or not np.isfinite(state).all()
                or np.any(state[:, 3:6] <= 0) or pose.shape != (4, 4)
                or not np.isfinite(pose).all() or not np.array_equal(pose[3], [0, 0, 0, 1])):
            raise ValueError('finite states and homogeneous pose required')
        rotation = pose[:3, :3]
        if not np.allclose(rotation.T @ rotation, np.eye(3), atol=2e-5, rtol=0) or not np.isclose(np.linalg.det(rotation), 1, atol=2e-5, rtol=0):
            raise ValueError('proper source-to-world rotation required')
        if any(type(i) is not int or i < 0 for i in ids):
            raise ValueError('native integer identities required')
        world = state[:, :3] @ rotation.T + pose[:3, 3]
        velocity = np.full((len(ids), 2), np.nan, dtype=np.float64)
        valid = np.zeros(len(ids), dtype=bool)
        predecessors = [None] * len(ids)
        ambiguous = {i for i in ids if ids.count(i) > 1}
        for j, identity in enumerate(ids):
            hkey = key + (identity,)
            if identity in ambiguous:
                history.pop(hkey, None)
                continue
            previous = history.get(hkey)
            if previous is not None:
                dt = (timestamp - previous[0]) / 1e6
                if dt <= 0:
                    raise ValueError('nonpositive identity time difference')
                local = ((world[j] - previous[1]) / dt) @ rotation
                velocity[j] = local[:2]
                valid[j] = True
                predecessors[j] = previous[2]
            history[hkey] = timestamp, world[j].copy(), ordinal
        out.append(dict(sequence_id=sequence, side=side, frame_ordinal=ordinal,
                        velocity_source=velocity, velocity_valid=valid,
                        predecessor_ordinals=predecessors, clock='nominal_10hz'))
    return out
