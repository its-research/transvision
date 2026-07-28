# V2XSet-Pair sequence adapter

`prepare_v2xset_pair.py` implements the paper-scoped, deterministic
**V2XSet-Pair** derivation boundary. It produces a camera-enabled one-Ego,
one-Infra subset; it does not implement or claim V2XSet-Standard multi-agent
evaluation.

## Exact selection protocol

The unit of selection is a sequence, not an independent frame.

- The input scene declares `split`. A validation scene must also carry the
  V2XSet-Standard `standard_fixed_ego_agent_id`; a training scene must set this
  field to `null`.
- Each agent declares its stable `agent_id`, integer `numeric_id`, and
  `agent_type` (`av` or `infrastructure`). Numeric order, never lexical order,
  selects the training Ego and breaks Infra distance ties.
- An eligible pair is at most 70 m apart at target time. Both members must have
  a verified LiDAR payload, all four RGB payloads, valid intrinsics, valid
  Camera-to-LiDAR rigid extrinsics, and valid poses at every required endpoint.
- For target time `t`, interval `Delta`, configured delay `d`, and
  `h = 0,...,history_limit`, required timestamps are
  `t - h*Delta` for Ego and `t - d - h*Delta` for Infra. Every configured delay
  must map exactly to the input frame interval. Missing frames, agents,
  payloads, calibration, or identity continuity make that candidate ineligible
  and are recorded with the exact delay, horizon, timestamp, member, and
  failure reason.
- The training anchor is the earliest target with any eligible AV/Infra pair.
  At that anchor, the smallest numeric eligible AV becomes Ego. Validation uses
  only its input Standard fixed Ego and never substitutes another AV. A scene
  with no valid anchor is excluded.
- At the anchor, the nearest eligible Infra is selected; numeric ID and then
  stable agent ID break an exact distance tie. Ego and Infra IDs are frozen for
  the whole sequence.
- Before-anchor frames are excluded. Every later frame is checked only with the
  frozen pair. An ineligible later frame is excluded with its exact reason and
  no replacement; a later eligible frame may remain in the derived subset.
- Every unselected current-frame agent is explicitly listed in
  `excluded_agent_ids`. Only the two frozen IDs appear in
  `target_eligibility_agent_ids`, and only those agents are embedded for target
  eligibility, encoding, and fusion.

## Normalized inventory schema

The input is canonical, self-hashed JSON with `schema_version: 2` and
`artifact_type: "v2xset_normalized_inventory"`. Its top-level protocol fields
are:

```json
{
  "configured_delays_ms": [0, 100, 200, 300],
  "history_limit": 3,
  "frame_interval_us": 100000,
  "scenes": [
    {
      "scene_id": "sequence-id",
      "split": "train",
      "standard_fixed_ego_agent_id": null,
      "frames": [
        {
          "timestamp_us": 0,
          "agents": [
            {
              "agent_id": "stable-source-id",
              "numeric_id": 1,
              "agent_type": "av",
              "world_from_agent": [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
              "lidar": {
                "payload": {"relative_path": "...", "size_bytes": 0, "sha256": "..."},
                "agent_from_lidar": [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
              },
              "cameras": []
            }
          ]
        }
      ]
    }
  ]
}
```

The abbreviated camera array above must contain exactly `camera_0` through
`camera_3`, in any input order; each record contains `image`, `intrinsic`, and
`agent_from_camera`. Every payload identity contains a relative path, byte size,
and lowercase SHA-256. Paths are resolved below `--data-root`, symlinks and
non-regular files are rejected, and bytes are rehashed before selection.
The inventory itself must include a valid `content_sha256` and use the canonical
JSON encoding used elsewhere in this repository.

## Output and condition invariance

```bash
python tools/resilient_v2x/prepare_v2xset_pair.py \
  artifacts/v2xset/normalized_inventory.json \
  --data-root data/V2XSet \
  --out artifacts/v2xset/pair_manifest.json
```

The output is canonical and self-hashed (`schema_version: 2`). It contains:

- one sequence record with all anchor attempts and the frozen choice;
- included pair/target records and frame-level exclusions with exact reasons;
- complete verified endpoint payload/calibration records for every included
  target;
- counted and independently hashed included/excluded sequence IDs, target
  sample IDs, and excluded frame IDs;
- a nested, independently self-hashed `v2xset_pair_target_manifest`; and
- one reference for Full-LC and both E+R faults at every configured delay, plus
  the four paper-scoped Ego-only/Infra-only diagnostics at 0 ms.

Every condition reference is validated against the exact same target-manifest
hash and sample count. The target sample IDs are derived before latency
selection and fault injection, so condition-specific target drift is rejected.
Repeating the same publication is idempotent; different content at an existing
destination is rejected.
