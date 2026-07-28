# V2XSet-Standard loader-trace manifest boundary

`export_v2xset_standard.py` is a fail-closed manifest exporter for the paper's
V2XSet-Standard track. It does not parse V2XSet, replace its source dataloader,
run CoFormerNet, run an evaluator, or create model measurements.

## Repository audit result

The current repository has no V2XSet/OpenCOOD multi-agent dataloader or
V2XSet evaluator to instrument. The similarly named CoFormer paths are a DAIR
one-vehicle/one-infrastructure implementation:

- `configs/coformer/coformer.py` points at a DAIR data root and loads exactly
  `points` plus `infrastructure_points`;
- `transvision/dataset/v2x_dataset.py` is the MMDetection3D DAIR dataset wrapper;
- `transvision/dataset/dair_v2x_for_detection.py` builds a DAIR `VICFrame` from
  one vehicle frame and one infrastructure frame;
- `transvision/models/detection_models/mmdet3d_lidar_coformer.py` accepts one
  vehicle LiDAR file and one infrastructure LiDAR file.

Those files cannot establish the source V2XSet Standard split, random training
Ego behavior, fixed validation Ego, `max_cav`, agent order, truncation, or
per-agent delay generation. The exporter therefore accepts only a trace emitted
by the actual source dataloader. It rejects unresolved fields instead of filling
them from common conventions.

## Trace unit and required coverage

Create one canonical trace per split and executed nominal-latency condition.
Source-aligned conditions are 0, 200, and 300 ms. A 100 ms trace must be labeled
`supplemental`; it is not silently promoted to a source-aligned point.

Every trace must be canonical UTF-8 JSON, have no duplicate keys, carry a
`content_sha256` self-hash, and use:

```text
schema_version = 1
artifact_type = v2xset_standard_dataloader_trace
dataset_name = V2XSet
split = train | validation
```

The exact top-level objects are:

- `dataset_identity`: resolved release ID, release-inventory SHA-256, source
  split ID, and source split SHA-256;
- `loader_identity`: implementation name, exact repository commit, executed
  config path and SHA-256, and trace-hook source location;
- `selection_policy`: communication range, `max_cav`, exact ordering rule and
  source, prefix truncation rule and source, and exact Ego-selection rule and
  source;
- `delay_policy`: nominal latency, source-aligned/supplemental label, generation
  rule, causal frame-mapping rule, whether remote delays are uniform, and code
  source location;
- `records`: the actual dataloader emissions, not a reconstructed sample list.

The validator fixes the communication range to 70.0 m and requires truncation
rule `prefix_after_ordering_to_max_cav`. It accepts a resolved `max_cav` from the
executed loader configuration; it never supplies a default.

### Training trace

`selection_policy.ego_selection_rule` must be `random_per_epoch`. Each record
contains the global `epoch` and `draw_index`, the complete available vehicle
candidate list in the actual random-choice order, selected index, RNG seed, and
SHA-256 of the RNG state immediately before the draw. This records what the
loader selected; it does not claim that a short trace statistically proves
randomness.

### Validation trace

`selection_policy.ego_selection_rule` must be `fixed_per_scene`. Each record has
`epoch = null` and no RNG fields. The exporter checks that the selected Ego is
unchanged throughout each scene and rejects substitution.

### Per-record multi-agent trace

Each record contains:

- every agent available before the communication-range filter, its type, and
  actual distance to the selected Ego;
- the exact in-range order after applying `distance <= 70.0 m`, beginning with
  the selected Ego;
- the admitted ordered prefix through `max_cav` and the exact truncated suffix;
- for every admitted agent, a LiDAR payload identity (`relative_path`, byte
  size, SHA-256), generated delay, selected source timestamp, causal arrival
  timestamp, and causal frame offset.

Payload paths must resolve beneath `--data-root`; files are verified as regular
files by size and SHA-256. Camera keys and model/evaluator results are not valid
trace fields. For every admitted packet the exporter verifies
`arrival = source + generated_delay` and `arrival <= target`. A uniform-delay
trace must assign the nominal delay to every admitted remote agent. The Ego must
remain current with zero delay.

## Export

```bash
python tools/resilient_v2x/export_v2xset_standard.py \
  artifacts/v2xset/standard/train-delay-200.trace.json \
  --data-root data/V2XSet \
  --out artifacts/v2xset/standard/train-delay-200.manifest.json
```

The output is canonical, self-hashed, idempotently published, and refuses to
overwrite different content. It preserves the trace and protocol identities,
the random/fixed Ego evidence, 70 m membership, `max_cav`, order, truncation,
LiDAR identities, and per-agent causal delay trace.

## Method and evidence boundary

Every exported manifest states:

```json
{"method_id":"resilient_v2x.one_ego_one_rsu","multi_agent_adapter_implemented":false,"reason_code":"standard_multi_agent_adapter_not_implemented","status":"N/A"}
```

It also fixes model outputs, evaluator outputs, and controlled result ID to
absent values. This is intentional: the current one-Ego/one-RSU ResilientV2X
method is not eligible for the Standard multi-agent track. A future genuine
multi-agent method adapter requires a different reviewed schema/configuration;
editing this manifest to change `N/A` is rejected.

V2XSet-Standard remains LiDAR-only and must not contain Camera/C-Fail,
two-agent R-only/E+R labels, or a standard L-Fail row. Standard and Pair use
separate traces, manifests, checkpoints, result IDs, and tables. This exporter
does not make either track an end-to-end reproduced result; an actual V2XSet
loader trace, model checkpoint, evaluator identity, predictions, and evaluator
outputs are still required before reporting any metric.
