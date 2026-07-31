# Resilient V2X core API

This document freezes the protocol-facing core implemented by
`transvision.models.resilient_v2x`. It describes in-memory contracts and Mac
arm64 CPU development evidence; it is not a detector, trainer, evidence
serializer, or deployment specification.

## Frozen public API

The following ordered tuple is the frozen 35-name prefix. Future package
versions follow an append-only rule: they may add names after this prefix but
must preserve every existing name and its order.

1. `Agent`
2. `Modality`
3. `UnsupportedReason`
4. `ProtocolInvariantError`
5. `SourceCandidate`
6. `RejectedCandidate`
7. `BranchSelection`
8. `BEVGridSpec`
9. `compose_source_to_target`
10. `build_backward_grid`
11. `align_bev_to_target`
12. `warp_with_displacement`
13. `arrived_candidates`
14. `select_causal_source`
15. `age_decay`
16. `branch_reliability`
17. `latest_arrived_rsu_age_intervals`
18. `PTFOutput`
19. `HorizonConditionedPTF`
20. `BranchDiagnostics`
21. `RepairedBranch`
22. `CausalBranchRepair`
23. `DepthwiseSeparableResidualBlock`
24. `ModalityAggregate`
25. `ModalityAggregator`
26. `RoutingOutput`
27. `DynamicExpertRouter`
28. `RoutingDiagnostics`
29. `SampleInferenceDiagnostics`
30. `DistillationLosses`
31. `freeze_teacher`
32. `assert_teacher_frozen`
33. `bernoulli_kl_from_logits`
34. `distillation_losses`
35. `ResilientFeatureBatch`

The controlled-comparison input surface is appended after that frozen prefix:

36. `CONTROLLED_BRANCH_KEYS`
37. `ControlledBaselineInputBatch`
38. `ControlledBaselineInputSelector`
39. `select_controlled_baseline_inputs`

No wildcard import defines this surface. The first 35 names and their order
remain frozen; the four controlled-baseline names follow the same append-only
compatibility rule.

## Controlled-baseline input contract

`CONTROLLED_BRANCH_KEYS` fixes the branch order to
`[L_E,L_R,C_E,C_R]`. `ControlledBaselineInputBatch` carries the four selected
BEV feature tensors, per-sample support masks, and causal ages without exposing
the paper method's PTF, dynamic router, teacher, or DER path.

`ControlledBaselineInputSelector` and
`select_controlled_baseline_inputs` select one causal source per branch, align
supported source features into the target frame, and leave unsupported inputs
exactly neutral. The shared fusion contract rejects any sample for which all
four branches are unavailable. Geometry is evaluated in FP32 before the
selected features are returned to the baseline fusion module.

## Public signatures

Protocol enums and errors:

```python
class Agent(str, Enum)
class Modality(str, Enum)
class UnsupportedReason(str, Enum)
class ProtocolInvariantError(ValueError)
```

Source and branch records:

```python
SourceCandidate(
    packet_id: str,
    n_s: int,
    tau_s_ms: int,
    arrival_tau_ms: int | None,
    payload_valid: bool,
    timestamp_valid: bool,
    pose_valid: bool,
    calibration_valid: bool,
    faulted: bool,
)

RejectedCandidate(
    packet_id: str,
    n_s: int,
    reason: UnsupportedReason,
)

BranchSelection(
    agent: Agent,
    modality: Modality,
    supported: bool,
    source: SourceCandidate | None,
    horizon: int | None,
    endpoint_tick: int | None,
    observed: bool,
    propagated: bool,
    rejected: tuple[RejectedCandidate, ...],
    reason: UnsupportedReason | None,
)

BranchDiagnostics(
    agent: Agent,
    modality: Modality,
    supported: bool,
    source_tick: int | None,
    source_tau_ms: int | None,
    horizon: int | None,
    observed: bool,
    propagated: bool,
    gamma: float | None,
    reliability: float,
    ptf_queried: bool,
    displacement: Tensor | None,
    confidence: Tensor | None,
    reason: UnsupportedReason | None,
)

RepairedBranch(
    feature: Tensor,
    reliability: Tensor,
    support: Tensor,
    observed: Tensor,
    propagated: Tensor,
    age_intervals: Tensor,
    displacement: Tensor | None,
    confidence: Tensor | None,
    diagnostics: tuple[BranchDiagnostics, ...],
)
```

Geometry:

```python
BEVGridSpec(
    x_min: float,
    y_min: float,
    resolution: float,
    height: int,
    width: int,
)

compose_source_to_target(
    source_world_from_agent: Tensor,
    target_world_from_ego: Tensor,
) -> Tensor

build_backward_grid(
    source_to_target: Tensor,
    spec: BEVGridSpec,
    dtype: torch.dtype,
    device: torch.device,
) -> Tensor

align_bev_to_target(
    source: Tensor,
    source_to_target: Tensor,
    spec: BEVGridSpec,
) -> Tensor

warp_with_displacement(
    source: Tensor,
    displacement_cells: Tensor,
) -> Tensor
```

Causal selection and repair:

```python
arrived_candidates(
    agent: Agent,
    target_tau_ms: int,
    candidates: Sequence[SourceCandidate],
) -> tuple[SourceCandidate, ...]

select_causal_source(
    agent: Agent,
    modality: Modality,
    n_t: int,
    target_tau_ms: int,
    delta_t_ms: int,
    history_limit: int,
    candidates: Sequence[SourceCandidate],
) -> BranchSelection

age_decay(horizon: Tensor, alpha: float) -> Tensor

branch_reliability(
    selection: BranchSelection,
    confidence: Tensor | None,
    alpha: float,
    batch_index: int,
) -> Tensor

latest_arrived_rsu_age_intervals(
    candidates: Sequence[SourceCandidate],
    target_tau_ms: int,
    history_limit: int,
    delta_t_ms: int,
) -> float

CausalBranchRepair(
    grid_spec: BEVGridSpec,
    channels: int,
    alpha: float,
    history_limit: int,
)
```

PTF:

```python
PTFOutput(displacement: Tensor, confidence: Tensor)

HorizonConditionedPTF(
    in_channels: int,
    history_positions: int,
    history_limit: int,
    projected_channels: int,
    context_channels: int,
    max_low_resolution_cells: float,
    mode: Literal["nonlinear", "linear"],
)
```

Aggregation and routing:

```python
DepthwiseSeparableResidualBlock(channels: int)

ModalityAggregate(
    feature: Tensor,
    support: Tensor,
    reliability: Tensor,
)

ModalityAggregator(channels: int)

RoutingOutput(
    fused: Tensor,
    weights: Tensor,
    descriptor: Tensor,
    expert_support: Tensor,
    lidar_expert: Tensor,
    camera_expert: Tensor,
    synergy_expert: Tensor,
    overall_support: Tensor,
)

DynamicExpertRouter(channels: int, hidden_channels: int)
```

Inference diagnostics:

```python
RoutingDiagnostics(
    descriptor: Tensor | None,
    expert_support: Tensor | None,
    weights: Tensor | None,
    not_applicable_reason: str | None,
)

SampleInferenceDiagnostics(
    schema_version: Literal[1],
    sample_id: str,
    method: str,
    branches: tuple[
        BranchDiagnostics,
        BranchDiagnostics,
        BranchDiagnostics,
        BranchDiagnostics,
    ],
    routing: RoutingDiagnostics,
)

ResilientFeatureBatch(
    fused: Tensor,
    overall_support: Tensor,
    routing_weights: Tensor,
    routing_descriptor: Tensor,
    branch_features: Mapping[str, Tensor],
    diagnostics: tuple[SampleInferenceDiagnostics, ...],
)
```

Distillation:

```python
DistillationLosses(
    feature: Tensor,
    bernoulli: Tensor,
    total: Tensor,
)

freeze_teacher(teacher: nn.Module) -> nn.Module
assert_teacher_frozen(teacher: nn.Module) -> None

bernoulli_kl_from_logits(
    teacher_logits: Tensor,
    student_logits: Tensor,
    temperature: float,
    epsilon: float,
) -> Tensor

distillation_losses(
    teacher_feature: Tensor,
    student_feature: Tensor,
    teacher_logits: Tensor,
    student_logits: Tensor,
    temperature: float,
    lambda_feature: float,
    lambda_logit: float,
    valid_sample_mask: Tensor,
) -> DistillationLosses
```

## Tensor and coordinate conventions

Feature tensors use `[B,C,Y,X]`: batch, channel, BEV row, then BEV column.
`BEVGridSpec` places samples at each cell-center. Rigid transforms map source
coordinates into target coordinates; resampling builds the inverse transform
for backward sampling. Identity geometry is exact.

PTF displacement is expressed in low-resolution cells with channel order
`[D_x,D_y]`: channel 0 moves along the BEV X/column direction and channel 1
moves along the BEV Y/row direction. Geometry transforms and learned
displacement are each applied exactly once.

## Causal selection and unsupported reasons

The arrival cutoff is `arrival_tau_ms <= target_tau_ms`; a future target packet
or a condition label never contributes delay metadata. Candidate filtering is:

Before fault and metadata filtering, `endpoint_tick` is fixed to `n_t` for
Ego and to the greatest arrived RSU `n_s` for RSU. Source selection then uses
the following filtering order without moving that endpoint:

1. reject a future source tick;
2. for RSU, reject null or future arrival;
3. reject a faulted packet;
4. validate timestamp, payload, pose, then calibration;
5. require horizon `0..3`;
6. select the greatest `(n_s, packet_id)` among survivors.

The unsupported-reason precedence is:

1. RSU with no arrived candidate: `EMPTY_ARRIVAL_SET`;
2. no unmasked history: `EMPTY_MODALITY_HISTORY`;
3. newest metadata rejection;
4. no candidate within history: `UNSUPPORTED_HORIZON`.

Within one candidate, metadata order is `INVALID_TIMESTAMP`,
`MISSING_PAYLOAD`, `INVALID_POSE`, `INVALID_CALIBRATION`.
`METHOD_NOT_APPLICABLE` denotes a branch the method does not implement.

A supported branch is observed exactly when its selected source tick equals
its branch `endpoint_tick`; selecting an older source after endpoint loss is
propagated. Consequently, the latest arrived RSU source is observed even when
its age relative to the Ego decision time is nonzero. The total age decay is
`gamma = alpha**h` and is applied once to the repaired feature. Only a truly
current Ego source has reliability one. Every other supported branch,
including an observed delayed RSU endpoint, has reliability
`gamma * mean(PTF confidence)`. Unsupported reliability is exact zero.

## PTF and causal repair

`HorizonConditionedPTF` consumes a finite history/context tensor and returns
displacement `[B,2,Y,X]` plus confidence `[B,1,Y,X]`. The configured bound
limits each displacement component, and confidence remains in `[0,1]`.

At `h=0`, the horizon embedding still participates in the declared model but
the displacement convention remains the same. The nonlinear mode uses its
nonlinear projection stack; the linear mode is the capacity-matched ablation.
Both modes preserve shapes, bounds, dtype, device, and finite outputs.

`CausalBranchRepair` gathers supported rows before alignment or PTF. No
unsupported branch calls PTF. Supported rows are aligned, queried, warped, and
age-decayed once, then scattered back. That internal neutral scatter uses zero
tensors so batched computation stays well-defined. By contrast, unavailable
algorithm evidence is diagnostic `None`, never a fabricated tensor.

## Four branches, aggregation, and DER

Every sample uses branch order `[L_E,L_R,C_E,C_R]`, meaning Ego LiDAR, RSU
LiDAR, Ego camera, RSU camera. Unsupported branch features are zeroed before
aggregation. Modality support is branch OR; modality reliability is always the
two branch reliabilities divided by two.

Expert order is LiDAR, camera, synergy. Synergy support requires both modality
aggregates. The DER routing modes are dynamic, static, and uniform. Invalid
experts are masked before GAP and before the weighted sum. An all-invalid
Resilient row is still DER-applicable: expert support, weights, and fused
feature are real exact-zero tensors with no NaN.

The descriptor arithmetic is `3*256 + 2 + 8 + 4 + 1 = 783`. Its slices are:

- `[0:256]`: masked LiDAR expert GAP;
- `[256:512]`: masked camera expert GAP;
- `[512:768]`: masked synergy expert GAP;
- `[768:770]`: LiDAR and camera reliability;
- `[770:778]`: observed/propagated flags in four-branch order;
- `[778:782]`: branch source ages `h=(t-s*)/Delta t` in sampling intervals,
  in four-branch order;
- `[782]`: latest causally arrived RSU source age `d_R/Delta t` in sampling
  intervals. This is independent of an older modality-specific fallback
  selected after a fault; it is zero when no valid RSU arrival is available.

Repair/runtime compute these interval-unit values; the router does not divide
them again.

## Distillation

The teacher freeze operation recursively selects eval mode, disables parameter gradients,
and clears stale gradients. Teacher feature and logits are stop-gradient loss
inputs.

The feature MSE averages `C*Y*X` per valid sample, then averages valid samples.
CPU float16 and bfloat16 features are promoted to float32 for MSE; float64
features retain float64 precision. Bernoulli KL includes positive and negative
terms over every dense logit, including background. The valid mask comes only
from student support; invalid rows are gathered out before numerical work and
receive zero gradient.

The returned total is
`lambda_feature*feature + lambda_logit*T^2*bernoulli`. The `T^2` factor appears
only in the total. An empty valid set is an explicit runtime error.

## Task 8 in-memory contracts and null states

`RoutingDiagnostics` has two exclusive states:

- DER applicable: detached descriptor `[783]`, bool expert support `[3]`,
  detached weights `[3]`, and no reason;
- DER non-applicable: all three tensors are Python `None`, plus a non-empty
  reason.

Mixed states fail. The applicable all-invalid state uses real zero support and
weights; it is distinct from a baseline without DER.

`SampleInferenceDiagnostics` fixes schema version 1, a trimmed sample ID and
method, four validated branch records, and one routing record. A missing
modality has null source/horizon/gamma/displacement/confidence, false flags,
zero reliability, `ptf_queried=False`, and
`METHOD_NOT_APPLICABLE`. A supported method without PTF retains its real
source and flags while gamma/displacement/confidence remain `None`.

`ResilientFeatureBatch` validates finite fused `[B,256,Y,X]`, bool support
`[B]`, weights `[B,3]`, descriptor `[B,783]`, branch feature batches, unique
sample IDs, and equality between each detached diagnostic row and its live
training tensor row. Training tensors may retain autograd; diagnostic tensors
may not.

Task 8 owns only these in-memory invariants. A test-local normalizer can show
that Python `None` becomes JSON `null` and enum reasons become stable strings.
Durable tensor transfer, arrays, hashing, indexing, and canonical JSON belong
to the later evidence exporter.

## Failure taxonomy and evidence boundary

- A `ProtocolInvariantError` means a caller constructed an internally
  inconsistent protocol object, tensor shape, diagnostic state, or batch.
- An unsupported outcome is expected domain state represented by
  `supported=False` plus a typed `UnsupportedReason`; it is not an exception.
- A runtime failure means finite validated inputs produced an invalid numerical
  result, or no valid sample exists for distillation.

The recorded results are Mac arm64 CPU development/protocol evidence. CUDA,
GPU, MPS, containers, custom ops, real data, and locked Linux acceptance are
outside this evidence boundary and require their own execution.
