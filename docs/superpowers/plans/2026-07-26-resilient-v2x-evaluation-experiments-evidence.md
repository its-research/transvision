# Resilient V2X 评估、实验与证据 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 交付直接消费 Runner predictions 的 `resilient-v2x-ap-v1` evaluator、确定性实验 DAG、不可变 attempts/provenance/test ledger、三 seed 聚合、profiling、定性渲染和论文候选证据导出。

**Architecture:** `transvision.experiments.resilient_v2x` 保存纯协议、ID、矩阵、证据和统计逻辑；`tools/resilient_v2x` 只做 CLI 编排。训练/测试仍由 MMEngine Runner 执行，但任何受控命令必须先经过 plan materialization 和 evidence gate，不能直接用旧 `DAIRV2XMetric`、旧 `eval_vic` 或旧模型包装。

**Tech Stack:** Python 3.10、NumPy、PyTorch/MMEngine/MMDetection3D、fvcore 0.1.5.post20221221、jsonschema、pytest；指标核心使用 float64。

## Global Constraints

- 开始前必须通过核心计划和数据/检测器/训练计划的 Completion Gate。
- 旧 `transvision/evaluation/metrics/dair_v2x_metric.py` 保留给旧配置，但 Resilient config 不得引用它；新 evaluator 禁止读取 `last_checkpoint` 或重新运行模型。
- evidence root 必须是用户显式提供、位于 Git worktree 之外的绝对目录；禁止默认写入 `./cache`、`runs/` 或 `training_runs/`。
- val 与 reserved test 使用同一 frozen temporal-manifest 文件/hash，通过不同
  `resolved_split`、plan 和 overlay 字段选择各自 rows；任何
  `test_dataloader` 解析成 val rows 都是启动前协议错误。
- test split 禁止 `--force`，并在首条 prediction 前写完整 claim family ledger。
- capacity-matched concat 使用完整受控训练协议，独立训练 3 个 seeds，并作为四路 multimodal method 在 val/test 各执行 36 个 evaluation runs；其 checkpoint 不依赖 Resilient teacher。
- `--matrix smoke --smoke-max-epochs 2` 只产生独立的 diagnostic protocol/attempt，固定两 epoch、一个 seed 和三个 val conditions；任何 smoke artifact 都不得进入 controlled aggregation、reserved-test claim 或论文 result slot。
- 所有统计只读取逐 seed 原始 `metrics.json`；不接受手录均值、标准差、PDR 或方法差。
- 本计划所有 `python` 命令必须在环境计划 Task 2 生成的
  `/Users/libin/transvision/.venv/resilient-v2x-runtime` 或相同锁定 Docker image
  内执行；新 shell 先激活并验证 Python 3.10 与精确依赖版本。开发 Mac
  无该 runtime 时只记录相关检查 `not_executed`，不得使用系统 Python 替代。
- 所有命令从 `/Users/libin/transvision` 执行。

---

### Task 1: 实现 oriented BEV/3D IoU 和 continuous AP evaluator

**Files:**

- Create: `transvision/evaluation/metrics/resilient_v2x_metric.py`
- Modify: `transvision/evaluation/metrics/__init__.py`
- Modify: `transvision/evaluation/__init__.py`
- Create: `tests/resilient_v2x/fixtures/metric_golden.json`
- Create: `tests/resilient_v2x/test_metric.py`

**Interfaces:**

```python
EVALUATOR_VERSION = "resilient-v2x-ap-v1"
IOU_THRESHOLDS = (0.5, 0.7)
CLASS_NAME = "Car"
METRIC_KEYS = (
    "BEV_AP_0.5",
    "BEV_AP_0.7",
    "3D_AP_0.5",
    "3D_AP_0.7",
)

@dataclass(frozen=True)
class Box7D:
    x: float
    y: float
    z_bottom: float
    length: float
    width: float
    height: float
    yaw: float

@dataclass(frozen=True)
class GroundTruthObject:
    box: Box7D
    original_ground_truth_index: int

@dataclass(frozen=True)
class SampleGroundTruth:
    sample_id: str
    objects: tuple[GroundTruthObject, ...]

@dataclass(frozen=True)
class PredictionObject:
    box: Box7D
    score: float
    original_prediction_index: int

@dataclass(frozen=True)
class SamplePrediction:
    sample_id: str
    objects: tuple[PredictionObject, ...]

@dataclass(frozen=True)
class EvaluationBundle:
    evaluator_version: Literal["resilient-v2x-ap-v1"]
    protocol_sha256: str
    temporal_manifest_sha256: str
    resolved_split: Literal["val", "test"]
    ordered_sample_ids_sha256: str
    evaluator_output: Mapping[str, object]
    metrics: Mapping[str, float]
    bundle_sha256: str

def oriented_bev_iou(first: Box7D, second: Box7D) -> float
def oriented_3d_iou(first: Box7D, second: Box7D) -> float
def continuous_interpolated_ap(recall: np.ndarray, precision: np.ndarray) -> float
def sample_prediction_from_datasample(
    sample_id: str,
    data_sample: Det3DDataSample,
) -> SamplePrediction
def ground_truth_from_temporal_manifest(
    manifest_path: Path,
    expected_manifest_sha256: str,
    expected_split_sha256: str,
    resolved_split: Literal["val", "test"],
    *,
    allow_fixture: bool = False,
) -> tuple[SampleGroundTruth, ...]
def evaluate_predictions(
    predictions: Sequence[SamplePrediction],
    ground_truth: Sequence[SampleGroundTruth],
    detection_range: Sequence[float],
    protocol_sha256: str,
    temporal_manifest_sha256: str,
    resolved_split: Literal["val", "test"],
) -> EvaluationBundle

@METRICS.register_module()
class ResilientV2XMetric(BaseMetric):
    def __init__(
        self,
        manifest_path: str,
        expected_manifest_sha256: str,
        expected_split_sha256: str,
        resolved_split: Literal["val", "test"],
        protocol_sha256: str,
        detection_range: Sequence[float],
        collect_device: str,
        prefix: str | None,
    ) -> None

    def process(self, data_batch: dict, data_samples: Sequence[Det3DDataSample]) -> None
    def evaluate_complete(
        self,
        predictions: Sequence[SamplePrediction],
    ) -> EvaluationBundle
    def compute_metrics(self, results: list[dict[str, object]]) -> dict[str, float]
```

`EVALUATOR_VERSION`、`IOU_THRESHOLDS` 和 `CLASS_NAME` 是代码常量，不允许
config 覆盖。Metric 是纯转换/计算组件，不接收任何 output/attempt 目录；逐样本
spool、恢复和 canonical artifact 写入只由 Task 4 的 `PredictAttempt` 管理。

- [ ] **Step 1: Write golden geometry and AP tests**

```python
import numpy as np
import pytest

from transvision.evaluation.metrics.resilient_v2x_metric import (
    Box7D,
    continuous_interpolated_ap,
    oriented_3d_iou,
    oriented_bev_iou,
)


def test_identical_and_disjoint_boxes() -> None:
    first = Box7D(10, 0, 0, 4, 2, 2, 0)
    second = Box7D(10, 0, 0, 4, 2, 2, 0)
    far = Box7D(30, 0, 0, 4, 2, 2, 0)
    assert oriented_bev_iou(first, second) == pytest.approx(1.0)
    assert oriented_3d_iou(first, second) == pytest.approx(1.0)
    assert oriented_bev_iou(first, far) == 0.0


def test_continuous_ap_includes_first_recall_increment() -> None:
    recall = np.array([0.5, 1.0], dtype=np.float64)
    precision = np.array([1.0, 0.5], dtype=np.float64)
    assert continuous_interpolated_ap(recall, precision) == pytest.approx(0.75)
```

The golden fixture must also cover:

- one canonical mini temporal-manifest payload whose world-frame source label and
  non-identity ego pose were normalized by the data plan to a known ego-LiDAR
  bottom-center `Car` box; the evaluator loader must return that exact box and
  original annotation index only when the test calls the pure loader with
  `allow_fixture=True`; the Metric constructor never exposes that flag;
- yaw `pi/2`, partial overlap, z non-overlap, exact IoU threshold equality;
- score ties resolved by `(sample_id, original_prediction_index)`;
- highest-IoU unmatched GT and lower original GT index tie-break;
- empty predictions with non-empty GT gives AP 0;
- a resolved split/filter whose total `Car` GT count is zero raises, but an
  individual manifest sample with `objects=()` is valid, remains in exact sample
  coverage, and any prediction on it counts as a false positive;
- a mixed golden fixture contains one empty-GT sample with a prediction and one
  non-empty-GT sample, proving the empty frame is neither rejected nor dropped;
- range filtering by bottom-center half-open interval;
- percentage conversion exactly once;
- flat metrics contain exactly `METRIC_KEYS` in that order for every run; values
  are percentages. In particular `metrics["BEV_AP_0.7"]` is the only evaluator
  value promoted to `selection/BEV_AP_0.7` by the separate val-selection loop;
- prediction export preserves raw score/box/index.
- a real `Det3DDataSample` with `LiDARInstance3DBoxes` center-z `1.0`, dimensions
  `(4,2,2)`, yaw and label 0 converts exactly to
  `[x,y,0.0,4,2,2,yaw]` with class `Car`, original prediction order and score;
- any label other than the sole Car label 0, wrong box type, coordinate
  convention, dimension order, length mismatch or non-finite tensor fails closed;
- constructor/config rejects evaluator version, threshold, class or output-path
  override fields rather than ignoring them;
- `process()` performs no filesystem write and appends only canonical serializable
  sample-result dictionaries to the BaseMetric in-memory `results` list;
- missing or unexpected sample IDs fail before metrics are written.
- `ground_truth_from_temporal_manifest()` reads only the manifest's normalized
  inline `GroundTruthBoxRecord` values, verifies both manifest and split hashes,
  selects only rows whose immutable sample split equals the explicit
  `resolved_split`, preserves `source_annotation_index` as the deterministic GT
  tie-break index, and rejects duplicate/missing sample IDs. It never invokes the legacy
  `V2XDataset` or performs a second coordinate conversion.
- the registered Metric always calls the loader with `allow_fixture=False`, so a
  fixture-scope manifest cannot enter Runner evaluation even if a config contains
  extra test-only keys;
- missing `resolved_split`, train evaluation, a split with zero rows, or any
  mismatch between this value, the dataset view, evaluation plan, Runner config
  and prediction attempt fails before model/dataloader construction;
- mutating a normalized GT value, its source annotation hash, the manifest hash or
  source split hash fails before evaluation; prediction and GT sample ID sets must
  be exactly equal in manifest order.
- `evaluate_complete()` returns one `EvaluationBundle` whose raw evaluator payload
  contains per-threshold TP/FP/score/match traces and whose flat metric mapping is
  the MMEngine-compatible percentage view; the two views are derived in one call.
  Top-level bundle metadata binds evaluator version, protocol, temporal manifest,
  resolved split and manifest-ordered sample-ID digest, and `bundle_sha256`
  authenticates the canonical payload excluding only itself.

- [ ] **Step 2: Run metric tests and confirm failure**

Run: `python -m pytest tests/resilient_v2x/test_metric.py -q`

Expected: FAIL because the new evaluator is absent.

- [ ] **Step 3: Implement float64 polygon clipping and matching**

Do not depend on Shapely. Convert each box to four counter-clockwise BEV corners, clip one convex polygon by the other, and compute shoelace area in float64. Reject non-positive dimensions and non-finite inputs.

AP must use:

```python
recall_with_sentinel = np.concatenate(([0.0], recall, [1.0]))
precision_with_sentinel = np.concatenate(([0.0], precision, [0.0]))
for index in range(precision_with_sentinel.size - 2, -1, -1):
    precision_with_sentinel[index] = max(
        precision_with_sentinel[index],
        precision_with_sentinel[index + 1],
    )
change = np.flatnonzero(
    recall_with_sentinel[1:] != recall_with_sentinel[:-1]
)
ap = np.sum(
    (recall_with_sentinel[change + 1] - recall_with_sentinel[change])
    * precision_with_sentinel[change + 1],
    dtype=np.float64,
)
```

`sample_prediction_from_datasample()` uses
`LiDARInstance3DBoxes.bottom_center` for xyz-bottom and `dims` for
length/width/height; it never assumes the tensor's z origin. It maps only label 0
to `Car`. At construction, the Metric calls
`ground_truth_from_temporal_manifest()` and freezes the verified
manifest-ordered ground truth in memory. `process()` converts current Runner
predictions in memory and appends their serializable forms for the standard
BaseMetric path. `evaluate_complete()` consumes the complete canonical
`SamplePrediction` sequence provided by `ResumableTestLoop`, including verified
samples from an earlier resume segment, and returns the sole
`EvaluationBundle`. `compute_metrics()` is only the standard MMEngine adapter:
it converts its complete serializable results to `SamplePrediction`, delegates
once to `evaluate_complete()`, asserts exact ordered `METRIC_KEYS`, and returns
`bundle.metrics`. All four values are always computed, including nonzero latency
runs, so downstream “at least BEV AP@0.7” tables never need an alternate
evaluator. Neither method opens
a prediction/evidence path. Remove any access to `last_checkpoint`,
`SUPPROTED_MODELS`, `eval_vic`, or a second model.

- [ ] **Step 4: Run metric tests**

Run: `python -m pytest tests/resilient_v2x/test_metric.py -q`

Expected: all golden tests pass.

- [ ] **Step 5: Commit**

```bash
git add transvision/evaluation/metrics/resilient_v2x_metric.py transvision/evaluation/metrics/__init__.py transvision/evaluation/__init__.py tests/resilient_v2x/fixtures/metric_golden.json tests/resilient_v2x/test_metric.py
git commit -m "feat: add resilient v2x ap evaluator"
```

---

### Task 2: 定义 canonical IDs、schemas 和 compatibility contract

**Files:**

- Create: `transvision/experiments/__init__.py`
- Create: `transvision/experiments/resilient_v2x/__init__.py`
- Create: `transvision/experiments/resilient_v2x/protocol.py`
- Create: `transvision/experiments/resilient_v2x/schemas/template.schema.json`
- Create: `transvision/experiments/resilient_v2x/schemas/training_plan.schema.json`
- Create: `transvision/experiments/resilient_v2x/schemas/evaluation_plan.schema.json`
- Create: `transvision/experiments/resilient_v2x/schemas/model_provenance.schema.json`
- Create: `transvision/experiments/resilient_v2x/schemas/training_protocol.schema.json`
- Create: `tests/resilient_v2x/test_experiment_protocol.py`

**Interfaces:**

```python
class ProtocolError(RuntimeError):
    """The requested experiment violates the frozen protocol."""

@dataclass(frozen=True)
class TrainingProtocolDeclaration:
    schema_version: Literal[1]
    train_split: Literal["train"]
    dataset_release_sha256: str
    train_split_manifest_sha256: str
    train_temporal_manifest_sha256: str
    epochs: int
    global_batch_size: int
    seeds: tuple[int, ...]
    train_delay_values_ms: tuple[int, ...]
    lidar_fault_probability: float
    camera_fault_probability: float
    fault_scope: str
    checkpoint_selector: str
    evaluator_version: Literal["resilient-v2x-ap-v1"]
    iou_thresholds: tuple[float, float]
    detection_range: tuple[float, float, float, float, float, float]
    evidence_contract_version: Literal["resilient-v2x-evidence-v1"]
    optimizer_recipe_id: str
    optimizer_recipe_sha256: str
    augmentation_recipe_id: str
    augmentation_recipe_sha256: str
    sampler_recipe_id: str
    sampler_recipe_sha256: str
    accumulation_by_world_size: tuple[tuple[int, int], ...]
    update_scheduler_id: str
    update_scheduler_sha256: str
    precision: Literal["amp"]
    gradient_clip_norm: float
    nonfinite_policy: Literal["fail"]
    resolved_config_sha256: str
    protocol_sha256: str
    deviations: tuple[str, ...]

@dataclass(frozen=True)
class TrainingProtocolAssessment:
    recomputed_protocol_sha256: str
    realized_checkpoint_sha256: str
    verified_fields: tuple[str, ...]
    reported_deviations: tuple[str, ...]
    blocking_deviations: tuple[str, ...]
    controlled_eligible: bool

@dataclass(frozen=True)
class TrainingProtocolVerification:
    accepted_training_manifest_sha256: str
    accepted_declaration: TrainingProtocolDeclaration
    assessment: TrainingProtocolAssessment

@dataclass(frozen=True)
class CapabilityDeclaration:
    agents: tuple[Literal["ego", "rsu"], ...]
    modalities: tuple[Literal["lidar", "camera"], ...]
    history_positions: int
    supports_faults: tuple[Literal["L-Fail", "C-Fail"], ...]
    supports_latency_ms: tuple[int, ...]
    input_contract_version: str
    output_contract_version: str

@dataclass(frozen=True)
class ProtocolRequirement:
    capability_class: Literal["multimodal", "lidar_only"]
    required_agents: tuple[Literal["ego", "rsu"], ...]
    required_modalities: tuple[Literal["lidar", "camera"], ...]
    history_positions: int
    condition: Literal["Full", "L-Fail", "C-Fail"]
    latency_ms: int
    fault_semantics: Literal[
        "none",
        "global_target_tick",
        "consecutive_duration",
        "arrival_relative",
    ]
    fault_scope: Literal["none", "ego", "rsu", "both"]
    consecutive_fault_ticks: int
    arrival_relative_source_rule: Literal["pre_mask_selected_source_once"] | None
    split: Literal["val", "test"]
    detection_range: tuple[float, float, float, float, float, float]
    evaluator_version: Literal["resilient-v2x-ap-v1"]
    iou_thresholds: tuple[float, float]
    input_contract_version: str
    output_contract_version: str

@dataclass(frozen=True)
class CompatibilityReport:
    compatible: bool
    classification: Literal["controlled", "cross_protocol", "incompatible"]
    reasons: tuple[str, ...]
    reported_deviations: tuple[str, ...]
    blocking_deviations: tuple[str, ...]

@dataclass(frozen=True)
class ExperimentTemplate:
    schema_version: Literal[1]
    matrix_name: Literal[
        "main",
        "ablation",
        "sensitivity",
        "duration",
        "arrival_relative",
        "concat",
        "baseline",
    ]
    result_id_slot: str
    method: str
    variant: str
    capability_class: Literal["multimodal", "lidar_only"]
    condition: Literal["Full", "L-Fail", "C-Fail"]
    latency_ms: int
    fault_probability: float
    fault_semantics: Literal[
        "none",
        "global_target_tick",
        "consecutive_duration",
        "arrival_relative",
    ]
    fault_scope: Literal["none", "ego", "rsu", "both"]
    consecutive_fault_ticks: int
    arrival_relative_source_rule: Literal["pre_mask_selected_source_once"] | None
    seed: Literal[0, 1, 2]
    split: Literal["val", "test"]

@dataclass(frozen=True)
class BuiltinTrainingSourcePlan:
    kind: Literal["builtin"]
    orchestrator_code_commit: str
    orchestrator_tree: str

@dataclass(frozen=True)
class ExternalTrainingSourcePlan:
    kind: Literal["external_local"]
    orchestrator_code_commit: str
    resolved_manifest_sha256: str
    repository_commit: str
    repository_tree: str
    submodule_commits: tuple[tuple[str, str], ...]

TrainingSourcePlan = BuiltinTrainingSourcePlan | ExternalTrainingSourcePlan

@dataclass(frozen=True)
class TrainingPlan:
    schema_version: Literal[1]
    method: str
    variant: str
    role: Literal["teacher", "student", "baseline"]
    seed: Literal[0, 1, 2]
    teacher_checkpoint_role: Literal[
        "none",
        "initialization_only",
        "initialization_and_distillation",
    ]
    teacher_training_plan_id: str | None
    dataset_release_sha256: str
    split_manifest_sha256: str
    temporal_manifest_sha256: str
    transport_overlay_sha256: str
    fault_overlay_sha256: str
    resolved_config_sha256: str
    training_protocol_sha256: str
    source_plan: TrainingSourcePlan
    environment_lock_sha256: str
    hardware_request_sha256: str
    world_size: Literal[1, 2, 4]

@dataclass(frozen=True)
class TrainingRealization:
    model_initialization_sha256: str
    resolved_teacher_checkpoint_sha256: str | None
    source_snapshot_sha256: str
    environment_manifest_sha256: str
    actual_hardware_fingerprint_sha256: str
    launcher: Literal["none", "pytorch"]
    world_size: Literal[1, 2, 4]

@dataclass(frozen=True)
class LocalTrainingProvenancePlanRef:
    kind: Literal["local_training"]
    model_provenance_plan_id: str
    training_plan_id: str

@dataclass(frozen=True)
class UpstreamCheckpointProvenancePlanRef:
    kind: Literal["upstream_checkpoint"]
    model_provenance_plan_id: str
    upstream_manifest_sha256: str
    repository_commit: str
    declared_checkpoint_sha256: str

ModelProvenancePlanRef = (
    LocalTrainingProvenancePlanRef | UpstreamCheckpointProvenancePlanRef
)

@dataclass(frozen=True)
class EvaluationPlan:
    schema_version: Literal[1]
    method: str
    variant: str
    capability_class: Literal["multimodal", "lidar_only"]
    model_provenance: ModelProvenancePlanRef
    checkpoint_selector: str
    condition: Literal["Full", "L-Fail", "C-Fail"]
    latency_ms: int
    fault_semantics: Literal[
        "none",
        "global_target_tick",
        "consecutive_duration",
        "arrival_relative",
    ]
    fault_scope: Literal["none", "ego", "rsu", "both"]
    consecutive_fault_ticks: int
    arrival_relative_source_rule: Literal["pre_mask_selected_source_once"] | None
    seed: Literal[0, 1, 2]
    split: Literal["val", "test"]
    dataset_release_sha256: str
    split_manifest_sha256: str
    temporal_manifest_sha256: str
    transport_overlay_sha256: str
    fault_overlay_sha256: str
    resolved_config_sha256: str
    protocol_sha256: str
    code_commit: str
    environment_lock_sha256: str
    hardware_request_sha256: str
    evaluator_version: Literal["resilient-v2x-ap-v1"]
    iou_thresholds: tuple[float, float]
    detection_range: tuple[float, float, float, float, float, float]
    world_size: Literal[1, 2, 4]

@dataclass(frozen=True)
class LocalEvaluationRealization:
    kind: Literal["local_training"]
    checkpoint_sha256: str
    selected_training_attempt_id: str
    prediction_source_snapshot_sha256: str
    adapter_execution_environment_manifest_sha256: str
    adapter_execution_actual_hardware_fingerprint_sha256: str
    shared_evaluator_environment_manifest_sha256: str
    shared_evaluator_actual_hardware_fingerprint_sha256: str
    launcher: Literal["none", "pytorch"]
    world_size: Literal[1, 2, 4]

@dataclass(frozen=True)
class UpstreamEvaluationRealization:
    kind: Literal["upstream_checkpoint"]
    checkpoint_sha256: str
    upstream_manifest_sha256: str
    prediction_source_snapshot_sha256: str
    adapter_execution_environment_manifest_sha256: str
    adapter_execution_actual_hardware_fingerprint_sha256: str
    shared_evaluator_environment_manifest_sha256: str
    shared_evaluator_actual_hardware_fingerprint_sha256: str
    launcher: Literal["none", "pytorch"]
    world_size: Literal[1, 2, 4]

EvaluationRealization = LocalEvaluationRealization | UpstreamEvaluationRealization

def canonical_id(namespace: str, payload: Mapping[str, object]) -> str
def training_protocol_sha256(declaration: TrainingProtocolDeclaration) -> str
def template_id(template: ExperimentTemplate) -> str
def training_plan_id(plan: TrainingPlan) -> str
def local_model_provenance_plan_id(training_plan_id: str) -> str
def upstream_model_provenance_plan_id(
    upstream_manifest_sha256: str,
    repository_commit: str,
    declared_checkpoint_sha256: str,
) -> str
def materialized_training_id(plan: TrainingPlan, realization: TrainingRealization) -> str
def evaluation_plan_id(plan: EvaluationPlan) -> str
def materialized_evaluation_id(
    plan: EvaluationPlan,
    realization: EvaluationRealization,
) -> str
def condition_display_name(
    capability_class: Literal["multimodal", "lidar_only"],
    condition: Literal["Full", "L-Fail", "C-Fail"],
) -> Literal["Full", "L-Fail", "C-Fail", "LiDAR-Full", "LiDAR-Fail"]
def parse_condition_display_name(
    capability_class: Literal["multimodal", "lidar_only"],
    display_name: str,
) -> Literal["Full", "L-Fail", "C-Fail"]
def determine_compatibility(
    candidate: CapabilityDeclaration,
    required: ProtocolRequirement,
    training: TrainingProtocolDeclaration | None = None,
    training_evidence: TrainingProtocolVerification | None = None,
) -> CompatibilityReport
def validate_schema(instance: Mapping[str, object], schema_name: str) -> None
```

- [ ] **Step 1: Write fixed ID and mutation tests**

```python
def test_canonical_id_fixed_vector() -> None:
    payload = {
        "method": "resilient_v2x",
        "variant": "full",
        "condition": "Full",
        "latency_ms": 0,
        "seed": 0,
        "split": "val",
        "result_id_slot": "main.full.0ms.seed0",
    }
    assert canonical_id("evaluation-template-v1", payload) == (
        "evaluation-template-v1:"
        "75be6ff82974f952d72918a32c6d7107de51f62b67afe65b3c3b1cba1597ccee"
    )


@pytest.mark.parametrize(
    "field",
    [
        "resolved_config_sha256",
        "transport_overlay_sha256",
        "fault_overlay_sha256",
        "dataset_release_sha256",
        "temporal_manifest_sha256",
        "source_plan",
        "environment_lock_sha256",
        "world_size",
    ],
)
def test_training_plan_id_changes_for_provenance_field(field: str) -> None:
    original = make_training_plan()
    changed = dataclasses.replace(original, **{field: different_value(field)})
    assert training_plan_id(original) != training_plan_id(changed)
```

Add tests for `allow_nan=false`, teacher checkpoint roles, plan/materialized
separation, local/upstream provenance union, LiDAR-only capability mismatch for
Camera conditions, evaluator version mismatch, and training declarations whose
actual accepted manifest disagrees on epochs/global batch/sampler/update count/
augmentation/precision/config digest, dataset/train split/temporal identity,
evaluator/IoU/range, or evidence-contract version. A declaration without accepted evidence is
never controlled; upstream checkpoint evidence remains cross-protocol. Template identity must include
`result_id_slot`; training/evaluation plan identity must deliberately exclude
`matrix_name` and `result_id_slot`, so two paper slots may reference one immutable
plan/run without becoming one template.
Role validation requires `teacher_training_plan_id` and realized teacher
checkpoint hash exactly for `initialization_only` or
`initialization_and_distillation`, and forbids both for `none`. Mutating the
always-required model-initialization hash or conditionally required teacher hash
must change `materialized_training_id`. Evaluation provenance schemas use a
discriminated union: local requires training plan/accepted attempt; upstream
requires manifest/repository/checkpoint provenance and forbids invented local
training IDs. Upstream remains cross-protocol even when its evaluation artifacts
are complete.

Training source identity is another discriminated union. Built-ins bind the
orchestrator commit; an external-local plan additionally binds the exact resolved
manifest hash, repository commit/tree and sorted submodule commits. Its
realization binds the materialized snapshot hash. Mutation tests change every
source field independently and prove `training_plan_id` or
`materialized_training_id` changes even when config and initialization bytes are
identical. The same typed source plan plus snapshot hash must match
`TrainingLaunch`, `TrainingAttemptManifest` and accepted model provenance; no
adapter may substitute the orchestrator commit for the external source commit.

All schemas, IDs, overlays, claims, and aggregation keys use only canonical
conditions `Full|L-Fail|C-Fail`. For `capability_class="lidar_only"`,
`condition_display_name()` is the sole presentation mapping:
`Full -> LiDAR-Full` and `L-Fail -> LiDAR-Fail`; `C-Fail` is incompatible.
`parse_condition_display_name()` provides the exact inverse only at UI/export
boundaries. Tests reject display strings in protocol schemas and verify canonical
ID stability plus round-trip display mapping, so `LiDAR-Fail` can never become a
fourth execution condition.

Template, requirement, and evaluation-plan validation fixes the fault fields as a
tagged union:

- `Full` requires `fault_semantics="none"`, `fault_scope="none"`,
  `consecutive_fault_ticks=0`, and no arrival-relative source rule;
- ordinary `L-Fail`/`C-Fail` rows require `global_target_tick`, scope `both`,
  zero duration, and no arrival-relative source rule;
- duration rows require `L-Fail`, `consecutive_duration`, scope `both`, a duration
  in `{1,2,3,4}`, and no arrival-relative source rule;
- arrival-relative rows require `L-Fail` or `C-Fail`, scope exactly `ego` or
  `rsu`, zero duration, and
  `arrival_relative_source_rule="pre_mask_selected_source_once"`.

Every tagged field participates in template/evaluation plan IDs and overlay
hashes. This prevents E-only and R-only diagnostics—or a per-sample pre-mask
selected source—from collapsing into a global target-tick failure.

- [ ] **Step 2: Run tests and confirm failure**

Run: `python -m pytest tests/resilient_v2x/test_experiment_protocol.py -q`

Expected: FAIL because experiment protocol modules are absent.

- [ ] **Step 3: Implement domain-separated IDs and strict schemas**

IDs use:

```python
digest = hashlib.sha256(
    namespace.encode("utf-8") + b"\x00" + canonical_json_bytes(payload)
).hexdigest()
return f"{namespace}:{digest}"
```

Every schema must set `additionalProperties: false`, require its version, and prohibit a `successful` state without all immutable hashes. Upstream provenance must always default compatibility to `cross_protocol` until shared-protocol training evidence is supplied.

`TrainingProtocolDeclaration` is owned by this protocol module; baseline adapters
import it rather than defining a competing type. `determine_compatibility()` checks
capabilities and the full declaration, then requires
`TrainingProtocolVerification` derived from an immutable accepted local-training
manifest before returning controlled eligibility. It recomputes deviations from
realized config/sampler/update/augmentation/checkpoint evidence and rejects a
falsely empty user declaration.
`TrainingProtocolDeclaration.protocol_sha256` is a domain-separated hash of its
canonical payload excluding only that field. The accepted training manifest
embeds the full declaration plus `TrainingProtocolAssessment`; after finalization
the outer verification repeats that typed declaration and must recompute the same
digest. Compatibility compares typed values from the accepted declaration—not an
opaque caller hash or `verified_fields` strings—against the requested dataset,
train split, temporal data, evaluator, IoU, range, evidence, optimizer, sampler,
augmentation, precision and update contracts. A mutation of any shared field
forces cross-protocol or incompatible classification.
`training_protocol_sha256()` uses the literal `training-protocol-v1` domain;
strict schema load recomputes it. A fixed vector and mutation of every declaration
field prevent an external manifest from supplying an arbitrary well-formed digest.
Shared blocking fields are split, 50 epochs, global batch 4, seeds, fault/delay
sampling, checkpoint selection, evaluator and evidence contract. Resilient and
capacity concat additionally require the exact section-10 optimizer recipe.
Other baselines may retain a verified public optimizer/augmentation/sampler recipe:
differences from Resilient are reported but non-blocking, while missing recipe
hashes or differences from their own declared/accepted training evidence are
blocking.

`template_id()` includes the stable paper `result_id_slot`. `training_plan_id()` and
`evaluation_plan_id()` contain only execution semantics from the approved spec;
they exclude presentation-only matrix/slot names. Materialized IDs add only the
specified realization fields: training adds model-initialization hash,
condition-dependent teacher checkpoint hash, realized environment/hardware,
launcher, world size and exact source-snapshot hash; evaluation adds the
discriminated local/upstream
checkpoint provenance plus the prediction-source snapshot, adapter-execution
environment/hardware and shared-evaluator environment/hardware. Built-ins require
the two environment pairs to match; an external adapter must preserve both
distinct accepted identities. Reuse therefore occurs
at plan/materialized ID level and never by erasing a distinct result-slot template.
Local model-provenance plan IDs are domain-separated hashes of the full canonical
training plan ID; upstream IDs hash the strict manifest, repository commit, and
declared checkpoint hash. The ID embedded in either provenance union member must
recompute exactly, so a planner never keys checkpoint provenance by a
collision-prone method/variant/seed tuple.

- [ ] **Step 4: Run protocol tests**

Run: `python -m pytest tests/resilient_v2x/test_experiment_protocol.py -q`

Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add transvision/experiments tests/resilient_v2x/test_experiment_protocol.py
git commit -m "feat: define resilient v2x experiment identities"
```

---

### Task 3: 枚举 template coverage 和 materializable plan DAG

**Files:**

- Create: `transvision/experiments/resilient_v2x/matrix.py`
- Create: `tests/resilient_v2x/fixtures/matrix_protocol.json`
- Create: `tests/resilient_v2x/test_matrix.py`

**Interfaces:**

```python
CAPACITY_CONCAT_PROFILE_SLOT = "baseline.concat.full.0ms.seed0"

@dataclass(frozen=True)
class MatrixRequest:
    matrix: Literal[
        "main",
        "ablation",
        "sensitivity",
        "duration",
        "arrival_relative",
        "concat",
        "all",
        "baselines",
        "reproduction",
        "smoke",
    ]
    split: Literal["val", "test"]
    method: str | None
    variant: str | None
    condition: Literal["Full", "L-Fail", "C-Fail"] | None
    latency_ms: int | None
    seed: Literal[0, 1, 2] | None
    smoke_max_epochs: int | None

@dataclass(frozen=True)
class MaterializationContext:
    data_root: Path
    temporal_manifest: Path
    evidence_root: Path
    environment_manifest: Path
    dataset_release_sha256: str
    split_manifest_sha256: str
    temporal_manifest_sha256: str
    code_commit: str
    environment_lock_sha256: str
    hardware_request_sha256: str
    world_size: Literal[1, 2, 4]
    provenance_by_model_provenance_plan_id: Mapping[
        str,
        ModelProvenancePlanRef,
    ]
    baseline_manifests: Mapping[str, tuple[Path, str]]

@dataclass(frozen=True)
class TemplateBinding:
    template_id: str
    result_id_slot: str

@dataclass(frozen=True)
class PlanNode:
    plan_id: str | None
    template_bindings: tuple[TemplateBinding, ...]
    kind: Literal["train", "evaluate", "aggregate", "profile", "qualitative", "export"]
    dependencies: tuple[str, ...]
    state: Literal["ready", "blocked"]
    blocked_reason: str | None
    payload: Mapping[str, object]

def enumerate_templates(matrix: MatrixRequest) -> tuple[ExperimentTemplate, ...]
def build_plan_dag(
    templates: Sequence[ExperimentTemplate],
    context: MaterializationContext | None,
) -> tuple[PlanNode, ...]
def validate_dag(nodes: Sequence[PlanNode]) -> None
def expected_counts(nodes: Sequence[PlanNode]) -> dict[str, int]
```

- [ ] **Step 1: Write exact coverage and de-duplication tests**

```python
def test_controlled_matrix_counts_after_reuse() -> None:
    nodes = build_controlled_matrix_fixture()
    counts = expected_counts(nodes)
    assert counts["teacher_training"] == 3
    assert counts["unique_student_training"] == 33
    assert counts["resilient_training"] == 36
    assert counts["concat_training"] == 3
    assert counts["all_training"] == 39
    assert counts["main_evaluation"] == 36
    assert counts["non_full_ablation_evaluation"] == 63
    assert counts["new_sensitivity_evaluation"] == 27
    assert counts["duration_evaluation"] == 12
    assert counts["arrival_relative_evaluation"] == 24
    assert counts["resilient_unique_val_evaluation"] == 162
    assert counts["concat_val_evaluation"] == 36
    assert counts["all_val_evaluation"] == 198
    assert counts["resilient_reserved_test_main_evaluation"] == 36
    assert counts["concat_reserved_test_evaluation"] == 36
    assert counts["all_reserved_test_evaluation"] == 72
    assert counts["all_evaluation"] == 270


def test_result_slots_remain_distinct_while_execution_is_reused() -> None:
    templates = enumerate_all_resilient_templates()
    main = find_template(
        templates,
        matrix="main",
        variant="full",
        corruption_probability=0.3,
        condition="Full",
        latency_ms=0,
    )
    ablation_full = find_template(
        templates,
        matrix="ablation",
        variant="full",
        corruption_probability=0.3,
        condition="Full",
        latency_ms=0,
    )
    sensitivity = find_template(
        templates,
        matrix="sensitivity",
        corruption_probability=0.3,
        condition="Full",
        latency_ms=0,
    )
    slots = (main, ablation_full, sensitivity)
    assert len({template.result_id_slot for template in slots}) == 3
    assert len({template_id(template) for template in slots}) == 3
    assert len({
        training_plan_id(training_plan_from_template(template))
        for template in slots
    }) == 1
    assert len({
        evaluation_plan_id(plan_from_template(template))
        for template in slots
    }) == 1
    training_realization = fixture_training_realization()
    assert len({
        materialized_training_id(
            training_plan_from_template(template),
            training_realization,
        )
        for template in slots
    }) == 1
    realization = fixture_evaluation_realization()
    assert len({
        materialized_evaluation_id(plan_from_template(template), realization)
        for template in slots
    }) == 1
    reused_node = find_evaluation_node_for_templates(build_controlled_matrix_fixture(), slots)
    assert tuple(
        binding.result_id_slot for binding in reused_node.template_bindings
    ) == tuple(sorted(template.result_id_slot for template in slots))
```

Add tests for:

- main 36 template slots;
- seven non-full variants × three seeds × three conditions;
- duration 12 and arrival-relative 24;
- the 24 arrival-relative slots are the Cartesian product of two scopes
  `(ego,rsu)`, two modality conditions `(L-Fail,C-Fail)`, two latencies
  `(0,300)`, and three seeds. E-only and R-only produce distinct template,
  evaluation-plan, overlay, and result-slot IDs, while all reuse the accepted
  full student checkpoint for their seed;
- invalid tagged combinations such as `arrival_relative+both`,
  `global_target_tick+ego`, a nonzero duration outside the duration matrix, or an
  integer pretending to be a per-sample arrival-relative source tick are rejected;
- capacity-matched concat has three `teacher_checkpoint_role=none` training plans,
  36 val evaluations and 36 reserved-test evaluations, with one checkpoint per seed
  reused across all conditions and splits;
- exactly one val template has result slot
  `CAPACITY_CONCAT_PROFILE_SLOT`; it resolves capacity-matched concat seed 0,
  `Full@0 ms`, and the accepted profiling checkpoint. No other template may reuse
  that slot, and the literal is included in `matrix_protocol.json` as a golden;
- main, ablation-full and sensitivity `p=0.3` keep distinct result-slot template IDs
  and remain as three sorted `TemplateBinding` records on the reused node while
  sharing the same applicable training/evaluation plan and materialized IDs;
- no-data dry-run emits template IDs plus blocked dataset dependency but no fake plan IDs;
- checkpoint dependency shared across conditions;
- two variants—or two training probabilities—of the same method and seed bind
  different provenance/checkpoint records through the canonical
  `model_provenance_plan_id`; tuple lookup by method/variant/seed is prohibited
  because sensitivity jobs can still collide. The mapping key must exactly equal
  the ID carried by its value, and a missing/duplicate ID blocks materialization;
- LiDAR-only 24 per split and multimodal 36 per split;
- cycle, duplicate plan ID, and missing dependency rejection.

- [ ] **Step 2: Run matrix tests and confirm failure**

Run: `python -m pytest tests/resilient_v2x/test_matrix.py -q`

Expected: FAIL because `matrix.py` is absent.

- [ ] **Step 3: Implement pure deterministic enumeration**

Sort every dimension explicitly; never iterate over a set. Node dependencies are:

```text
environment/data/code freeze
  -> teacher training
  -> student training
  -> selected checkpoint
  -> val evaluation
  -> val aggregate / qualitative

environment/data/code freeze
  -> capacity-matched concat training (three seeds, no teacher dependency)
  -> selected concat checkpoint
  -> 36 val evaluations
  -> paired full-vs-concat profiling

all val aggregate / qualitative / paired profile accepted
  -> architecture/config/checkpoint submission freeze
  -> full test-claim acquisition
  -> reserved-test evaluations
  -> test aggregate

accepted val aggregate / qualitative / profile + accepted test aggregate
  -> evidence export
```

Nodes are de-duplicated only by non-null execution `plan_id`; every contributing
template remains in a stable, sorted `template_bindings` tuple so reuse cannot
erase a paper result slot. Baseline nodes use manifest validation before
prepare/train/predict/parser/evaluator. A condition never creates a new training
node unless training protocol itself differs.
Within this plan, `all_training=39` means 36 Resilient teacher/student jobs plus
three capacity-matched concat jobs; it intentionally excludes the six adapters
implemented by the separate baseline plan. `all_val_evaluation=198` is
162 Resilient val runs plus 36 concat val runs. The frozen reserved-test scope is
36 Resilient main runs plus 36 concat runs, for 72; the combined evaluation total is
270. Smoke diagnostics are excluded from every count above.

- [ ] **Step 4: Run matrix tests**

Run: `python -m pytest tests/resilient_v2x/test_matrix.py -q`

Expected: all count, reuse, and DAG tests pass.

- [ ] **Step 5: Commit**

```bash
git add transvision/experiments/resilient_v2x/matrix.py tests/resilient_v2x/fixtures/matrix_protocol.json tests/resilient_v2x/test_matrix.py
git commit -m "feat: enumerate resilient v2x experiment dag"
```

---

### Task 4: 实现 immutable AttemptStore、artifact manifest 和 hash-chained ledgers

**Files:**

- Create: `transvision/experiments/resilient_v2x/evidence.py`
- Create: `transvision/experiments/resilient_v2x/schemas/training_manifest.schema.json`
- Create: `transvision/experiments/resilient_v2x/schemas/run_manifest.schema.json`
- Create: `transvision/experiments/resilient_v2x/schemas/ledger_record.schema.json`
- Create: `transvision/experiments/resilient_v2x/schemas/test_claim.schema.json`
- Create: `transvision/experiments/resilient_v2x/schemas/submission_freeze.schema.json`
- Create: `transvision/experiments/resilient_v2x/schemas/prediction_progress.schema.json`
- Create: `transvision/experiments/resilient_v2x/schemas/evaluation_bundle.schema.json`
- Create: `transvision/experiments/resilient_v2x/schemas/inference_diagnostics.schema.json`
- Create: `tests/resilient_v2x/test_evidence.py`

**Interfaces:**

```python
class EvidenceIntegrityError(ProtocolError):
    """Immutable evidence is missing, conflicting, or hash-invalid."""

@dataclass(frozen=True)
class ProtocolEligibility:
    protocol_namespace: str
    controlled: bool
    paper_eligible: bool
    reserved_test_eligible: bool

@dataclass(frozen=True)
class ArtifactRecord:
    relative_path: str
    byte_size: int
    sha256: str

@dataclass(frozen=True)
class ArtifactManifest:
    schema_version: Literal[1]
    attempt_id: str
    materialized_id: str
    eligibility: ProtocolEligibility
    artifacts: tuple[ArtifactRecord, ...]
    detached_manifest_sha256: str

@dataclass(frozen=True)
class AttemptHandle:
    kind: Literal["training", "evaluation", "profiling", "qualitative", "export"]
    materialized_id: str
    attempt_id: str
    eligibility: ProtocolEligibility
    path: Path

@dataclass(frozen=True)
class LedgerRecord:
    schema_version: Literal[1]
    sequence: int
    previous_record_sha256: str | None
    payload: Mapping[str, object]
    record_sha256: str

@dataclass(frozen=True)
class LocalModelProvenance:
    kind: Literal["local_training"]
    model_provenance_plan_id: str
    training_plan_id: str
    selected_training_attempt_id: str
    checkpoint_sha256: str
    source_plan: TrainingSourcePlan
    source_snapshot_sha256: str

@dataclass(frozen=True)
class UpstreamModelProvenance:
    kind: Literal["upstream_checkpoint"]
    model_provenance_plan_id: str
    upstream_manifest_sha256: str
    repository_commit: str
    repository_tree: str
    submodule_commits: tuple[tuple[str, str], ...]
    checkpoint_sha256: str
    source_snapshot_sha256: str

ModelProvenance = LocalModelProvenance | UpstreamModelProvenance

class AttemptStore:
    def __init__(
        self,
        evidence_root: Path,
        kind: Literal["training", "evaluation", "profiling", "qualitative", "export"],
    ) -> None
    def create_attempt(
        self,
        materialized_id: str,
        eligibility: ProtocolEligibility,
        allow_force: bool,
    ) -> AttemptHandle
    def finalize_manifest(self, attempt: AttemptHandle) -> ArtifactManifest
    def mark_success(self, attempt: AttemptHandle) -> None
    def accepted_attempt(self, materialized_id: str) -> AttemptHandle | None

class HashChainedLedger:
    def append(self, payload: Mapping[str, object]) -> LedgerRecord
    def verify(self) -> tuple[LedgerRecord, ...]

@dataclass(frozen=True)
class TestPlanAttempt:
    evaluation_plan_id: str
    materialized_evaluation_id: str
    attempt_id: str
    condition: str
    latency_ms: int
    fault_semantics: Literal[
        "none",
        "global_target_tick",
        "consecutive_duration",
        "arrival_relative",
    ]
    fault_scope: Literal["none", "ego", "rsu", "both"]
    consecutive_fault_ticks: int
    arrival_relative_source_rule: Literal["pre_mask_selected_source_once"] | None
    seed: int
    checkpoint_sha256: str
    resolved_config_sha256: str
    protocol_sha256: str
    transport_overlay_sha256: str
    fault_overlay_sha256: str

@dataclass(frozen=True)
class CoreSubmissionFreeze:
    schema_version: Literal[1]
    component: Literal["core"]
    val_aggregate_index_id: str
    val_aggregate_index_sha256: str
    profiling_success_sha256: str
    qualitative_success_sha256: str
    checkpoint_sha256s: tuple[str, ...]
    code_commit: str
    protocol_sha256s: tuple[str, ...]
    deployment_config_sha256s: tuple[str, ...]
    freeze_sha256: str

@dataclass(frozen=True)
class BaselineSubmissionFreeze:
    schema_version: Literal[1]
    component: Literal["baselines"]
    core_freeze_sha256: str
    val_aggregate_index_id: str
    val_aggregate_index_sha256: str
    checkpoint_sha256s: tuple[str, ...]
    code_commit: str
    protocol_sha256s: tuple[str, ...]
    deployment_config_sha256s: tuple[str, ...]
    freeze_sha256: str

@dataclass(frozen=True)
class ReproductionSubmissionFreeze:
    schema_version: Literal[1]
    component: Literal["reproduction"]
    core_freeze_sha256: str
    baselines_freeze_sha256: str
    val_aggregate_index_id: str
    val_aggregate_index_sha256: str
    freeze_sha256: str

SubmissionFreeze = (
    CoreSubmissionFreeze
    | BaselineSubmissionFreeze
    | ReproductionSubmissionFreeze
)

@dataclass(frozen=True)
class TestClaim:
    schema_version: Literal[1]
    test_matrix_plan_id: str
    method: str
    variant: str
    capability_class: Literal["multimodal", "lidar_only"]
    expected_run_count: Literal[36, 24]
    dataset_release_sha256: str
    split_manifest_sha256: str
    temporal_manifest_sha256: str
    test_matrix_plan_sha256: str
    submission_component: Literal["core", "baselines"]
    submission_freeze_sha256: str
    code_commit: str
    environment_lock_sha256: str
    hardware_request_sha256: str
    prediction_source_snapshot_sha256: str
    adapter_execution_environment_manifest_sha256: str
    adapter_execution_actual_hardware_fingerprint_sha256: str
    shared_evaluator_environment_manifest_sha256: str
    shared_evaluator_actual_hardware_fingerprint_sha256: str
    evaluator_version: Literal["resilient-v2x-ap-v1"]
    iou_thresholds: tuple[float, float]
    checkpoints_by_seed: tuple[tuple[int, str], ...]
    plan_attempts: tuple[TestPlanAttempt, ...]

@dataclass(frozen=True)
class TrainingAttemptManifest:
    schema_version: Literal[1]
    training_plan_id: str
    materialized_training_id: str
    attempt_id: str
    eligibility: ProtocolEligibility
    source_plan: TrainingSourcePlan
    source_snapshot_sha256: str
    model_initialization_sha256: str
    resolved_teacher_checkpoint_sha256: str | None
    environment_manifest_sha256: str
    actual_hardware_fingerprint_sha256: str
    resolved_config_sha256: str
    training_protocol_sha256: str
    training_protocol: TrainingProtocolDeclaration
    transport_overlay_sha256: str
    fault_overlay_sha256: str
    command_sha256: str
    expected_epochs: int
    completed_epochs: int
    global_batch_size: int
    completed_update_steps: int
    sampler_trace_sha256: str
    augmentation_trace_sha256: str
    precision: Literal["amp"]
    nonfinite_count: Literal[0]
    checkpoint_selector: str
    selected_checkpoint_relative_path: str
    selected_checkpoint_sha256: str
    protocol_assessment: TrainingProtocolAssessment

@dataclass(frozen=True)
class AcceptedTrainingRun:
    training_plan_id: str
    materialized_training_id: str
    attempt_id: str
    attempt_path: Path
    training_manifest_sha256: str
    artifact_manifest_sha256: str
    selected_checkpoint_path: Path
    selected_checkpoint_sha256: str
    protocol_verification: TrainingProtocolVerification

class PredictAttempt:
    def __init__(
        self,
        attempt_dir: Path,
        attempt_id: str,
        split: Literal["val", "test"],
        protocol_sha256: str,
        temporal_manifest_sha256: str,
    ) -> None
    def initialize(
        self,
        expected_sample_ids: Sequence[str],
        command: Sequence[str],
        world_size: int,
        eligibility: ProtocolEligibility,
    ) -> None
    def record_rank_sample(
        self,
        rank: int,
        manifest_index: int,
        sample_id: str,
        prediction: SamplePrediction,
        diagnostics: SampleInferenceDiagnostics,
    ) -> Path
    def verified_completed_sample_ids(self) -> tuple[str, ...]
    def remaining_sample_ids(self) -> tuple[str, ...]
    def promote_rank_spools(self) -> tuple[Path, ...]
    def load_complete_predictions(self) -> tuple[SamplePrediction, ...]
    def write_evaluation_artifacts(
        self,
        bundle: EvaluationBundle,
    ) -> tuple[Path, Path]

@dataclass(frozen=True)
class TrainingFinalizationState:
    attempt_id: str
    completed_phases: tuple[
        Literal[
            "training_manifest",
            "artifact_manifest",
            "realization_ledger",
            "success_pointer",
        ],
        ...,
    ]
    phase_record_sha256s: tuple[str, ...]

@dataclass(frozen=True)
class EvaluationFinalizationState:
    attempt_id: str
    bundle_sha256: str
    completed_phases: tuple[
        Literal[
            "evaluator_output",
            "metrics",
            "artifact_manifest",
            "success_pointer",
        ],
        ...,
    ]
    phase_record_sha256s: tuple[str, ...]

def finalize_evaluation_attempt(
    store: AttemptStore,
    attempt: AttemptHandle,
    prediction_attempt: PredictAttempt,
    bundle: EvaluationBundle,
) -> ArtifactManifest

def finalize_training_attempt(
    store: AttemptStore,
    attempt: AttemptHandle,
    training_manifest: TrainingAttemptManifest,
) -> AcceptedTrainingRun
def validate_artifact_manifest(path: Path, schema_name: str) -> ArtifactManifest
def validate_training_manifest(path: Path) -> TrainingAttemptManifest
def accepted_training_run(
    evidence_root: Path,
    training_plan_id: str,
) -> AcceptedTrainingRun | None
def inference_diagnostics_from_datasample(
    sample_id: str,
    data_sample: Det3DDataSample,
) -> SampleInferenceDiagnostics
def validate_model_provenance(path: Path) -> ModelProvenance
def assert_no_orphan_test_artifacts(evidence_root: Path) -> None
def acquire_test_claim(ledger: HashChainedLedger, claim: TestClaim) -> LedgerRecord
def acquire_submission_freeze(
    evidence_root: Path,
    component: Literal["core", "baselines", "reproduction"],
) -> SubmissionFreeze
def preflight_test_access(evidence_root: Path, claim: TestClaim) -> None
def accepted_test_claim_family(
    evidence_root: Path,
    test_matrix_plan_id: str,
) -> tuple[AttemptHandle, ...]
```

Each `SubmissionFreeze.freeze_sha256` excludes only itself and uses the exact
domain matching its discriminator:
`submission-core-freeze-v1`, `submission-baselines-freeze-v1`, or
`submission-reproduction-freeze-v1`. Checkpoint, protocol and deployment-config
hash tuples are canonically sorted and duplicate-free. Checked-in fixed vectors
plus mutation tests prove every aggregate/profile/qualitative/checkpoint/code/
protocol/config/component-freeze field changes the applicable digest.

- [ ] **Step 1: Write O_EXCL, immutability, hash, and ledger tests**

```python
def test_success_pointer_is_create_exclusive(tmp_path: Path) -> None:
    store = AttemptStore(tmp_path, "evaluation")
    attempt = store.create_attempt(
        "evaluation-v1:abc",
        eligibility=controlled_val_eligibility(),
        allow_force=False,
    )
    write_complete_fake_artifacts(attempt)
    store.finalize_manifest(attempt)
    store.mark_success(attempt)
    with pytest.raises(FileExistsError):
        store.mark_success(attempt)


def test_manifest_excludes_itself_and_detached_hash(tmp_path: Path) -> None:
    store, attempt = complete_fake_attempt(tmp_path)
    manifest = store.finalize_manifest(attempt)
    paths = {artifact.relative_path for artifact in manifest.artifacts}
    assert "run_manifest.json" not in paths
    assert "run_manifest.sha256" not in paths
    serialized = (attempt.path / "run_manifest.json").read_bytes()
    assert manifest.detached_manifest_sha256 == hashlib.sha256(serialized).hexdigest()
    assert "detached_manifest_sha256" not in json.loads(serialized)


def test_ledger_detects_middle_record_mutation(tmp_path: Path) -> None:
    ledger = HashChainedLedger(tmp_path / "test_access_ledger.jsonl")
    ledger.append({"claim": "first"})
    ledger.append({"claim": "second"})
    mutate_first_record(tmp_path / "test_access_ledger.jsonl")
    with pytest.raises(EvidenceIntegrityError, match="hash chain"):
        ledger.verify()


def test_ledger_record_hash_fixed_vector(tmp_path: Path) -> None:
    first = HashChainedLedger(tmp_path / "ledger.jsonl").append({"claim": "first"})
    assert first.previous_record_sha256 is None
    assert first.record_sha256 == fixed_first_record_digest()


def test_test_claim_is_complete_before_first_prediction(tmp_path: Path) -> None:
    claim = complete_multimodal_test_claim()
    assert claim.expected_run_count == 36
    assert len(claim.plan_attempts) == 36
    assert tuple(seed for seed, _ in claim.checkpoints_by_seed) == (0, 1, 2)
    ledger = HashChainedLedger(tmp_path / "test_access_ledger.jsonl")
    record = acquire_test_claim(ledger, claim)
    assert record.payload["test_matrix_plan_id"] == claim.test_matrix_plan_id


def test_partial_test_claim_fails_before_predictor(tmp_path: Path, predictor_spy) -> None:
    claim = dataclasses.replace(
        complete_multimodal_test_claim(),
        plan_attempts=complete_multimodal_test_claim().plan_attempts[:-1],
    )
    with pytest.raises(EvidenceIntegrityError, match="complete claim family"):
        preflight_and_predict(tmp_path, claim, predictor_spy)
    predictor_spy.assert_not_called()
```

Add tests for:

- attempt directories never overwrite;
- attempt creation writes a canonical immutable `eligibility.json`, returns the
  same typed value on `AttemptHandle`, and `finalize_manifest()` requires an
  exact byte/hash match before copying it into `ArtifactManifest`. Eligibility
  cannot be inferred from a child artifact or supplied only at finalization;
- `--force` makes next attempt only for non-test;
- success accepts exactly one attempt;
- controlled training, val evaluation, and reserved-test evaluation freeze the
  expected `ProtocolEligibility`; smoke freezes
  `diagnostic-smoke-v1/false/false/false`. Copying a smoke artifact into the
  controlled store, editing only its directory, or omitting eligibility is
  rejected by accepted-run/aggregate/claim/profile/export APIs;
- a zero-exit training child is still rejected unless `training_manifest.json`
  proves the exact plan/materialization/realization, 50 completed epochs, expected
  global batch/update count, sampler and augmentation trace hashes, AMP,
  `nonfinite_count=0`, checkpoint-selection rule, and selected checkpoint bytes;
- `training_manifest.json` embeds `TrainingProtocolAssessment`, never
  `accepted_training_manifest_sha256`; after canonical serialization its detached
  SHA is wrapped in `TrainingProtocolVerification`. A fixed-vector test proves
  neither the manifest payload nor assessment is self-referential;
- 49 epochs, a missing update, mismatched initialization/teacher/environment hash,
  a selector pointing outside the attempt, or a post-selection checkpoint mutation
  prevents artifact finalization and the training success pointer;
- inject a crash after training manifest, artifact manifest, realization-ledger
  publication and success-pointer publication.
  `finalize_training_attempt()` reopens the same attempt, verifies the
  phase journal and byte-identical artifacts, idempotently replays an already
  committed realization-ledger record, and publishes the sole success pointer
  only as the final phase. A pointer without a fully committed matching ledger
  can never be accepted;
- `accepted_training_run()` returns only a detached-hash-verified manifest and
  checkpoint from the sole accepted attempt, and the plan-realization ledger maps
  one training plan to at most one accepted materialization;
- missing/empty/hash-mismatched artifact rejected;
- local training provenance resolves selected checkpoint hash;
- upstream provenance remains cross-protocol;
- one plan cannot have two accepted materializations;
- `TestClaim` requires exactly 36 multimodal or 24 LiDAR-only unique plan/attempt
  entries whose `(condition, latency_ms, seed)` key set exactly equals the frozen
  matrix plan, all three seed checkpoint hashes, dataset/split hashes, full code
  commit, temporal-manifest and matrix-plan hashes, every resolved
  config/protocol/overlay hash, the complete tagged fault semantics/scope fields,
  environment lock, hardware request, fixed
  prediction-source snapshot, adapter-execution environment/hardware, shared
  evaluator environment/hardware, fixed evaluator and exact IoU threshold tuple
  `(0.5, 0.7)`. Built-ins require the two environment/hardware pairs to be
  byte-equal; external-local/upstream predictions may use a distinct accepted
  adapter environment, and mutating either pair or the source snapshot changes
  the materialized evaluation ID and invalidates the claim;
- the core component freeze requires the complete accepted val `all` aggregate,
  qualitative result and Full-vs-concat profile; the baseline component freeze
  requires that core freeze plus the complete val `baselines` aggregate. Each
  hashes its exact checkpoints/code/protocol/deployment configs. The
  `reproduction` freeze is a composition record binding both component-freeze
  hashes plus the val reproduction aggregate; it is an export/orchestration gate,
  never a replacement claim freeze;
- test selectors `main|ablation|sensitivity|duration|arrival_relative|all` and
  core nodes reached through `reproduction` always bind `submission_component=
  "core"` and the same core freeze. Baseline nodes bind `"baselines"` whether
  reached through `baselines` or `reproduction`. `smoke` is forbidden. A
  filtered/resumed test reads its component/hash from the existing claim;
- order tests execute `all` then `reproduction` and the reverse. In both orders,
  core Full/concat families retain one identical claim and predictor invocation;
  reproduction adds only missing baseline families and its composition freeze.
  Thus the one-family rule and coexistence under one evidence root are compatible;
- submission freeze publication uses a fully written/fsynced sibling staging
  file, then atomic no-replace publication to
  `<evidence-root>/submission_freezes/<core|baselines|reproduction>.json`, followed by
  directory fsync. A published exact byte/hash match is idempotently reused;
  a conflict fails. Crash tests at write/fsync/publish/dir-fsync boundaries prove
  an uncommitted staging file can be safely removed and a published complete
  freeze is recoverable;
- a filtered test execution still acquires and verifies the complete family before
  its first selected node; a missing field/run/checkpoint/attempt blocks before any
  predictor or test artifact access;
- an existing test prediction/metric without the matching prior ledger record,
  a second family for the same method/variant/release, and any test `--force` are
  rejected;
- `accepted_test_claim_family()` returns only when every one of the claimed 36/24
  attempt IDs has an accepted pointer and complete verified artifact manifest;
  one failed/missing attempt rejects the entire family rather than returning the
  successful subset;
- per-sample rank-spool writes are O_EXCL, canonical and hash-verified; identical
  distributed duplicates promote once, conflicting duplicates fail, and completed
  canonical samples are never recomputed on resume;
- inject a crash after each D/Q/DER tensor, diagnostics index, prediction file,
  and sample commit boundary. Uncommitted files live only in an attempt/rank/sample
  staging directory; resume verifies then replaces that staging directory and
  recomputes the sample, while only an atomically renamed directory with a commit
  marker counts as complete. No half-sample is promoted;
- every completed sample requires both its prediction and complete diagnostics
  record/tensor files; missing, extra, wrong-order or hash-mismatched diagnostics
  make the sample incomplete and block promotion/evaluation;
- the Metric never writes a spool itself; `PredictAttempt` is the only owner of
  rank-spool/canonical-prediction state and only rank 0 may write evaluator/metrics
  artifacts after exact full-manifest coverage;
- an interrupted attempt with two old canonical predictions plus two new rank
  spools loads all four in manifest order into one evaluator call;
- inject crashes after evaluator output, metrics, artifact manifest and success
  pointer. `finalize_evaluation_attempt()` resumes the same reserved-test attempt
  without recomputing predictions, reuses only byte/hash/metadata-identical
  artifacts, and terminally rejects conflicts without overwrite;
- concurrent ledger append via two processes yields two valid serialized records or one explicit conflict, never corrupt JSONL.

- [ ] **Step 2: Run evidence tests and confirm failure**

Run: `python -m pytest tests/resilient_v2x/test_evidence.py -q`

Expected: FAIL because evidence APIs are absent.

- [ ] **Step 3: Implement safe filesystem operations**

Requirements:

- evidence root is absolute, exists or is created with mode 0750, and is outside repository root;
- attempt creation uses `mkdir(exist_ok=False)`;
- every success/freeze pointer is first written and fsynced as a canonical
  same-directory temporary file, then atomically published without replacement
  by hard-linking temp to the final name (or an equivalently tested
  no-replace primitive), fsyncing the directory, and unlinking the temp. The
  final path is never opened for incremental writes;
- each JSON file is canonical, flushed, `fsync`ed, and atomically renamed before reference;
- ledgers lock a stable sibling `<ledger>.lock` inode that is created once and
  never renamed/unlinked; they never lock the replaceable ledger inode itself.
  Only after acquiring that lock does a writer reopen and re-read/verify the
  current ledger path, write the complete old-plus-new JSONL bytes to a sibling
  temp, fsync, atomically replace the ledger and fsync its directory before
  unlocking. Retry observes either the complete old or complete new chain, never
  a torn tail, and treats an identical final record as idempotent. A barrier-based
  3+ process stress/fault test proves every distinct record survives with one
  contiguous sequence and no lost update across replacement;
- manifest hashes every required non-empty artifact, sorted by relative path;
- failed attempt writes failure metadata but never a success pointer.

Crash-injection tests cover pointer temp create/write/file-fsync/no-replace
publish/directory-fsync/cleanup. Retry discards only an unpublished temp owned by
the same attempt, reuses a byte-identical fully published pointer, and rejects a
different or malformed final pointer; no empty/partial final pointer can exist.

`ArtifactManifest.detached_manifest_sha256` is return-time verified metadata, not
a serialized field in `run_manifest.json`. It is exactly SHA-256 over the final
canonical manifest bytes and must equal the contents of
`run_manifest.sha256`; both manifest files remain excluded from `artifacts`.
This rule has a fixed-vector test and avoids a self-referential manifest hash.
Each ledger `record_sha256` is
`sha256(b"resilient-v2x-ledger-record-v1\\0" + canonical_json(record_without_record_sha256))`;
`previous_record_sha256` is null for sequence zero and exactly the prior record's
verified digest thereafter. First/second-record fixed vectors plus field mutation
tests freeze this non-self-referential chain.

`test_claim.schema.json` sets `additionalProperties: false` at every object level.
`acquire_test_claim()` first verifies the existing ledger and scans for genuinely
unbound test artifacts, validates exact family coverage, and deterministically
computes every `TestPlanAttempt.attempt_id` without creating directories. It then
atomically commits/fsyncs the one family ledger record before idempotently
creating claim-bound attempt directories. A crash after ledger commit resumes the
same family and creates only missing directories; a directory whose exact ID is
bound by that committed claim is not an orphan. The function returns only after
all 36/24 directories contain matching immutable claim metadata and their parent
directory is fsynced. Callers may not open a test dataloader, prediction,
checkpoint-backed model, or existing test metrics before this return. Crash tests
at each ID/allocation/ledger publish/fsync boundary prove retry neither rejects
its own committed transaction nor creates a second claim. On every
post-acquisition entry and resume, `preflight_test_access()` repeats the
claim-aware orphan scan and verifies the matching ledger record and full hash
chain.

`PredictAttempt.initialize()` writes immutable `prediction_plan.json` containing
the manifest-ordered sample IDs, temporal-manifest hash, exact argv digest, world
size, split/protocol, eligibility and attempt ID. Each rank atomically writes one hashed sample record under
`diagnostics/rank-spool/rank-<rank>/`; rank 0 alone promotes verified records to
`predictions/`. `remaining_sample_ids()` is computed only after rehashing canonical
predictions and rank spools. A recoverable interruption appends diagnostics but
does not finalize the attempt or write a success pointer. A terminal protocol/hash
error writes failure metadata and is never resumable as the same accepted run.

`inference_diagnostics_from_datasample()` requires the typed
`resilient_diagnostics` field, matching sample ID and `[L_E,L_R,C_E,C_R]` branch
order. `record_rank_sample()` writes one canonical diagnostics index plus
little-endian, C-contiguous `.npy` tensors under the rank spool: per-branch
displacement `D` and confidence `Q` when present, and DER descriptor,
expert-support and weights when applicable. The index records dtype, shape, byte
size and SHA-256 for each tensor plus source tick/time, horizon, support/reason,
observed/propagated, gamma and reliability. Unsupported/not-applicable tensor
fields are JSON `null`, never zero arrays. Files use create-exclusive atomic
writes, `allow_pickle=False`, flush/fsync and detached hashes. They are first
written under `<rank-spool>/.staging/<sample-id>-<attempt-id>/`; only after every
prediction/index/tensor hash verifies is a commit marker fsynced and the whole
sample directory atomically renamed into the committed rank spool. On resume an
uncommitted staging directory may be removed/replaced only after its path,
attempt, rank and lack of commit marker are verified; committed directories are
immutable. Partial tensor files therefore neither count as completion nor block
same-attempt recomputation.

`load_complete_predictions()` requires exact expected sample coverage and reads
only promoted, detached-hash-verified canonical predictions in immutable manifest
order. `write_evaluation_artifacts()` is rank-0-only, uses O_EXCL plus fsync, and
is the sole writer of `evaluator_output.json` and `metrics.json`. It accepts the
one pure `EvaluationBundle` returned by the Metric, verifies its top-level bundle,
evaluator/protocol/manifest/split/sample-set hashes against the prediction plan
and attempt before either file is created, then writes the raw output and a
metrics artifact that repeats that verified metadata alongside the flat values;
the Metric itself never participates in the spool/file lifecycle.
Final publication is a journaled state machine. Both training and evaluation
`finalization.jsonl` are instances of the same safe `HashChainedLedger`: every
phase record uses the permanent sibling lock, reopens/verifies the current chain,
writes the whole next chain to temp, fsyncs, atomically replaces, fsyncs the
directory, and only then releases the lock. Raw append is forbidden. Each phase
commits a record only after its artifact is durable. Crash tests cover journal
temp write/file-fsync/replace/directory-fsync plus every artifact/pointer phase;
retry sees the complete old or new chain, while a complete record with an invalid
hash is terminally rejected. On resume, `finalize_evaluation_attempt()` replays that journal and, for
an already present evaluator output or metrics file, recomputes the exact
canonical bytes expected from the same `EvaluationBundle`; byte/hash/metadata
equality advances without rewriting, while any difference is terminal. It then
verifies or creates the artifact manifest and sole success pointer under the same
rule. A crash at any final boundary is recoverable in the original test attempt
without weakening O_EXCL or overwriting conflicting evidence.
`evaluation_bundle.schema.json` forbids additional properties and fixes the
metrics artifact envelope to
`{schema_version,evaluator_version,protocol_sha256,temporal_manifest_sha256,
resolved_split,ordered_sample_ids_sha256,bundle_sha256,values}`. Aggregators read
only `values` after verifying all envelope hashes; they never treat metadata as a
numeric metric.
`inference_diagnostics.schema.json` fixes every scalar/array-index field and
allows `null` only under the explicit unsupported/not-applicable rules. Promotion
moves prediction and diagnostics as one sample unit; the run artifact manifest
hashes every diagnostics JSON/array file. Resuming verifies the whole unit before
placing the sample in `verified_completed_sample_ids()`.

`accepted_test_claim_family()` verifies the matching ledger record, preserves its
frozen attempt order, and checks every attempt through `AttemptStore` plus its
manifest. It is the only supported input for reserved-test aggregation, ranking,
qualitative selection, and export.

Training acceptance follows the same immutable order: validate the strict
`training_manifest.schema.json`, rederive `TrainingProtocolAssessment` from logs,
resolved config, sampler/update/augmentation traces and checkpoint selector,
then call the journaled `finalize_training_attempt()`. It durably finalizes the
artifact manifest, appends/idempotently verifies the detached manifest hash plus
assessment as `TrainingProtocolVerification` and the materialized realization in
`plan_realizations.jsonl`, and creates the sole success pointer last. Merely
returning zero from `tools/train.py` never makes a checkpoint discoverable.
`accepted_training_run()` re-verifies that ledger/pointer/manifest chain and is the
only source for `LocalModelProvenance`, teacher dependencies, student/concat/
baseline deployment, or checkpoint selection.

- [ ] **Step 4: Run evidence tests**

Run: `python -m pytest tests/resilient_v2x/test_evidence.py -q`

Expected: all integrity and concurrency tests pass.

- [ ] **Step 5: Commit**

```bash
git add transvision/experiments/resilient_v2x/evidence.py transvision/experiments/resilient_v2x/schemas tests/resilient_v2x/test_evidence.py
git commit -m "feat: add immutable experiment evidence store"
```

---

### Task 5: 实现 dirty-tree gate、可逐样本恢复的 matrix runner 和 smoke 诊断路径

**Files:**

- Create: `transvision/experiments/resilient_v2x/predict_attempt.py`
- Create: `transvision/experiments/resilient_v2x/inference.py`
- Modify: `transvision/experiments/resilient_v2x/__init__.py`
- Create: `tools/resilient_v2x/run_matrix.py`
- Create: `tools/resilient_v2x/predict_attempt.py`
- Create: `tools/resilient_v2x/validate_protocol.py`
- Create: `tests/resilient_v2x/test_run_matrix.py`
- Create: `tests/resilient_v2x/test_predict_attempt.py`

**Interfaces:**

```python
@dataclass(frozen=True)
class SmokeRequest:
    split: Literal["val"]
    seed: Literal[0]
    max_epochs: Literal[2]

@dataclass(frozen=True)
class ControlledWorktreeIdentity:
    commit: str
    tree: str
    source_snapshot_sha256: str

@dataclass(frozen=True)
class ExecutionContext:
    repo_root: Path
    data_root: Path
    evidence_root: Path
    temporal_manifest: Path
    environment_manifest: Path
    launcher: Literal["none", "pytorch"]
    nproc_per_node: Literal[1, 2, 4] | None
    sanitized_environment: Mapping[str, str]
    dry_run: bool
    resume: bool
    force: bool
    source_commit: str
    source_tree: str
    source_snapshot_sha256: str

@dataclass(frozen=True)
class TrainingLaunch:
    backend: str
    argv: tuple[str, ...]
    cwd: Path
    environment: Mapping[str, str]
    environment_manifest_sha256: str
    actual_hardware_fingerprint_sha256: str
    source_plan: TrainingSourcePlan
    source_snapshot_sha256: str
    initialization: InitializationArtifact
    training_evidence_parser: str

class TrainingBackend(Protocol):
    name: str

    def build_launch(
        self,
        node: PlanNode,
        plan: TrainingPlan,
        context: ExecutionContext,
    ) -> TrainingLaunch

@dataclass(frozen=True)
class ExecutionResult:
    plan_id: str | None
    materialized_id: str | None
    attempt_id: str | None
    state: Literal["successful", "failed", "skipped", "blocked"]
    command_sha256: str | None
    artifact_manifest: ArtifactManifest | None
    metrics: Mapping[str, float] | None
    blocked_reason: str | None

@DATA_SAMPLERS.register_module()
class ResumableSampleSampler(Sampler[int]):
    def __init__(
        self,
        dataset_sample_ids: Sequence[str],
        attempt_dir: str,
        expected_world_size: int,
    ) -> None
    def __iter__(self) -> Iterator[int]
    def __len__(self) -> int

@RUNNERS.register_module()
class IndependentInferenceRunner(Runner):
    def wrap_model(
        self,
        model_wrapper_cfg: Mapping[str, object] | None,
        model: nn.Module,
    ) -> nn.Module

@LOOPS.register_module()
class ResumableTestLoop(BaseLoop):
    def run(self) -> dict[str, float]

def require_resilient_metric(evaluator: Evaluator) -> ResilientV2XMetric
def validate_controlled_worktree(repo_root: Path) -> ControlledWorktreeIdentity
def build_diagnostic_smoke_plan(
    request: SmokeRequest,
    context: MaterializationContext,
) -> tuple[PlanNode, ...]
def build_runner_command(
    node: PlanNode,
    launcher: Literal["none", "pytorch"],
    nproc_per_node: int | None,
) -> tuple[str, ...]
def materialize_deployment_config(
    training_resolved_config: Config,
) -> Config
def materialize_evaluation_runtime_config(
    deployment_config: Config,
    plan: EvaluationPlan,
    attempt: PredictAttempt,
) -> Config
def strict_load_deployment_checkpoint(
    deployment_config: Config,
    selected_checkpoint: Path,
    expected_checkpoint_sha256: str,
) -> nn.Module
def materialize_training_realization(
    plan: TrainingPlan,
    context: ExecutionContext,
    model_initialization_sha256: str,
    accepted_teacher: AcceptedTrainingRun | None,
) -> TrainingRealization
def build_training_launch(
    node: PlanNode,
    plan: TrainingPlan,
    context: ExecutionContext,
) -> TrainingLaunch
def register_training_backend(name: str, backend: TrainingBackend) -> None
def execute_plan_node(node: PlanNode, context: ExecutionContext) -> ExecutionResult
def execute_training_attempt(
    node: PlanNode,
    plan: TrainingPlan,
    context: ExecutionContext,
    store: AttemptStore,
) -> ExecutionResult
def execute_predict_attempt(
    node: PlanNode,
    context: ExecutionContext,
    attempt: PredictAttempt,
) -> ExecutionResult
def resume_plan_node(node: PlanNode, context: ExecutionContext) -> ExecutionResult
def main(argv: Sequence[str] | None = None) -> int
def prediction_main(argv: Sequence[str] | None = None) -> int
```

- [ ] **Step 1: Write worktree, dry-run, smoke, resume, and reserved-test tests**

Use a temporary Git repo. Assert:

```python
identity = validate_controlled_worktree(clean_repo)
assert identity.commit == clean_repo_head
assert identity.tree == clean_repo_tree
with pytest.raises(ProtocolError, match="tracked worktree"):
    validate_controlled_worktree(repo_with_tracked_change)
with pytest.raises(ProtocolError, match="untracked controlled path"):
    validate_controlled_worktree(repo_with_untracked_config)
```

Add CLI tests that:

- no-data `--dry-run` emits all template slots and blocked reasons, no plan IDs;
- fixture-manifest CLI dry-run emits all template IDs plus
  `fixture_manifest_not_controlled` blocked reasons and no plan/materialized IDs;
  a separate pure `controlled_semantics_fixture_context()` (not reachable from
  CLI/evidence writers) uses the frozen official hash literals to test exact DAG
  IDs/counts without loading or accepting a fake production manifest;
- `ExecutionResult(state="blocked")` for no-data/fixture planning has all three
  IDs, command, manifest and metrics set to `None` plus a non-empty reason.
  Successful/failed materialized execution requires all three IDs and a command
  hash; skipped-success requires the accepted attempt ID. Invalid partial
  combinations fail dataclass construction;
- unfiltered `--matrix all --split val` materializes the complete 39-training /
  198-val controlled DAG; unfiltered `--matrix all --split test` materializes only
  the frozen 72-run Resilient-main-plus-concat reserved-test family. Any filtered
  invocation is a single-node inspection/resume operation, not a prerequisite
  workflow substitute;
- `--matrix smoke --split val --seed 0 --smoke-max-epochs 2` produces exactly
  two diagnostic training nodes (teacher then Full student) and three diagnostic
  val evaluation nodes (`Full@0`, `L-Fail@0`, `Full@300`);
- every smoke node uses the `diagnostic-smoke-v1` namespace, has
  `controlled=false` and `paper_eligible=false`, stores attempts only below
  `<evidence-root>/diagnostic_runs/`, and resolves both training configs to
  `max_epochs=2`;
- smoke rejects `test`, any seed other than `0`, a missing or non-`2`
  `--smoke-max-epochs`, and selectors that would change its fixed five-node DAG;
- every controlled matrix rejects `--smoke-max-epochs` rather than silently
  changing a paper training protocol;
- controlled aggregators, test-claim acquisition, accepted-run lookup and paper
  export all reject a diagnostic protocol/attempt even if its artifacts are valid;
- duplicate successful node is skipped;
- a training node computes its `TrainingRealization` from the verified
  initialization artifact, optional accepted teacher, and accepted environment,
  creates its materialized attempt before launching `tools/train.py`, then
  validates and promotes the exact training-manifest/artifact/success-pointer/
  realization-ledger chain. A child exit-code spy alone never marks success;
- teacher, student, concat, and locally trained baseline fixtures prove their
  accepted checkpoint is subsequently the only checkpoint available to
  evaluation; missing teacher acceptance, wrong teacher role/hash, wrong 50-epoch
  evidence, or a forged `selected_checkpoint.json` blocks before evaluation-plan
  materialization;
- core backend dispatch is exact: the only backend registered in this task is
  `builtin_mmengine`, which launches `tools/train.py`. Unknown backend names fail
  closed before attempt creation; the registration API is stable, but concrete
  baseline/external registration and parity tests are deferred to baseline Task
  8, after adapter implementations exist. Upstream nodes never create a training
  launch;
- failed non-test node resumes in a new attempt only with `--force`;
- a filtered test node first acquires the complete 36-run multimodal or 24-run
  LiDAR-only claim family only after `acquire_submission_freeze()` verifies the
  val aggregate/profile/qualitative acceptance; it fsyncs its ledger record, and only then may open a
  checkpoint, dataloader, prediction file, or invoke the mocked predictor;
- failed test node with `--resume` reuses the same attempt ID and exact argv digest,
  verifies every existing artifact hash, promotes valid rank spools, and requests
  only the manifest-ordered missing samples;
- test resume never creates a second attempt or a second claim-family ledger record;
- test `--force` exits non-zero before any artifact mutation;
- wrong split manifest or test config pointing to val fails;
- materializing a deployment config extracts the bare `model.student` detector,
  removes every teacher/distillation/checkpoint field, sets dataset
  `teacher_mode=False` and preprocessor `view_mode="student"` without embedding a
  split, checkpoint or attempt path; `strict_load_deployment_checkpoint()` then
  strict-loads the selected student-only state dict. A teacher-constructor spy is never called,
  every checkpoint key is consumed exactly once, and missing/extra or `teacher.*`
  keys fail before inference;
- val/test/profile/qualitative use the same checkpoint-free bare deployment config
  hash. Changing split/manifest/range changes `evaluation_plan_id` only; changing
  checkpoint changes the materialized evaluation ID only; changing attempt path
  changes only runtime config/command/prediction-plan hashes. Tests prove none of
  these mutations creates a plan/config/attempt identity cycle;
- concat and built-in baseline configs remain bare but pass the same strict
  checkpoint/config-hash gate; evaluation/profiling may never instantiate the
  training wrapper simply because it was present in the training config;
- a real `Runner.from_cfg` build produces one MMEngine `Evaluator` containing
  exactly one `ResilientV2XMetric`; zero, multiple, nested, or any additional
  Metric fails before inference, and the loop obtains the sole metric through
  `require_resilient_metric(runner.test_evaluator)` rather than calling a method
  on `Evaluator` itself;
- shell command is an argv tuple executed without `shell=True`;
- external adapter working directory and environment are explicit.
- unfiltered core `run_matrix` executes only `train` and `evaluate` nodes
  (reserved-test claim acquisition is part of `evaluate`). Baseline Task 8 later
  extends the kind registry with `adapter_preflight`;
  Aggregate/profile/qualitative/export nodes are readiness-only DAG nodes;
  execution spies prove the matrix runner never invokes their libraries or CLIs,
  and `execute_plan_node()` rejects those kinds with a named downstream-CLI error;
- `--launcher none` rejects `--nproc-per-node` and materializes world size 1;
- `--launcher pytorch --nproc-per-node {1,2,4}` prefixes the child command with
  the locked interpreter's `-m torch.distributed.run --standalone
  --nproc-per-node=<N>`, records world size `N`, and rejects missing/mismatched
  GPU availability before attempt creation.

Write a real interruption test, not a file-count surrogate:

```python
def test_prediction_resume_never_recomputes_completed_samples(tmp_path: Path) -> None:
    attempt = initialized_predict_attempt(
        tmp_path,
        sample_ids=("a", "b", "c", "d"),
        world_size=1,
    )
    first_predictor = predictor_that_records(("a", "b"), then_raises=KeyboardInterrupt)
    with pytest.raises(KeyboardInterrupt):
        execute_predict_attempt(fixture_node(), fixture_context(first_predictor), attempt)

    resumed_predictor = recording_predictor()
    execute_predict_attempt(
        fixture_node(),
        fixture_context(resumed_predictor, resume=True),
        attempt,
    )
    assert resumed_predictor.requested_sample_ids == ("c", "d")
    assert canonical_prediction_ids(attempt) == ("a", "b", "c", "d")
```

Add sampler tests for manifest-order preservation, non-contiguous remaining IDs,
world-size partitioning without padding/duplicates, identical-spool de-duplication,
exact `__len__`, runtime rank/world-size agreement, and failure before inference
when command digest, world size, sample manifest,
split, protocol, or attempt ID differs from `prediction_plan.json`.
Spy the custom loop to prove each `Det3DDataSample` is converted and durably
recorded exactly once before the next sample, BaseMetric's distributed collector
is never called, and rank 0 invokes `evaluate_complete()` exactly once only after
full old-plus-new coverage.
Add a real two-process CPU/Gloo test with three remaining samples: both ranks
build `IndependentInferenceRunner`, rank 0 owns two samples and rank 1 one, every
sample is forwarded exactly once, the job exits without a collective hang, rank 0
alone promotes/writes, and both ranks receive the same broadcast metrics. Assert
`runner.model` is the original module, not `MMDistributedDataParallel`, and reject
SyncBatchNorm or any configured model wrapper.

- [ ] **Step 2: Run runner and prediction-attempt tests and confirm failure**

Run:

```bash
python -m pytest tests/resilient_v2x/test_predict_attempt.py tests/resilient_v2x/test_run_matrix.py -q
```

Expected: FAIL because matrix CLI is absent.

- [ ] **Step 3: Implement filters and deterministic state transitions**

Required CLI fields:

```text
--matrix
--split
--method
--variant
--condition
--latency-ms
--seed
--smoke-max-epochs
--baseline-manifest METHOD=ABS_PATH  # repeatable, optional
--data-root
--temporal-manifest
--evidence-root
--environment-manifest
--launcher
--nproc-per-node
--dry-run
--resume
--force
```

`validate_controlled_worktree()` rejects tracked changes anywhere and untracked
files under `transvision/`, `configs/`, `tools/`, `scripts/`, `tests/`, or
`environments/`. It records the full 40-character HEAD and `HEAD^{tree}`.
For built-ins, `source_snapshot_sha256` is exactly
`sha256(b"builtin-source-snapshot-v1\0" +
canonical_json({commit, tree}))`; strict clean-tree validation makes this a
reconstructible immutable source snapshot without copying the repository. A
fixed vector plus commit/tree/dirty-state mutation tests bind the value. External
backends instead provide the hash of their atomically materialized file inventory.
`--data-root` is always required and absolute, including dry-run; dry-run may name
an absent root and then emits a blocked dependency, but it cannot omit the
resolution base. Controlled execution path-contains it, resolves every manifest
relative path beneath it, and includes its canonical path plus verified raw/
prepared inventory hashes in the command digest.
`--baseline-manifest` is repeatable only for `v2x_vit|cobevt|lrcp`. The parser
requires an absolute regular file, strict schema/method agreement, unique method,
and a pre-materialization SHA-256; it never scans a directory or selects a newest
file. The resulting exact path/hash mapping enters `MaterializationContext`, each
adapter command manifest, and model provenance. Omission keeps that external
adapter's checked-in template as a hashed blocked preflight rather than failing
unrelated methods.

The runner calls `tools/train.py` for training and the custom
`predict_attempt.py` entrypoint for every resumable controlled/diagnostic
evaluation through `subprocess.run(argv, check=True, shell=False)`;
`tools/test.py` is permitted only for an explicitly non-resumable local
diagnostic outside the matrix. Before launch it writes an
immutable command manifest containing the exact argv and digest, explicit cwd,
sanitized environment, code/data/config/protocol hashes, launcher, rank and fixed
world size. It never calls legacy detection-model wrappers.

`execute_training_attempt()` is the sole training-node evidence path. It resolves an
accepted teacher through `accepted_training_run()` when the role requires one,
calls `build_training_launch()`, verifies the model-initialization bytes, builds
`TrainingRealization` from the launch's actual training environment, computes
the materialized ID, and creates a training attempt. It writes the immutable
resolved config, overlays, protocol declaration, environment validation, source
snapshot identity and exact argv under that attempt before launch. This task
registers only the built-in backend, which runs `tools/train.py` with an
attempt-local work directory. `register_training_backend()` rejects duplicate or
reserved names and permits a later plan phase to supply a backend only through
the same typed launch/evidence contract; this task imports no baseline adapter
module. Upstream checkpoint nodes are forbidden here. After a successful child exit,
rank 0 verifies the complete logs/sampler/update/augmentation traces and
`selected_checkpoint.json`, independently replays the checkpoint selector,
constructs `TrainingAttemptManifest` with the assessment, and calls only
`finalize_training_attempt()`. That journaled finalizer writes/verifies the
training and artifact manifests, constructs the outer verification from the
detached manifest hash, atomically/idempotently commits the plan-realization
ledger, and publishes the sole success pointer last. It replays exact completed
phases after a crash and rejects conflicts, so a success pointer can never precede
the discovery ledger required by `accepted_training_run()`.
Any failed invariant
leaves a failed, non-accepted attempt. A training retry requires `--force` and a
new attempt; it cannot overwrite or silently resume the prior evidence.

Before any evaluation Runner is built,
`materialize_deployment_config()` derives an immutable, checkpoint-free and
attempt-independent bare deployment config. For
`ResilientV2XTeacherStudent` it promotes the nested student config to the top-level
bare detector, removes teacher/distillation/initialization/checkpoint fields and
all split/dataloader/evaluator/attempt paths, and fixes the preprocessor to the
student-only view. Bare concat/baseline models are retained under the same
contract. The derived model/preprocessor config's canonical SHA-256 is the
`EvaluationPlan.resolved_config_sha256`; it is never the training-wrapper config
hash. `strict_load_deployment_checkpoint()` separately verifies and loads the
realized checkpoint. The same accepted deployment config/hash is used by val,
reserved test, qualitative rendering and profiling.

For every evaluation attempt, `materialize_evaluation_runtime_config()` overlays
the already-hashed deployment config with the `EvaluationPlan`'s split, manifest,
range and evaluator semantics, then installs
`IndependentInferenceRunner`, `ResumableTestLoop`, and
`ResumableSampleSampler`, imports their registry module, fixes
`test_evaluator.type=ResilientV2XMetric`, and injects only the current absolute
attempt state into the loop. The semantic overlay fields already participate in
`evaluation_plan_id`; the attempt-only overlay hash participates only in the
command manifest and `prediction_plan.json`, never
`EvaluationPlan.resolved_config_sha256`. Evaluator, dataset, sampler, plan and
prediction attempt must all carry that same split. It rejects
configurable evaluator version, thresholds, class or prediction directories. At
Runner construction, `require_resilient_metric()` verifies
`isinstance(evaluator, Evaluator)`, `len(evaluator.metrics)==1`, and the exact
metric class, then returns `evaluator.metrics[0]`; BaseMetric collection/evaluate
is not invoked by the custom loop. At
process startup, rank 0 first verifies/promotes valid rank spools, computes the
canonical remaining-ID tuple, broadcasts it, and all ranks enter a barrier before
constructing their sampler. The sampler calls `get_dist_info()` at runtime,
compares world size to `expected_world_size`, maps the broadcast remaining
manifest sample IDs back to dataset indices and partitions them as
`remaining_indices[rank::world_size]`; there is no DDP padding. The same argv/config
is used on resume, while the sampler derives current remaining IDs from the
verified attempt state. A sample becomes complete only after its detached hash and
canonical prediction are durable.

`IndependentInferenceRunner.wrap_model()` is used only by this prediction
entrypoint. It rejects `model_wrapper_cfg` and SyncBatchNorm, moves the model to
the rank-local device, and intentionally returns the original module even though
the spool-coordination process group is initialized. Evaluation workers load
identical checkpoints into these independent modules and perform no model-forward
collective, so uneven sampler lengths cannot deadlock. After each rank finishes
its local loop, `ResumableTestLoop` converts every returned `Det3DDataSample`
with both `sample_prediction_from_datasample()` and
`inference_diagnostics_from_datasample()`, then immediately passes both canonical
values plus manifest index/sample ID to `PredictAttempt.record_rank_sample()`; it
does not use BaseMetric result collection or a distributed gather. All ranks then
enter one barrier; rank 0 promotes verified spools,
loads the complete old-plus-new canonical `SamplePrediction` objects, calls
`require_resilient_metric(runner.test_evaluator).evaluate_complete(predictions)`
exactly once, passes that complete bundle
to `write_evaluation_artifacts()`, broadcasts the
flat metrics, and all ranks enter a final barrier.
`__len__()` returns the exact local shard length.
Val and reserved test use the same frozen temporal-manifest file/hash; the
explicit plan/CLI split selects immutable rows. No per-split temporal-manifest
artifact is inferred or required.

This three-layer contract is acyclic: the bare deployment hash is computed before
an evaluation plan; the plan canonically owns semantic split/manifest/range
fields; and only after plan/materialized IDs exist may the runtime layer add
attempt paths and world-size state. Mutation tests recompute each layer and
reject any attempt-dependent value in a plan hash.

For reserved test, claim-family acquisition and ledger fsync happen before
checkpoint/model/dataloader construction or any existing test artifact read.
The test entrypoint first resolves or create-exclusively acquires the immutable
component freeze for that method family and requires its component/hash in the
claim. A reproduction invocation additionally creates/verifies the composition
freeze but reuses existing core claims rather than acquiring a second family.
`--resume` is the only recovery path: reopen the original attempt, verify the
claim ledger, command/prediction plan and partial hashes, and execute only missing
sample IDs with the same deterministic command. Any mismatch terminally aborts the
claim family rather than creating a new attempt.

The claim's requested-hardware hash states the frozen request; its actual-hardware
fingerprint and environment-manifest hashes bind the realization used to compute
every `materialized_evaluation_id`. Preflight recomputes those IDs from the claim,
plan and accepted environment artifact rather than trusting IDs supplied in the
ledger.

`--matrix smoke` is a separate diagnostic planner, not a filter over the
controlled matrix. It requires exactly `--split val --seed 0
--smoke-max-epochs 2`, creates new `diagnostic-smoke-v1` plan and attempt IDs under
an `AttemptStore` rooted at `<evidence-root>/diagnostic_runs/` (never the
controlled store), trains a two-epoch teacher and two-epoch Full student, then
evaluates only `Full@0`, `L-Fail@0`, and `Full@300`. Its manifests explicitly set
`controlled=false`, `paper_eligible=false`, and `reserved_test_eligible=false`;
no controlled plan/materialized ID, test claim, accepted-run pointer, aggregate,
profile, qualitative output, or paper result slot may reference it.

For `launcher=pytorch`, `build_runner_command()` emits an argv tuple beginning
`(sys.executable, "-m", "torch.distributed.run", "--standalone",
"--nproc-per-node", str(nproc_per_node), ...)`; it never assumes that passing
MMEngine `--launcher pytorch` creates ranks. Training children receive
`tools/train.py ... --launcher pytorch`. Prediction children receive the custom
`predict_attempt.py` entrypoint; that entrypoint initializes only the spool
coordination process group and keeps each inference model unwrapped as specified
above. The environment manifest, realized training/evaluation ID, sampler,
accumulation mapping and command manifest must all agree on world size before
launch.

The core matrix CLI stops after executable train/evaluate nodes. It
records aggregate/profile/qualitative/export nodes only as downstream readiness
with dependencies and never invokes their libraries or CLIs.
`execute_plan_node()` rejects those readiness-only kinds with an error naming the
dedicated CLI. Those CLIs are the sole O_EXCL publishers, so an unfiltered matrix
run cannot race or duplicate the documented downstream commands.

- [ ] **Step 4: Run runner and matrix tests**

Run:

```bash
python -m pytest tests/resilient_v2x/test_matrix.py tests/resilient_v2x/test_predict_attempt.py tests/resilient_v2x/test_run_matrix.py -q
```

Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add transvision/experiments/resilient_v2x/predict_attempt.py transvision/experiments/resilient_v2x/inference.py transvision/experiments/resilient_v2x/__init__.py tools/resilient_v2x/run_matrix.py tools/resilient_v2x/predict_attempt.py tools/resilient_v2x/validate_protocol.py tests/resilient_v2x/test_predict_attempt.py tests/resilient_v2x/test_run_matrix.py
git commit -m "feat: orchestrate resilient v2x experiment matrix"
```

---

### Task 6: 实现 paired three-seed aggregation、PDR 和表格选择

**Files:**

- Create: `transvision/experiments/resilient_v2x/aggregate.py`
- Create: `transvision/experiments/resilient_v2x/schemas/aggregate_collection.schema.json`
- Create: `tools/resilient_v2x/aggregate_results.py`
- Create: `tests/resilient_v2x/test_aggregate.py`

**Interfaces:**

```python
@dataclass(frozen=True)
class SeedMetric:
    method: str
    variant: str
    result_id_slot: str
    condition: Literal["Full", "L-Fail", "C-Fail"]
    latency_ms: int
    seed: Literal[0, 1, 2]
    split: Literal["val", "test"]
    metric_key: Literal[
        "BEV_AP_0.5",
        "BEV_AP_0.7",
        "3D_AP_0.5",
        "3D_AP_0.7",
    ]
    value: float
    evaluation_plan_id: str
    materialized_evaluation_id: str
    accepted_attempt_id: str
    artifact_manifest_sha256: str
    evaluator_version: Literal["resilient-v2x-ap-v1"]
    protocol_sha256: str
    temporal_manifest_sha256: str
    ordered_sample_ids_sha256: str
    prediction_source_snapshot_sha256: str
    adapter_execution_environment_manifest_sha256: str
    adapter_execution_actual_hardware_fingerprint_sha256: str
    shared_evaluator_environment_manifest_sha256: str
    shared_evaluator_actual_hardware_fingerprint_sha256: str
    compatibility_classification: Literal["controlled", "cross_protocol"]

@dataclass(frozen=True)
class SummaryStatistic:
    values: tuple[float, ...]
    mean: float
    std: float

@dataclass(frozen=True)
class MethodSummary:
    summary_id: str
    method: str
    variant: str
    result_id_slot: str
    metric_key: str
    statistic: SummaryStatistic
    compatibility_classification: Literal["controlled", "cross_protocol"]

@dataclass(frozen=True)
class AggregationProtocol:
    schema_version: Literal[1]
    seed_set: tuple[Literal[0], Literal[1], Literal[2]]
    standard_deviation_ddof: Literal[1]
    pdr_formula: Literal["100*(normal-degraded)/normal-per-seed"]
    method_difference_formula: Literal["candidate-minus-reference-per-seed"]
    tie_tolerance: float
    metric_directions: tuple[tuple[str, Literal["higher", "lower"]], ...]
    aggregator_code_sha256: str
    source_code_commit: str
    protocol_sha256: str

@dataclass(frozen=True)
class AggregatePlan:
    schema_version: Literal[1]
    matrix: Literal["all", "baselines", "reproduction"]
    split: Literal["val", "test"]
    expected_result_slots: tuple[str, ...]
    accepted_coverage_sha256: str
    protocol: AggregationProtocol

@dataclass(frozen=True)
class AggregateCollectionIndex:
    schema_version: Literal[1]
    matrix: Literal["all", "baselines", "reproduction"]
    split: Literal["val", "test"]
    index_id: str
    expected_result_slots: tuple[str, ...]
    controlled_tables: tuple[ArtifactRecord, ...]
    cross_protocol_reference_table: ArtifactRecord | None
    blocked_job_manifests: tuple[ArtifactRecord, ...]
    coverage_sha256: str
    aggregation_protocol_sha256: str
    aggregator_code_sha256: str
    content_sha256: str

def aggregate_metric(records: Sequence[SeedMetric]) -> SummaryStatistic
def paired_pdr(normal: Sequence[SeedMetric], degraded: Sequence[SeedMetric]) -> SummaryStatistic
def paired_method_difference(
    candidate: Sequence[SeedMetric],
    reference: Sequence[SeedMetric],
) -> SummaryStatistic
def select_best(
    summaries: Sequence[MethodSummary],
    direction: Literal["higher", "lower"],
    tie_tolerance: float,
) -> tuple[str, ...]
def aggregate_plan_id(plan: AggregatePlan) -> str
def write_aggregate_collection_index(
    evidence_root: Path,
    index: AggregateCollectionIndex,
) -> Path
def accepted_aggregate_collection_index(
    evidence_root: Path,
    matrix: Literal["all", "baselines", "reproduction"],
    split: Literal["val", "test"],
    index_id: str,
) -> AggregateCollectionIndex
```

- [ ] **Step 1: Write exact arithmetic and rejection tests**

```python
def test_sample_standard_deviation_uses_ddof_one() -> None:
    summary = aggregate_metric([
        metric(seed=0, value=50.0),
        metric(seed=1, value=52.0),
        metric(seed=2, value=54.0),
    ])
    assert summary.mean == pytest.approx(52.0)
    assert summary.std == pytest.approx(2.0)


def test_pdr_is_paired_before_aggregation() -> None:
    normal = [metric(0, 80), metric(1, 60), metric(2, 40)]
    degraded = [metric(0, 72), metric(1, 48), metric(2, 20)]
    summary = paired_pdr(normal, degraded)
    assert summary.values == pytest.approx((10.0, 20.0, 50.0))
    assert summary.mean == pytest.approx(80.0 / 3.0)
```

Add tests for missing/duplicate seeds, protocol/environment/hardware/evaluator
mismatch, zero normal, candidate-reference sign, unrounded best selection,
`1e-12` ties, and AP/PDR/latency/memory directions. Aggregation requires the
prediction-source snapshot plus all four adapter-execution/shared-evaluator
environment/hardware fields to match across three seeds and in paired
candidate/reference comparisons; built-ins require both pairs equal, while an
external adapter preserves its two distinct pairs. Mutating any one field blocks
the summary.
`AggregationProtocol.protocol_sha256` uses the `aggregation-protocol-v1` domain
over the exact ddof, per-seed PDR/difference formulae, tie rule, metric
directions, seed set and aggregator code/commit, excluding only itself.
Construction requires `tie_tolerance == 1e-12` exactly.
`aggregate_plan_id()` is `canonical_id("aggregate-plan-v1", plan)` and therefore
binds selector, split, expected slots, exact accepted-attempt coverage and the
protocol. Fixed vectors plus accepted-input/code/formula mutations all change the
ID before an output directory is allocated.
`select_best()` returns sorted stable `MethodSummary.summary_id` values, never a
bare method label. `summary_id` binds method, variant, result slot and metric key;
tests rank/tie two variants of the same method without collision.
For `--matrix all --split val`, freeze aggregate table coverage as 12 main rows,
24 ablation rows (full plus seven non-full variants), 12 sensitivity rows, four
duration rows, eight arrival-relative rows, and 12 concat rows. Each row consumes
exactly seeds `(0,1,2)` even when multiple result slots reuse one accepted
evaluation. Reserved test contains only the frozen main and concat families; the
other val-only collections must be explicitly marked `not_in_reserved_test_scope`
rather than silently omitted.
Each `SeedMetric` must be constructed only from a schema-valid `metrics.json`
envelope after its bundle/protocol/manifest/split/sample-set hashes and accepted
attempt manifest are verified; numeric values come only from the envelope's
`values` object.
For `split=test`, add a fixture with 35 successful attempts and one failed attempt
inside a 36-run claim and assert aggregation fails before reading any metric; only
a fully accepted claim family may aggregate.
Inject crashes after each controlled/reference/blocked table, after index write,
and after the success marker. All work remains in a unique staging directory;
retry uses a new staging directory and atomically publishes one complete
collection. A pre-existing accepted directory is reused only after exact
index/artifact hash verification, while a conflict is never overwritten.

- [ ] **Step 2: Run aggregation tests and confirm failure**

Run: `python -m pytest tests/resilient_v2x/test_aggregate.py -q`

Expected: FAIL because aggregation APIs are absent.

- [ ] **Step 3: Implement evidence-first reads**

The CLI accepts only an evidence root, split, and collection selector from
`main|ablation|sensitivity|duration|arrival_relative|concat|all|baselines|reproduction`.
Expansion is split-aware: on val, `all` expands every core result collection and
`reproduction` adds baseline collections; on reserved test, `all` expands only
the frozen main-plus-concat families and `reproduction` adds only test-eligible
baseline families. Val-only core collections are recorded as
`not_in_reserved_test_scope` and are never requested from the evidence store.
It resolves accepted attempts, verifies all artifact
manifests, loads `metrics.json`, and refuses arbitrary loose JSON paths. Each
expanded `ExperimentTemplate.matrix_name` produces its own schema-valid table and
coverage manifest; no ablation/sensitivity/duration/arrival-relative row may be
silently reduced to `main`. For reserved test it first calls
`accepted_test_claim_family()` and never selects a successful subset from an
incomplete family. Use `statistics.stdev` or `numpy.std(..., ddof=1)` only after
exact seed pairing.

Tests freeze the exact expected result-slot set for each `(collection, split)`, the
three-seed coverage of every aggregatable row, and a `reproduction` index containing
controlled core/baseline tables, a separate cross-protocol reference table, and
blocked-job manifests. They also prove test aggregation never looks up an
ablation/sensitivity/duration/arrival-relative attempt. Cross-protocol rows are
never paired against controlled methods and blocked rows never receive numeric
placeholders.
`AggregateCollectionIndex.index_id` is exactly the `aggregate_plan_id` above.
`coverage_sha256` must equal the plan's hash of sorted slot-to-accepted-attempt
coverage; the index repeats the verified aggregation protocol and aggregator code
hashes. `content_sha256` hashes the entire canonical index excluding only itself.
The schema fixes every referenced table/reference/blocked path, size and SHA
through `ArtifactRecord`. Fixed vectors and field mutations freeze all identities.

The CLI writes all tables, the canonical index and `successful_index.json` inside
a unique sibling staging directory, fsyncs the full tree, then atomically renames
that directory to
`<evidence-root>/aggregates/<matrix>/<split>/<index-id>/`. No partial collection
occupies the accepted path. If that path already exists, only complete exact
index/artifact equality is idempotent success; conflicts fail without overwrite.
Readers never scan for a latest file:
`accepted_aggregate_collection_index()` requires the caller-supplied plan-derived
`index_id`, resolves that exact selector directory, and re-verifies plan identity,
protocol/code, index content hash, coverage and every referenced artifact.

- [ ] **Step 4: Run aggregation tests**

Run: `python -m pytest tests/resilient_v2x/test_aggregate.py -q`

Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add transvision/experiments/resilient_v2x/aggregate.py transvision/experiments/resilient_v2x/schemas/aggregate_collection.schema.json tools/resilient_v2x/aggregate_results.py tests/resilient_v2x/test_aggregate.py
git commit -m "feat: aggregate resilient v2x evidence"
```

---

### Task 7: 实现固定 corpus 的 FLOPs、memory 和 latency profiling

**Files:**

- Create: `transvision/experiments/resilient_v2x/profiling.py`
- Create: `transvision/experiments/resilient_v2x/schemas/profile_corpus.schema.json`
- Create: `transvision/experiments/resilient_v2x/schemas/profiling_report.schema.json`
- Create: `tools/resilient_v2x/profile.py`
- Create: `tests/resilient_v2x/fixtures/profile_corpus_golden.json`
- Create: `tests/resilient_v2x/test_profiling.py`

**Interfaces:**

```python
PROFILE_PRECISIONS = ("fp32", "amp")

@dataclass(frozen=True)
class ProfileInputAudit:
    sample_id: str
    sensor_role: str
    modality: Literal["point_cloud", "image", "calibration", "metadata"]
    temporal_offset: int
    source_relative_path: str
    source_sha256: str
    relative_path: str
    sha256: str
    byte_size: int
    decoded_shape: tuple[int, ...]
    decoded_dtype: str
    element_count: int

@dataclass(frozen=True)
class ProfileCorpus:
    schema_version: Literal[1]
    selection_domain: Literal["profile-v1"]
    resolved_split: Literal["val"]
    temporal_manifest_sha256: str
    sample_ids: tuple[str, ...]
    inputs: tuple[ProfileInputAudit, ...]
    corpus_sha256: str

@dataclass(frozen=True)
class AcceptedRun:
    result_id_slot: str
    evaluation_plan_id: str
    materialized_evaluation_id: str
    attempt_id: str
    accepted_manifest_path: Path
    accepted_manifest_sha256: str
    checkpoint_path: Path
    checkpoint_sha256: str
    deployment_config_path: Path
    deployment_config_sha256: str
    protocol_sha256: str
    temporal_manifest_sha256: str
    resolved_split: Literal["val", "test"]
    condition: Literal["Full", "L-Fail", "C-Fail"]
    latency_ms: int
    prediction_source_snapshot_sha256: str
    adapter_execution_environment_manifest_sha256: str
    adapter_execution_actual_hardware_fingerprint_sha256: str
    shared_evaluator_environment_manifest_sha256: str
    shared_evaluator_actual_hardware_fingerprint_sha256: str

@dataclass(frozen=True)
class DecodedSample:
    sample_id: str
    model_inputs: Mapping[str, object]
    preprocessing_sha256: str

@dataclass(frozen=True)
class ProfilingProtocol:
    schema_version: Literal[1]
    warmup_iterations: Literal[50]
    timed_iterations: Literal[200]
    corpus_size: Literal[32]
    precisions: tuple[Literal["fp32"], Literal["amp"]]
    batch_size: Literal[1]
    flops_backend: Literal["fvcore-flop-count-analysis-v1"]
    multiply_add_convention: Literal["one-mac"]
    latency_start: Literal["decoded-cpu-input-before-preprocess"]
    latency_end: Literal["postprocess-complete-after-cuda-sync"]
    disk_and_image_decode_timed: Literal[False]
    cuda_sync_boundary: Literal["before-and-after-each-timed-iteration"]
    memory_protocol: Literal[
        "reset-peak-before-run-read-absolute-max-memory-allocated"
    ]
    pin_memory: Literal[True]
    non_blocking_transfer: Literal[False]
    torch_num_threads: Literal[1]
    torch_num_interop_threads: Literal[1]
    tf32_enabled: Literal[False]
    cudnn_benchmark: Literal[False]
    cpu_affinity: tuple[int, ...]
    gpu_index: int
    profiler_code_sha256: str
    source_code_commit: str
    protocol_sha256: str

@dataclass(frozen=True)
class ProfilingEnvironment:
    manifest_path: Path
    environment_manifest_sha256: str
    actual_hardware_fingerprint_sha256: str
    world_size: Literal[1]

@dataclass(frozen=True)
class ProfilingResult:
    method: str
    precision: Literal["fp32", "amp"]
    corpus_sha256: str
    protocol_sha256: str
    checkpoint_sha256: str
    preprocessing_sha256: str
    actual_hardware_fingerprint_sha256: str
    per_sample_flops: tuple[tuple[str, float], ...]
    unsupported_operations: Mapping[str, int]
    parameter_count: int
    latency_ms: tuple[float, ...]
    latency_statistics: Mapping[str, float]
    pre_model_allocated_bytes: int
    absolute_peak_allocated_bytes: int
    incremental_peak_allocated_bytes: int
    artifact_sha256: str

@dataclass(frozen=True)
class PairedProfilingResult:
    precision: Literal["fp32", "amp"]
    full: ProfilingResult
    concat: ProfilingResult
    comparison_sha256: str

@dataclass(frozen=True)
class DualPrecisionProfilingResult:
    headline_precision: Literal["fp32"]
    fp32: PairedProfilingResult
    amp: PairedProfilingResult
    report_sha256: str

@dataclass(frozen=True)
class ProfilingSuccess:
    profile_plan_id: str
    attempt_id: str
    full_materialized_evaluation_id: str
    concat_materialized_evaluation_id: str
    full_accepted_attempt_id: str
    concat_accepted_attempt_id: str
    corpus_sha256: str
    protocol_sha256: str
    profiler_code_sha256: str
    source_code_commit: str
    environment_manifest_sha256: str
    actual_hardware_fingerprint_sha256: str
    report_relative_path: str
    report_sha256: str
    artifact_manifest_sha256: str

def profiling_protocol_sha256(protocol: ProfilingProtocol) -> str
def profile_plan_id(
    full_run: AcceptedRun,
    concat_run: AcceptedRun,
    corpus: ProfileCorpus,
    protocol: ProfilingProtocol,
    environment: ProfilingEnvironment,
) -> str
def accept_profiling_result(
    store: AttemptStore,
    success: ProfilingSuccess,
) -> AttemptHandle
def accepted_profiling_result(
    evidence_root: Path,
    expected_profile_plan_id: str,
) -> tuple[ProfilingSuccess, DualPrecisionProfilingResult]

def select_profile_corpus(
    eligible_samples: Sequence[tuple[int, str, Literal["train", "val", "test"]]],
    manifest_sha256: str,
    count: int,
    domain: str,
    resolved_split: Literal["val"],
) -> tuple[str, ...]

def build_profile_corpus(
    temporal_manifest: Path,
    dataset_root: Path,
    sample_ids: Sequence[str],
    resolved_split: Literal["val"],
) -> ProfileCorpus
def validate_profile_corpus(corpus_path: Path, dataset_root: Path) -> ProfileCorpus
def latency_summary(milliseconds: Sequence[float]) -> dict[str, float]
def flops_summary(per_sample_flops: Sequence[float]) -> dict[str, float]
def profile_model(
    model: nn.Module,
    corpus: ProfileCorpus,
    decoded_samples: Sequence[DecodedSample],
    precision: Literal["fp32", "amp"],
    warmup: int,
    repetitions: int,
) -> ProfilingResult
def profile_pair(
    full_run: AcceptedRun,
    concat_run: AcceptedRun,
    corpus: ProfileCorpus,
    protocol: ProfilingProtocol,
    environment: ProfilingEnvironment,
) -> DualPrecisionProfilingResult
```

- [ ] **Step 1: Write corpus-audit, paired-input, quantile, and boundary tests**

```python
def test_corpus_selection_is_stable_and_method_independent() -> None:
    indexed = [(index, f"{index:06d}", "val") for index in range(100)]
    first = select_profile_corpus(indexed, "a" * 64, 32, "profile-v1", "val")
    second = select_profile_corpus(
        list(reversed(indexed)), "a" * 64, 32, "profile-v1", "val"
    )
    assert first == second
    assert len(first) == 32
    assert first == tuple(load_profile_golden()["selected_sample_ids"])


def test_latency_uses_float64_linear_quantile() -> None:
    summary = latency_summary([1.0, 2.0, 3.0, 100.0])
    assert summary["median_ms"] == pytest.approx(2.5)
    assert summary["mean_ms"] == pytest.approx(26.5)
    assert summary["p95_ms"] == pytest.approx(85.45)


def test_flops_are_summarized_per_sample() -> None:
    summary = flops_summary([10.0, 20.0, 30.0, 40.0])
    assert summary == pytest.approx({
        "min": 10.0,
        "mean": 25.0,
        "p95": 38.5,
        "max": 40.0,
    })


def test_full_and_concat_must_use_the_same_serialized_corpus() -> None:
    corpus = verified_profile_corpus()
    result = profile_pair(
        full_seed0_run(),
        concat_seed0_run(),
        corpus,
        protocol(),
        profiling_environment_ws1(),
    )
    assert result.headline_precision == "fp32"
    assert result.fp32.full.corpus_sha256 == corpus.corpus_sha256
    assert result.fp32.concat.corpus_sha256 == corpus.corpus_sha256
    assert result.amp.full.corpus_sha256 == corpus.corpus_sha256
    assert result.amp.concat.corpus_sha256 == corpus.corpus_sha256
```

The golden fixture contains the exact 32 selected sample IDs and expected corpus
digest. Add tests that every input referenced by those temporal-manifest rows binds
both its raw provenance name/hash and the actual verified load artifact. LiDAR must
profile the prepared little-endian BIN declared by the manifest rather than its raw
PCD; images/calibration/metadata may have identical source/load paths. Every load
artifact has a path-contained relative name, SHA-256, byte count, decoded
shape/dtype and element/point count; changing either source or load byte, shape,
count, path, manifest hash or sample order fails before either model is loaded.

`profile_corpus.schema.json` and `profiling_report.schema.json` set
`additionalProperties: false` at every object level. The corpus schema requires
`resolved_split="val"`, exactly 32 unique val sample IDs in manifest order and
requires each input audit to refer to one selected sample. Missing and extra
manifest inputs, or any train/test row, are protocol errors. The report schema
requires exactly `headline_precision=fp32`, one `fp32`
result, one `amp` result, and no alternate precision key.
The accepted success record additionally binds both evaluation materializations
and accepted attempt IDs, corpus/protocol/environment/hardware hashes, report
hash, and complete profiling artifact-manifest hash. A missing precision child,
failed child, stale evaluation pointer, report mutation, or second accepted
pointer prevents `ProfilingSuccess`.
An injected crash immediately after `profile_corpus.json` publication can retry
the same profile plan: the existing corpus is accepted only after strict schema,
detached/source/load hash audit and deterministic re-derivation produce identical
canonical bytes. Conflicting bytes fail and are never overwritten.
Hashing is explicitly layered and domain-separated: a `ProfilingResult` hash uses
`profiling-child-v1` over its canonical payload excluding only
`artifact_sha256`; a `PairedProfilingResult` hash uses `profiling-pair-v1` over
precision plus the complete verified child payloads excluding only
`comparison_sha256`; the dual report uses `profiling-dual-v1` over headline
precision and both verified pairs excluding only `report_sha256`. Fixed vectors
and mutations at every child field prove propagation child → pair → dual without
self-reference.
`ProfilingProtocol.protocol_sha256` similarly uses
`profiling-protocol-v1` over its canonical payload excluding only that field and
is recomputed on load. A fixed vector plus every-field mutation test freezes
warmup/timed counts, corpus size, precision order, affinity and GPU identity.

Mock `fvcore` and CUDA to assert:

- Full student seed 0 and capacity-matched concat seed 0 are accepted controlled
  `Full@0 ms` val runs, use the same checkpoint-selection rule and the exact same
  serialized `ProfileCorpus`; teacher, smoke and cross-protocol artifacts fail;
- both profiles reuse the accepted evaluation-only bare deployment configs and
  strict student/concat checkpoints; a teacher or distillation wrapper constructor
  spy is never touched;
- FLOPs are measured separately for all 32 samples, preserve each sample ID and
  raw value, then report float64 `min/mean/p95/max`; one fused multiply-add follows
  fvcore's one-operation convention and is not multiplied by two;
- parameter count is exactly
  `sum(p.numel() for p in deployment_model.parameters() if p.requires_grad)` on
  the bare deployment model. Frozen parameters, buffers and any teacher-wrapper
  trap are excluded; tests cover all three and reconcile the capacity-matched
  concat count;
- unsupported fvcore operators and their per-sample counts are retained rather
  than silently discarded;
- latency uses 50 untimed + 200 timed iterations, synchronize before/after each
  timed call, resets memory before timing, and traverses the shared corpus in
  round-robin order;
- within each Full-versus-concat precision pair, reject different corpus hashes,
  evaluator/protocol versions, split/condition, precision, hardware fingerprint
  or input preprocessing digest;
- profiling requires a separately validated world-size-1 environment manifest
  representing the actual single-process run. Reusing a world-size-4 evaluation
  manifest, omitting `--environment-manifest`, or changing its hardware hash
  blocks before any profile child starts;
- one invocation always produces both fixed precision modes `("fp32", "amp")`;
  omitting either mode, accepting a configurable subset/order, or labeling AMP as
  the headline report fails schema validation. FP32 is the headline and AMP is
  explicitly supplemental;
- each model/precision profile runs in its own fresh spawned process; the next
  model is not loaded until the prior process exits. All four children receive identical CPU
  affinity/GPU/corpus/protocol settings, record allocated-memory baseline plus
  absolute and incremental peak, and return only a hashed profile artifact.
- after `torch.cuda.reset_peak_memory_stats()`, headline/comparison memory is
  `torch.cuda.max_memory_allocated()` (`absolute_peak_allocated_bytes`) exactly as
  the approved design. Pre-model and incremental values remain supplemental and
  can never replace the headline selector;
- Full and concat `AcceptedRun` inputs must be built-in runs whose
  adapter-execution and shared-evaluator environment/hardware pairs are equal and
  whose prediction-source snapshots are verified. These evaluation identities
  are bound by `profile_plan_id`; a cross-environment external run cannot enter
  the paper headline profile accidentally;
- `ProfileCorpus.corpus_sha256` is exactly
  `sha256(b"profile-corpus-v1\0" +
  canonical_json(payload_without_corpus_sha256))`; a checked-in fixed vector plus
  sample/input/reordering mutation tests prove the field is excluded only from
  its own digest and every other field is bound;
- `ProfilingProtocol.protocol_sha256` uses the
  `profiling-protocol-v1` domain over every typed execution/timing/FLOP/memory/
  transfer/thread/determinism/code field above, excluding only itself. A fixed
  vector and one-field mutation suite proves `profile_plan_id()` changes for each
  semantic or profiler code/commit change, and `ProfilingSuccess` repeats the
  verified protocol and code identities;

- [ ] **Step 2: Run profiling tests and confirm failure**

Run: `python -m pytest tests/resilient_v2x/test_profiling.py -q`

Expected: FAIL because profiling APIs are absent.

- [ ] **Step 3: Implement the exact profiling contract**

Before loading either model, the CLI deterministically selects 32 IDs from the
val rows of the frozen temporal manifest, rejects a selected ID from train/test,
audits every raw provenance file plus actual prepared/load artifact, config and
calibration input referenced
by those rows, and creates `profile_corpus.json` with create-if-absent semantics.
If it already exists, the CLI re-audits all sources, deterministically rebuilds
the expected canonical bytes, and reuses only an exact byte/hash match; it never
overwrites a conflict. `corpus_sha256` uses the exact `profile-corpus-v1`
domain-separated canonical formula above, excluding only that field.
Both models receive the same validated in-memory object loaded from that one file;
the concat profile may not regenerate, copy, reorder or independently decode a
second corpus manifest.

Selection hashes `(manifest_sha256, sample_id)` to choose the lowest 32 candidates,
then reorders that subset by the supplied immutable manifest index. The API rejects
duplicate indices/IDs, so reversing its input sequence cannot alter selection or
final canonical manifest order.

`profile.py` requires the frozen result-slot selectors
`--full-result-slot main.full.0ms.seed0` and
`--concat-result-slot baseline.concat.full.0ms.seed0`, plus
`--temporal-manifest`, `--dataset-root`, `--profile-corpus`,
`--environment-manifest` and
`--evidence-root`. It resolves each slot through its `TemplateBinding`,
materialized evaluation plan and create-exclusive `successful_attempt.json`;
the user cannot supply or invent a materialized-ID file. It then resolves accepted
Full-student and capacity-matched-concat seed-0 `Full@0 ms` val provenance from
the evidence store and validates:

- Linux x86_64 and CUDA;
- a strict accepted `EnvironmentManifest` with `world_size=1`, revalidated on the
  profiling host and bound into the profile plan/success IDs;
- batch size 1;
- exactly the SHA-fixed 32-sample `ProfileCorpus`;
- `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`;
- one exclusive CPU core affinity;
- `torch.set_num_threads(1)`, `set_num_interop_threads(1)`;
- TF32 and cudnn benchmark disabled;
- `pin_memory=True`, CPU-to-GPU `non_blocking=False`.

The CLI deliberately exposes no `--precision` selector: one successful invocation
must execute FP32 and AMP in that order and atomically publish one
`DualPrecisionProfilingResult`. A partial report is an unsuccessful attempt.
`profile.py` accepts `--force` only when this profile plan has no accepted pointer
and a prior attempt is failed/incomplete; it creates a new profiling attempt while
reusing the exact verified corpus. Without `--force` the retry fails clearly, and
`--force` can never replace an accepted profile. Tests preserve partial-child
diagnostics, force a second attempt, and accept only the complete four-child
report.

Latency begins with decoded CPU arrays/tensors before deterministic preprocessing and ends after post-processing plus synchronization. Disk and JPEG decoding are outside the timed callable.

Run `FlopCountAnalysis` once per decoded sample at batch size 1 and write
`per_sample_flops.json` before aggregation. The final paired manifest includes the
32 raw FLOPs records, `min/mean/p95/max`, parameter count, peak allocated memory,
latency samples/statistics, unsupported operations, both checkpoint/protocol/input
hash chains, hardware fingerprint and the single corpus hash. No headline statistic
may be emitted if one sample or one audit record is missing.

`profile_pair()` is an orchestrator: for FP32 and then AMP, it launches the Full
profile in a fresh `multiprocessing` spawn child, waits for verified completion
and process exit, then launches concat in a second fresh child with the same
affinity and device. No model/CUDA allocation survives between any two of the
four children. Each precision result records pre-model allocated bytes, absolute
peak and incremental peak; comparison/headline uses the absolute peak while
preserving pre-model and incremental values as diagnostics. The final schema requires both results, names FP32 as
`headline_precision`, names AMP as supplemental, and hashes the whole report.
The orchestrator writes all four child artifacts inside one create-exclusive
profiling attempt, finalizes its artifact manifest, then creates exactly one
`successful_attempt.json` through `AttemptStore(kind="profiling")`.
`accepted_profiling_result()` re-verifies both referenced accepted evaluation
runs, every artifact hash, the dual-precision report hash, and the success pointer;
a partial report is never accepted or discoverable by export.

- [ ] **Step 4: Run profiling unit tests**

Run: `python -m pytest tests/resilient_v2x/test_profiling.py -q`

Expected: mocked protocol tests pass; real CUDA profile remains not executed locally.

- [ ] **Step 5: Commit**

```bash
git add transvision/experiments/resilient_v2x/profiling.py transvision/experiments/resilient_v2x/schemas/profile_corpus.schema.json transvision/experiments/resilient_v2x/schemas/profiling_report.schema.json tools/resilient_v2x/profile.py tests/resilient_v2x/fixtures/profile_corpus_golden.json tests/resilient_v2x/test_profiling.py
git commit -m "feat: profile resilient v2x models"
```

---

### Task 8: 实现确定性定性渲染和候选论文证据导出

**Files:**

- Create: `tools/resilient_v2x/render_qualitative.py`
- Create: `tools/resilient_v2x/export_evidence.py`
- Create: `transvision/experiments/resilient_v2x/export.py`
- Create: `transvision/experiments/resilient_v2x/schemas/paper_patch.schema.json`
- Create: `tests/resilient_v2x/fixtures/qualitative_selection_golden.json`
- Create: `tests/resilient_v2x/test_export.py`

**Interfaces:**

```python
QUALITATIVE_PROTOCOL_SEED = 0
QUALITATIVE_SAMPLE_COUNT = 12
QUALITATIVE_CONDITIONS = (
    ("Full", 0),
    ("L-Fail", 0),
    ("C-Fail", 0),
    ("Full", 300),
)

@dataclass(frozen=True)
class RendererProtocol:
    schema_version: Literal[1]
    renderer_version: Literal["resilient-v2x-qualitative-v1"]
    renderer_code_sha256: str
    canvas_width: Literal[1600]
    canvas_height: Literal[900]
    dpi: Literal[100]
    view: Literal["ego-bev-topdown"]
    detection_range: tuple[float, float, float, float]
    palette: tuple[tuple[str, str], ...]
    point_radius_px: Literal[1]
    line_width_px: Literal[2]
    font_contract: Literal["none-no-text"]
    png_encoder: Literal["pillow-10.4.0"]
    png_compress_level: Literal[9]
    strip_png_metadata: Literal[True]
    protocol_sha256: str

@dataclass(frozen=True)
class QualitativePlan:
    schema_version: Literal[1]
    resolved_split: Literal["val"]
    protocol_seed: Literal[0]
    sample_count: Literal[12]
    conditions: tuple[tuple[str, int], ...]
    sample_ids: tuple[str, ...]
    selection_sha256: str
    checkpoint_sha256: str
    accepted_evaluation_attempts: tuple[tuple[str, str], ...]
    temporal_manifest_sha256: str
    renderer: RendererProtocol
    environment_manifest_sha256: str
    actual_hardware_fingerprint_sha256: str
    code_commit: str

@dataclass(frozen=True)
class QualitativeImageRecord:
    sample_id: str
    condition: Literal["Full", "L-Fail", "C-Fail"]
    latency_ms: Literal[0, 300]
    relative_path: str
    byte_size: int
    sha256: str

@dataclass(frozen=True)
class QualitativeSuccess:
    qualitative_plan_id: str
    attempt_id: str
    sample_ids: tuple[str, ...]
    selection_sha256: str
    checkpoint_sha256: str
    accepted_evaluation_attempts: tuple[tuple[str, str], ...]
    temporal_manifest_sha256: str
    renderer_protocol_sha256: str
    environment_manifest_sha256: str
    actual_hardware_fingerprint_sha256: str
    images: tuple[QualitativeImageRecord, ...]
    artifact_manifest_sha256: str
    content_sha256: str

@dataclass(frozen=True)
class ExportPlan:
    schema_version: Literal[1]
    matrix: Literal["all", "reproduction"]
    val_aggregate_index_id: str
    val_aggregate_index_sha256: str
    test_aggregate_index_id: str
    test_aggregate_index_sha256: str
    submission_freeze_sha256: str
    profiling_success_sha256: str
    qualitative_success_sha256: str
    result_id_mapping_sha256: str
    paper_registry_sha256: str
    paper_repository_commit: str
    paper_repository_tree: str
    exporter_code_sha256: str
    source_code_commit: str
    apply: bool

@dataclass(frozen=True)
class ExportResult:
    export_id: str
    attempt_id: str
    matrix: Literal["all", "reproduction"]
    val_aggregate_index_sha256: str
    test_aggregate_index_sha256: str
    candidate_patch: Path
    evidence_bundle: Path
    bundle_sha256: str
    export_manifest_sha256: str
    paper_repository_modified: bool

def select_qualitative_samples(
    indexed_sample_ids: Sequence[tuple[int, str]],
    protocol_seed: int,
    count: int,
) -> tuple[str, ...]
def qualitative_plan_id(plan: QualitativePlan) -> str
def render_qualitative_result(
    accepted_runs: Sequence[AcceptedRun],
    indexed_sample_ids: Sequence[tuple[int, str]],
    output_root: Path,
    resume: bool,
    force: bool,
) -> QualitativeSuccess
def accepted_qualitative_result(
    evidence_root: Path,
    qualitative_plan_id: str,
) -> QualitativeSuccess

def build_candidate_registry_patch(
    accepted_runs: Sequence[AcceptedRun],
    result_id_mapping: Mapping[str, str],
    accepted_profile: ProfilingSuccess,
) -> dict[str, object]

def export_id(plan: ExportPlan) -> str
def export_evidence_bundle(
    evidence_root: Path,
    output_dir: Path,
    paper_repository: Path,
    matrix: Literal["all", "reproduction"],
    apply: bool,
    resume: bool,
    force: bool,
) -> ExportResult
def accepted_export(
    output_dir: Path,
    export_id: str,
) -> ExportResult
```

- [ ] **Step 1: Write selection, no-cherry-pick, and no-default-apply tests**

```python
def test_qualitative_selection_does_not_read_predictions(monkeypatch) -> None:
    monkeypatch.setattr(
        "transvision.experiments.resilient_v2x.export.load_predictions",
        lambda path: (_ for _ in ()).throw(AssertionError("read too early")),
    )
    selected = select_qualitative_samples(
        [(index, f"sample-{index}") for index in range(100)],
        protocol_seed=0,
        count=12,
    )
    assert len(selected) == 12


def test_export_defaults_to_candidate_patch(tmp_path: Path) -> None:
    result = export_fixture(tmp_path, matrix="all", apply=False)
    assert result.candidate_patch.exists()
    assert result.paper_repository_modified is False
```

Add tests that:

- four required conditions use identical 12 sample IDs and seed-0 full checkpoint;
- `RendererProtocol.protocol_sha256` is
  `sha256(b"qualitative-renderer-v1\0" +
  canonical_json(payload_without_protocol_sha256))`, and
  `qualitative_plan_id()` is the `qualitative-plan-v1:` domain-separated digest
  of the complete `QualitativePlan`. Checked-in fixed vectors plus mutations of
  renderer code/settings/environment/hardware, accepted attempts, selection,
  checkpoint, manifest and code commit all change the appropriate identity;
- selection hashes
  `b"qualitative-v1\\0"+canonical_json({seed:0,sample_id})`, sorts by
  `(digest,sample_id)`, takes 12, then returns them by immutable manifest index.
  The checked-in golden vector is unchanged when indexed inputs are reversed;
  duplicate index/ID, nonzero seed, or count other than 12 fails before prediction
  access;
- missing accepted prediction condition blocks rendering;
- the accepted qualitative artifact contains exactly 48 deterministic
  `<manifest-index>-<sample-id>/<condition>-<latency>.png` images. Missing,
  extra, swapped-condition/stale evaluation attempts, wrong checkpoint, or a
  mutated image blocks its sole success pointer and export;
- two renders under the exact accepted renderer environment produce
  byte-identical PNGs. Canvas `1600x900`, DPI 100, ego-BEV top-down view,
  detection range, ordered hex palette, 1-pixel points, 2-pixel boxes, no text or
  font lookup, Pillow 10.4.0 compression level 9 and stripped PNG metadata are
  schema constants; alternate defaults are rejected before rendering;
- registry mapping has no duplicate or unknown result ID;
- only verified controlled runs can populate controlled slots;
- a reserved-test patch/render/export requires
  `accepted_test_claim_family()` for the entire 36/24-run family; 35 successful
  attempts plus one failed/missing attempt blocks all output;
- upstream/cross-protocol goes to reference slots;
- `aggregate-tables.json` is the verified collection index, not a loose list: a
  core `all` export requires exact main/ablation/sensitivity/duration/
  arrival-relative/concat coverage, and a `reproduction` export additionally
  requires the baseline controlled/cross-protocol/blocked indexes;
- `--matrix all|reproduction` is mandatory. It resolves exactly one verified val
  and one verified test aggregate index for that selector, binds both index IDs/
  hashes into `ExportResult` and the export manifest, and rejects a stale,
  multiple, missing, or cross-selector index;
- `export_id()` is
  `canonical_id("evidence-export-v1", dataclasses.asdict(ExportPlan))`.
  `ExportPlan` binds the selector, exact val/test index IDs+hashes, applicable
  component/composition freeze, profile, qualitative result, canonical
  result-ID mapping, input paper registry, clean paper repository commit/tree,
  exporter code hash/source commit and `apply` choice; output path is
  location-only and excluded. A checked-in fixed vector and one-field-at-a-time
  mutation tests prove no changed input can reuse an accepted export directory;
- export resolves the one accepted `ProfilingSuccess` for the frozen Full-versus-
  concat val pair and includes its exact dual-precision report; missing, partial,
  stale, wrong-corpus, wrong-hardware, or unaccepted profiling evidence blocks
  the entire export;
- `--apply` requires explicit paper path and successful submission gate;
- export bundle records source evidence hashes and never fabricates missing values.
- inject crashes after every candidate file, export manifest, success marker,
  atomic publish, and optional paper patch application. Retry produces/reuses one
  byte-identical accepted export; partial staging is never visible, and a
  conflicting accepted bundle or paper-tree mutation is never overwritten;
- `accepted_export(output_dir, export_id)` resolves exactly
  `<output_dir>/<export_id>/`, re-hashes its manifest, success marker and every
  byte, and rejects a different/missing output root. A test exports to two
  arbitrary absolute roots and proves the reader never guesses from
  `evidence_root` or process cwd;
- qualitative rendering itself uses a unique staging attempt and atomic directory
  publish through `AttemptStore(kind="qualitative")`; crashes after any image or
  inventory retry safely, and only a complete 48-image hashed inventory becomes
  accepted;
- `QualitativeSuccess.content_sha256` uses `qualitative-result-v1` over its
  canonical payload excluding only that field; selection and result fixed vectors
  plus image/attempt mutations propagate to the success hash.

- [ ] **Step 2: Run export tests and confirm failure**

Run: `python -m pytest tests/resilient_v2x/test_export.py -q`

Expected: FAIL because export modules are absent.

- [ ] **Step 3: Implement rendering from accepted artifacts only**

Use existing `tools/visual/local_visualizer.py` only as a geometry/rendering
primitive behind the immutable `RendererProtocol`; do not use hard-coded
`tools/visual/visual.py` or any visualizer default. The wrapper supplies every
canvas/view/range/palette/line/PNG setting, disables text/font discovery and
strips timestamps/software metadata. It verifies the pinned renderer code hash,
Pillow version and accepted world-size-1 environment/hardware before computing
`qualitative_plan_id`. Save the sample selection manifest before reading
predictions, then render Full@0, L-Fail@0, C-Fail@0, Full@300 in fixed order.

Before reading any reserved-test result, resolve the full family through
`accepted_test_claim_family()`. Candidate export cannot cherry-pick accepted
attempts out of an incomplete ledger claim.

Default export writes:

```text
candidate-evidence/<export-id>/
├── registry.patch.json
├── evidence-index.json
├── aggregate-tables.json
├── profiling-report.json
├── qualitative/
│   ├── qualitative-selection.json
│   ├── qualitative-inventory.json
│   └── images/...
└── export-manifest.json
```

No paper-repository write occurs unless `apply=True`.
Before allocating staging, the exporter validates a clean exact paper
commit/tree, hashes the registry and canonical result mapping, builds the typed
`ExportPlan`, and derives its sole `export_id`; `ExportResult.export_id`, bundle
directory, manifest, success pointer and accepted reader must all repeat that ID.
The exporter builds this tree plus its success marker in a unique sibling staging
directory, fsyncs all files, and atomically renames it to
`candidate-evidence/<export-id>/`. Existing accepted output is idempotent only
after `accepted_export(output_dir, export_id)` verifies every byte and input
index/profile/qualitative hash at the caller-supplied output root. If `--apply`
crashes after the paper patch lands, retry verifies the exact
expected paper tree/patch state before completing the success record; any
unrelated or conflicting paper mutation blocks.
`aggregate-tables.json` is copied only from a create-exclusive, hash-verified
aggregate collection index whose expected result-slot set is complete. Export
re-hashes every referenced table and rejects a stale, partial, extra, or
wrong-split table before creating `registry.patch.json`.
There is no implicit "latest aggregate" discovery: the mandatory matrix selector
is part of the export plan ID and resolves the corresponding val+test index
pair. Core-only `all` export and baseline-inclusive `reproduction` export can
coexist under one evidence root without ambiguity.
The exporter derives the required profile plan from the two frozen result slots
and audited corpus, calls `accepted_profiling_result()`, verifies the Full/concat
evaluation attempts are the same accepted runs used by the val aggregate index, and
hashes `profiling-report.json` into both `evidence-index.json` and
`export-manifest.json`. There is no flag to omit profiling evidence.
It also calls `accepted_qualitative_result()`, verifies the fixed selection,
four accepted val attempts and all 48 image hashes, then copies the inventory and
images into the candidate bundle. A selection file without accepted render
artifacts is insufficient.

Both `render_qualitative.py` and `export_evidence.py` expose mutually exclusive
`--resume` and `--force`. A crash leaves an `in_progress` journal keyed by the
deterministic plan/materialized ID; `--resume` reopens that exact attempt,
verifies every completed staged artifact, and continues from the first missing
phase. A recorded terminal invariant failure is not resumable; `--force` creates
a new non-test attempt while reusing only re-verified immutable inputs (selection,
accepted indexes/profile/qualitative records), never partial output. With neither
flag, an unfinished/failed attempt blocks. A sole accepted pointer is immutable
and makes an exact repeat idempotent regardless of flags. Tests inject crashes at
each file, manifest, publish, success-pointer and optional paper-apply boundary;
they also prove resume never creates a second attempt and force never replaces an
accepted result.

- [ ] **Step 4: Run export tests**

Run: `python -m pytest tests/resilient_v2x/test_export.py -q`

Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add tools/resilient_v2x/render_qualitative.py tools/resilient_v2x/export_evidence.py transvision/experiments/resilient_v2x/export.py transvision/experiments/resilient_v2x/schemas/paper_patch.schema.json tests/resilient_v2x/fixtures/qualitative_selection_golden.json tests/resilient_v2x/test_export.py
git commit -m "feat: export resilient v2x paper evidence"
```

---

### Task 9: 端到端 dry-run、文档和阶段回归

**Files:**

- Create: `docs/resilient_v2x/experiments.md`
- Modify: `docs/resilient_v2x/training.md`
- Modify: `tests/resilient_v2x/test_integration.py`

**Interfaces:**

No new production Python API is introduced. This task wires and documents only
the interfaces frozen in Tasks 1–8; all fixture constructors remain test-private.

- [ ] **Step 1: Add a fixture-backed end-to-end dry-run test**

The test must:

1. load the synthetic fixture-scope temporal manifest and prove the production
   materializer rejects it; use it only for diagnostic sample inference;
2. construct the internal pure `controlled_semantics_fixture_context()` with the
   official hash constants, enumerate all Resilient and
   capacity-matched-concat val/test result-slot
   templates, retaining distinct main/ablation-full/sensitivity template IDs;
3. materialize execution plan IDs and prove that applicable duplicate slots share
   the same training/evaluation plan and materialized IDs;
4. assert exactly 39 controlled training nodes (36 Resilient + 3 concat), 198 val
   evaluation nodes (162 + 36), 72 reserved-test evaluation nodes (36 + 36), and
   270 controlled evaluation nodes in total;
5. under the explicit `test-fixture-v1` store namespace, create fake accepted
   local-training attempts for a teacher, Full student and capacity-matched concat
   checkpoint, preserving their different dependency DAGs; production accepted
   pointers/exporters must reject this namespace;
6. build separate complete 36-entry multimodal `TestClaim` families for Resilient
   Full and capacity-matched concat, including all dataset,
   temporal/matrix/code/config/protocol/overlay/environment/hardware-request/
   actual-hardware/checkpoint/attempt hashes, and assert
   ledger acquisition/fsync occurs before the predictor, checkpoint or dataloader
   spy is touched;
7. materialize Full@0 evaluation, interrupt after two sample predictions, resume
   the same attempt/argv and prove that only the remaining sample IDs are inferred;
8. run the fixed evaluator, including a two-rank duplicate, and create canonical
   prediction/diagnostics/evaluator/metrics artifacts exactly once from rank 0;
   verify every diagnostics scalar and D/Q/DER tensor hash survives interruption,
   promotion and final manifest validation;
9. aggregate three paired seed fixtures for every expected core result slot and
   assert the `all` index contains exact main/ablation/sensitivity/duration/
   arrival-relative/concat table coverage on val plus explicit reserved-test
   scope markers;
10. create and validate one shared audited 32-sample profiling corpus, then use
    that exact corpus hash for Full student and concat per-sample FLOPs profiles
    in four isolated processes: FP32 Full/concat and AMP Full/concat; verify FP32
    is headline and AMP supplemental, then create and re-verify its sole profiling
    success pointer;
11. export a candidate paper patch, require that accepted dual-precision profile,
    and verify every manifest/hash chain;
12. materialize the independent smoke diagnostic and prove its two training plus
    three val evaluation attempts are absent from all controlled counts, claims,
    aggregation and exported result slots.

The internal context/store helpers are test-only constructors, are not exported
from production modules or accepted by CLI parsers, and cannot create a
controlled success pointer, reserved-test ledger entry or paper-eligible bundle.
They exercise deterministic identity/count/claim logic without allowing a
fixture-scope manifest to masquerade as the official release.

- [ ] **Step 2: Run the new end-to-end test and confirm the final wiring failure**

Run: `python -m pytest tests/resilient_v2x/test_integration.py -k experiment_evidence_end_to_end -q`

Expected: FAIL before the final fixture wiring and experiment documentation are completed.

- [ ] **Step 3: Complete fixture wiring and document exact user commands**

`experiments.md` must include exact commands for:

```bash
python tools/resilient_v2x/run_matrix.py --matrix all --split val --dry-run --data-root /abs/path/dataset --temporal-manifest /abs/path/temporal_manifest.json --evidence-root /abs/path/evidence --environment-manifest /abs/path/environment-ws1.json
CUDA_VISIBLE_DEVICES=0 python tools/resilient_v2x/run_matrix.py --matrix smoke --split val --seed 0 --smoke-max-epochs 2 --data-root /abs/path/dataset --temporal-manifest /abs/path/temporal_manifest.json --evidence-root /abs/path/evidence --environment-manifest /abs/path/environment-ws1.json --launcher none
CUDA_VISIBLE_DEVICES=0,1,2,3 python tools/resilient_v2x/run_matrix.py --matrix all --split val --data-root /abs/path/dataset --temporal-manifest /abs/path/temporal_manifest.json --evidence-root /abs/path/evidence --environment-manifest /abs/path/environment-ws4.json --launcher pytorch --nproc-per-node 4
python tools/resilient_v2x/aggregate_results.py --evidence-root /abs/path/evidence --matrix all --split val
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 taskset -c 2 python tools/resilient_v2x/profile.py --evidence-root /abs/path/evidence --environment-manifest /abs/path/environment-ws1.json --full-result-slot main.full.0ms.seed0 --concat-result-slot baseline.concat.full.0ms.seed0 --temporal-manifest /abs/path/temporal_manifest.json --dataset-root /abs/path/dataset --profile-corpus /abs/path/evidence/profile/profile_corpus.json
python tools/resilient_v2x/render_qualitative.py --evidence-root /abs/path/evidence --split val --sample-count 12
CUDA_VISIBLE_DEVICES=0,1,2,3 python tools/resilient_v2x/run_matrix.py --matrix all --split test --data-root /abs/path/dataset --temporal-manifest /abs/path/temporal_manifest.json --evidence-root /abs/path/evidence --environment-manifest /abs/path/environment-ws4.json --launcher pytorch --nproc-per-node 4
python tools/resilient_v2x/aggregate_results.py --evidence-root /abs/path/evidence --matrix all --split test
python tools/resilient_v2x/export_evidence.py --matrix all --evidence-root /abs/path/evidence --paper-repository /abs/path/ResilientV2X --output-dir /abs/path/candidate-evidence
```

This block is one ordered workflow. The complete unfiltered val invocation must
finish before aggregation, profiling, or qualitative rendering. After those
artifacts and the architecture/config/checkpoint choices are frozen, the complete
unfiltered reserved-test invocation acquires/fsyncs both 36-entry claim families
before opening any test asset. `profile.py` always writes FP32 headline and AMP
supplemental results; there is no single-precision success state.

Document filtered commands such as
`--matrix main --split val --method resilient_v2x --variant full --condition Full
--latency-ms 0 --seed 0 --resume` in a separate “single-run inspection/resume”
section. They must never appear as the producer step for aggregate, profile,
qualitative, test-claim or export commands.

The CUDA smoke command is the final preflight command: in one diagnostic DAG it
trains the teacher for two epochs, trains the Full student for two epochs, then
runs `Full@0`, `L-Fail@0`, and `Full@300`. Documentation must state that passing it
does not mark the protocol controlled or paper-ready and does not authorize
reserved-test access.

Explain the 39/198/72/270 controlled counts, concat's independent 3/36/36 branch,
result-slot template versus plan/materialized reuse, val versus reserved test,
complete one-shot claim-family ledger, controlled versus diagnostic versus
cross-protocol, and local `not executed` GPU/data status.

- [ ] **Step 4: Run the full phase suite**

Run:

```bash
python -m pytest tests/resilient_v2x/test_metric.py tests/resilient_v2x/test_experiment_protocol.py tests/resilient_v2x/test_matrix.py tests/resilient_v2x/test_evidence.py tests/resilient_v2x/test_predict_attempt.py tests/resilient_v2x/test_run_matrix.py tests/resilient_v2x/test_aggregate.py tests/resilient_v2x/test_profiling.py tests/resilient_v2x/test_export.py tests/resilient_v2x/test_integration.py -q
python -m compileall -q transvision/evaluation/metrics/resilient_v2x_metric.py transvision/experiments/resilient_v2x tools/resilient_v2x
git diff --check
```

Expected: all CPU/fixture tests pass; compileall and diff checks are silent.

- [ ] **Step 5: Run placeholder, legacy-call, and tracked-file gates**

Run:

```bash
rg -n 'TODO|TBD|NotImplemented|pass$' transvision/evaluation/metrics/resilient_v2x_metric.py transvision/experiments/resilient_v2x tools/resilient_v2x tests/resilient_v2x
rg -n 'last_checkpoint|eval_vic|SUPPROTED_MODELS|shutil\.rmtree|shell=True' transvision/evaluation/metrics/resilient_v2x_metric.py transvision/experiments/resilient_v2x tools/resilient_v2x
git ls-files transvision/evaluation/metrics/resilient_v2x_metric.py transvision/experiments/resilient_v2x tools/resilient_v2x tests/resilient_v2x docs/resilient_v2x
```

Expected: first two scans return no matches; Git lists every planned source, schema, tool, test, and document.

- [ ] **Step 6: Commit**

```bash
git add docs/resilient_v2x/experiments.md docs/resilient_v2x/training.md tests/resilient_v2x/test_integration.py
git commit -m "docs: document resilient v2x experiment workflow"
```

## Completion Gate

This plan is complete only when:

- evaluator is fixed to `resilient-v2x-ap-v1` and IoU thresholds `(0.5, 0.7)`,
  consumes exactly the Runner predictions and hash-bound normalized manifest GT,
  returns one pure raw-plus-flat `EvaluationBundle`, passes all golden AP fixtures,
  performs stable manifest-order DDP de-duplication, and permits rank 0 alone to
  write canonical outputs;
- dry-run covers every distinct paper result-slot template while reusing only
  matching plan/materialized IDs, emits exactly 39 controlled training, 198 val,
  72 reserved-test and 270 total evaluation nodes, and never invents plan IDs
  without data;
- the capacity-matched concat branch has exactly 3 independent training, 36 val
  and 36 reserved-test runs, never depends on a Resilient teacher, and uses its
  seed checkpoint across conditions;
- attempts, command manifests, plan realizations, provenance, per-sample
  prediction progress and ledgers are immutable and hash-verified; interrupted
  prediction resumes the same attempt/argv and never recomputes a completed
  manifest sample;
- every prediction sample durably carries hash-verified source/horizon/support/
  reason, D/Q, gamma/reliability, flags and DER descriptor/support/weights
  diagnostics (or schema-valid explicit nulls) into the run manifest;
- a complete 36-run multimodal or 24-run LiDAR-only `TestClaim` contains the
  frozen dataset/matrix/code/config/protocol/overlay/environment/checkpoint/attempt
  fields plus temporal manifest, requested and actual hardware hashes, realized
  environment manifest, and fixed evaluator/IoU thresholds, and is
  acquired/fsynced before any
  reserved-test checkpoint, dataloader, prediction or metric access; test cannot
  use force or create a second family;
- `--matrix smoke --split val --seed 0 --smoke-max-epochs 2` creates only its
  independent two-training/three-evaluation diagnostic protocol and attempts;
  smoke can never satisfy a controlled gate, test claim, aggregate, profile,
  qualitative selection or paper slot;
- Full student and capacity-matched concat profiling use one SHA-audited
  32-sample raw-input corpus, preserve per-sample fvcore FLOPs and
  `min/mean/p95/max`, share the exact corpus/protocol/hardware fingerprint, and
  produce a mandatory FP32 headline plus AMP supplemental report in four isolated
  processes;
- paired three-seed aggregation, PDR, method difference, profiling, qualitative selection, and candidate evidence export are automated;
- legacy zero-box, output-deletion, second-model evaluation, and val-as-test paths are unreachable from controlled execution.
