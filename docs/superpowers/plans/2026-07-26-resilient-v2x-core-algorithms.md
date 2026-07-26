# Resilient V2X 核心算法与契约 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 交付不依赖 DAIR-V2X 数据文件、可用 synthetic tensors 独立验证的因果选择、BEV 几何对齐、PTF、可靠性修复、DER 和蒸馏核心。

**Architecture:** 保留 `ResilientV2XNet` 作为后续薄编排层；本计划只新增 `transvision.models.resilient_v2x` 纯核心包。协议状态由不可变 dataclass 表达，tensor 运算由小型 `nn.Module` 完成，unsupported 分支必须在调用 PTF 前短路。

**Tech Stack:** Python 3.10、PyTorch 2.0.1、pytest 7.4.4、MMEngine 0.10.7；CPU 单元测试不依赖 CUDA、自定义算子或真实数据。

## Global Constraints

- 规范来源固定为 `docs/superpowers/specs/2026-07-26-resilient-v2x-reproduction-design.md`；本计划不得重新解释 `k=3`、`Delta_t_ms=100`、`alpha=0.9`、BEV shape `256x288x288` 或 descriptor 维度 783。
- 先完成环境计划的 Task 1–2，使核心包可以在未加载 CUDA custom ops 时独立导入，且 `python -m pytest --version` 输出 `pytest 7.4.4`；没有 PyTorch 的宿主机只允许做静态检查，不能把测试标记为通过。
- 所有 `python` 命令必须在环境计划 Task 2 生成的
  `/Users/libin/transvision/.venv/resilient-v2x-runtime` 或相同锁定 Docker image
  内执行；新 shell 先激活并验证 Python 3.10 与精确依赖版本，禁止用系统
  Python 代替。
- 所有公开 API 都显式接收协议参数，不在 constructor 内藏论文参数默认值。
- 所有 tensor API 固定使用 `[B,C,Y,X]`；位移 channel 固定为 `[D_x,D_y]`。
- 不可用 raw payload 不得经过 encoder；本计划用 encoded-history fixture 验证 gather/scatter 和 PTF 的不变性，raw gather 在数据计划实现。
- 每个任务结束后只提交该任务列出的文件；不要顺带格式化或修改现有无关代码。
- 测试命令均从 `/Users/libin/transvision` 执行。

---

### Task 1: 建立核心包和不可变协议契约

**Files:**

- Create: `transvision/models/resilient_v2x/__init__.py`
- Create: `transvision/models/resilient_v2x/contracts.py`
- Create: `tests/resilient_v2x/__init__.py`
- Create: `tests/resilient_v2x/test_contracts.py`

**Interfaces:**

```python
class Agent(str, Enum):
    EGO = "ego"
    RSU = "rsu"

class Modality(str, Enum):
    LIDAR = "lidar"
    CAMERA = "camera"

class UnsupportedReason(str, Enum):
    EMPTY_ARRIVAL_SET = "empty_arrival_set"
    EMPTY_MODALITY_HISTORY = "empty_modality_history"
    UNSUPPORTED_HORIZON = "unsupported_horizon"
    INVALID_TIMESTAMP = "invalid_timestamp"
    INVALID_POSE = "invalid_pose"
    INVALID_CALIBRATION = "invalid_calibration"
    MISSING_PAYLOAD = "missing_payload"
    METHOD_NOT_APPLICABLE = "method_not_applicable"

@dataclass(frozen=True)
class SourceCandidate:
    packet_id: str
    n_s: int
    tau_s_ms: int
    arrival_tau_ms: int | None
    payload_valid: bool
    timestamp_valid: bool
    pose_valid: bool
    calibration_valid: bool
    faulted: bool

@dataclass(frozen=True)
class BranchSelection:
    agent: Agent
    modality: Modality
    supported: bool
    source: SourceCandidate | None
    horizon: int | None
    observed: bool
    propagated: bool
    rejected: tuple[RejectedCandidate, ...]
    reason: UnsupportedReason | None
```

- [ ] **Step 1: Write the failing contract tests**

```python
from dataclasses import FrozenInstanceError

import pytest

from transvision.models.resilient_v2x.contracts import (
    Agent,
    BranchSelection,
    Modality,
    SourceCandidate,
    UnsupportedReason,
)


def test_source_candidate_is_immutable() -> None:
    candidate = SourceCandidate(
        packet_id="seq-1:10",
        n_s=10,
        tau_s_ms=1000,
        arrival_tau_ms=1100,
        payload_valid=True,
        timestamp_valid=True,
        pose_valid=True,
        calibration_valid=True,
        faulted=False,
    )
    with pytest.raises(FrozenInstanceError):
        candidate.n_s = 11


def test_supported_selection_requires_source_and_horizon() -> None:
    with pytest.raises(ValueError, match="supported branch requires"):
        BranchSelection(
            agent=Agent.EGO,
            modality=Modality.LIDAR,
            supported=True,
            source=None,
            horizon=None,
            observed=False,
            propagated=False,
            rejected=(),
            reason=None,
        )


def test_unsupported_selection_requires_reason_and_null_source() -> None:
    selection = BranchSelection.unsupported(
        agent=Agent.RSU,
        modality=Modality.CAMERA,
        reason=UnsupportedReason.EMPTY_ARRIVAL_SET,
        rejected=(),
    )
    assert selection.source is None
    assert selection.horizon is None
    assert selection.reason is UnsupportedReason.EMPTY_ARRIVAL_SET
```

- [ ] **Step 2: Run the test and confirm the import failure**

Run: `python -m pytest tests/resilient_v2x/test_contracts.py -q`

Expected: FAIL with `ModuleNotFoundError: No module named 'transvision.models.resilient_v2x'`.

- [ ] **Step 3: Implement the immutable contracts and invariant checks**

`contracts.py` must contain the enums above plus this validation shape:

```python
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class ProtocolInvariantError(ValueError):
    """Raised when a protocol object is internally inconsistent."""


@dataclass(frozen=True)
class RejectedCandidate:
    packet_id: str
    n_s: int
    reason: UnsupportedReason


@dataclass(frozen=True)
class BranchSelection:
    agent: Agent
    modality: Modality
    supported: bool
    source: SourceCandidate | None
    horizon: int | None
    observed: bool
    propagated: bool
    rejected: tuple[RejectedCandidate, ...]
    reason: UnsupportedReason | None

    def __post_init__(self) -> None:
        if self.supported:
            if self.source is None or self.horizon is None or self.reason is not None:
                raise ProtocolInvariantError(
                    "supported branch requires source and horizon and forbids reason"
                )
            if self.horizon < 0:
                raise ProtocolInvariantError("horizon must be non-negative")
            if self.observed == self.propagated:
                raise ProtocolInvariantError(
                    "supported branch requires exactly one observed/propagated flag"
                )
        elif (
            self.source is not None
            or self.horizon is not None
            or self.reason is None
            or self.observed
            or self.propagated
        ):
            raise ProtocolInvariantError(
                "unsupported branch requires null source/horizon, false flags, and reason"
            )

    @classmethod
    def unsupported(
        cls,
        agent: Agent,
        modality: Modality,
        reason: UnsupportedReason,
        rejected: tuple[RejectedCandidate, ...],
    ) -> "BranchSelection":
        return cls(
            agent=agent,
            modality=modality,
            supported=False,
            source=None,
            horizon=None,
            observed=False,
            propagated=False,
            rejected=rejected,
            reason=reason,
        )
```

`__init__.py` 必须只导出稳定 public API，并为后续模块预留的名字等到对应任务完成后再加入，避免不存在的导入。

- [ ] **Step 4: Run contract tests**

Run: `python -m pytest tests/resilient_v2x/test_contracts.py -q`

Expected: `3 passed`.

- [ ] **Step 5: Commit**

```bash
git add transvision/models/resilient_v2x/__init__.py transvision/models/resilient_v2x/contracts.py tests/resilient_v2x/__init__.py tests/resilient_v2x/test_contracts.py
git commit -m "feat: add resilient v2x protocol contracts"
```

---

### Task 2: 实现当前 ego BEV 几何对齐

**Files:**

- Create: `transvision/models/resilient_v2x/geometry.py`
- Create: `tests/resilient_v2x/test_geometry.py`
- Modify: `transvision/models/resilient_v2x/__init__.py`

**Interfaces:**

```python
@dataclass(frozen=True)
class BEVGridSpec:
    x_min: float
    y_min: float
    resolution: float
    height: int
    width: int

def compose_source_to_target(
    source_world_from_agent: Tensor,
    target_world_from_ego: Tensor,
) -> Tensor

def build_backward_grid(
    source_to_target: Tensor,
    spec: BEVGridSpec,
    dtype: torch.dtype,
    device: torch.device,
) -> Tensor

def align_bev_to_target(
    source: Tensor,
    source_to_target: Tensor,
    spec: BEVGridSpec,
) -> Tensor

def warp_with_displacement(source: Tensor, displacement_cells: Tensor) -> Tensor
```

- [ ] **Step 1: Write identity, translation, rotation, and invalid-matrix tests**

Use a `5x5` fixture with `resolution=1.0`, a single impulse at cell `(2,2)`, and exact homogeneous transforms:

```python
import math

import pytest
import torch

from transvision.models.resilient_v2x.geometry import (
    BEVGridSpec,
    align_bev_to_target,
    compose_source_to_target,
    warp_with_displacement,
)


SPEC = BEVGridSpec(x_min=0.0, y_min=-2.5, resolution=1.0, height=5, width=5)


def impulse() -> torch.Tensor:
    value = torch.zeros(1, 1, 5, 5)
    value[0, 0, 2, 2] = 1.0
    return value


def test_identity_alignment_is_exact() -> None:
    actual = align_bev_to_target(impulse(), torch.eye(4).unsqueeze(0), SPEC)
    torch.testing.assert_close(actual, impulse(), atol=0.0, rtol=0.0)


def test_positive_x_translation_moves_impulse_one_column() -> None:
    transform = torch.eye(4).unsqueeze(0)
    transform[:, 0, 3] = 1.0
    actual = align_bev_to_target(impulse(), transform, SPEC)
    assert actual[0, 0, 2, 3].item() == pytest.approx(1.0)


def test_positive_quarter_turn_rotates_source_counter_clockwise() -> None:
    symmetric = BEVGridSpec(
        x_min=-2.5,
        y_min=-2.5,
        resolution=1.0,
        height=5,
        width=5,
    )
    source = torch.zeros(1, 1, 5, 5)
    source[0, 0, 2, 3] = 1.0
    transform = torch.eye(4).unsqueeze(0)
    transform[:, 0, 0] = 0.0
    transform[:, 0, 1] = -1.0
    transform[:, 1, 0] = 1.0
    transform[:, 1, 1] = 0.0
    actual = align_bev_to_target(source, transform, symmetric)
    assert actual[0, 0, 3, 2].item() == pytest.approx(1.0)


def test_compose_source_to_target_uses_inverse_target_times_source() -> None:
    source_world_from_agent = torch.tensor(
        [
            [0.0, -1.0, 0.0, 13.0],
            [1.0, 0.0, 0.0, 2.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    ).unsqueeze(0)
    target_world_from_ego = torch.tensor(
        [
            [1.0, 0.0, 0.0, 10.0],
            [0.0, 1.0, 0.0, -2.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    ).unsqueeze(0)
    expected = torch.tensor(
        [
            [0.0, -1.0, 0.0, 3.0],
            [1.0, 0.0, 0.0, 4.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    ).unsqueeze(0)
    actual = compose_source_to_target(
        source_world_from_agent,
        target_world_from_ego,
    )
    torch.testing.assert_close(actual, expected, atol=0.0, rtol=0.0)


def test_backward_displacement_uses_x_then_y_channels() -> None:
    displacement = torch.zeros(1, 2, 5, 5)
    displacement[:, 0] = -1.0
    actual = warp_with_displacement(impulse(), displacement)
    assert actual[0, 0, 2, 3].item() == pytest.approx(1.0)


def test_non_finite_transform_fails_closed() -> None:
    transform = torch.eye(4).unsqueeze(0)
    transform[:, 0, 0] = math.nan
    with pytest.raises(ValueError, match="finite"):
        align_bev_to_target(impulse(), transform, SPEC)


def test_singular_transform_fails_closed() -> None:
    transform = torch.eye(4).unsqueeze(0)
    transform[:, 3, 3] = 0.0
    with pytest.raises(ValueError, match="invertible"):
        align_bev_to_target(impulse(), transform, SPEC)
```

- [ ] **Step 2: Run the tests and confirm the missing-module failure**

Run: `python -m pytest tests/resilient_v2x/test_geometry.py -q`

Expected: FAIL because `geometry.py` does not exist.

- [ ] **Step 3: Implement cell-center backward sampling**

The implementation must:

1. validate `[B,4,4]`, finite values, and invertibility with `torch.linalg.inv_ex`;
2. implement `compose_source_to_target()` exactly as
   `inverse(target_world_from_ego) @ source_world_from_agent`, with no reversed
   multiplication or silent pseudo-inverse fallback;
3. create target centers with `x=x_min+(j+0.5)*resolution` and `y=y_min+(i+0.5)*resolution`;
4. transform target centers by `inverse(source_to_target)`;
5. normalize source indices as `2*(index+0.5)/size-1`;
6. call `grid_sample(mode="bilinear", padding_mode="zeros", align_corners=False)`.

`warp_with_displacement` must use the exact normalization:

```python
def warp_with_displacement(source: Tensor, displacement_cells: Tensor) -> Tensor:
    if source.ndim != 4 or displacement_cells.ndim != 4:
        raise ValueError("source and displacement must be rank-4 tensors")
    batch, _, height, width = source.shape
    if displacement_cells.shape != (batch, 2, height, width):
        raise ValueError("displacement shape must be [B,2,Y,X]")
    rows = torch.arange(height, dtype=source.dtype, device=source.device)
    cols = torch.arange(width, dtype=source.dtype, device=source.device)
    row_grid, col_grid = torch.meshgrid(rows, cols, indexing="ij")
    x = col_grid.unsqueeze(0) + displacement_cells[:, 0]
    y = row_grid.unsqueeze(0) + displacement_cells[:, 1]
    u = 2.0 * (x + 0.5) / width - 1.0
    v = 2.0 * (y + 0.5) / height - 1.0
    grid = torch.stack((u, v), dim=-1)
    return torch.nn.functional.grid_sample(
        source,
        grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=False,
    )
```

- [ ] **Step 4: Run the geometry tests**

Run: `python -m pytest tests/resilient_v2x/test_geometry.py -q`

Expected: all tests pass; identity uses zero absolute tolerance.

- [ ] **Step 5: Commit**

```bash
git add transvision/models/resilient_v2x/geometry.py transvision/models/resilient_v2x/__init__.py tests/resilient_v2x/test_geometry.py
git commit -m "feat: add ego frame bev alignment"
```

---

### Task 3: 实现到达集合、fault 后回退和 source selection

**Files:**

- Create: `transvision/models/resilient_v2x/causal_repair.py`
- Create: `tests/resilient_v2x/test_causal_repair.py`
- Modify: `transvision/models/resilient_v2x/__init__.py`

**Interfaces:**

```python
def arrived_candidates(
    agent: Agent,
    target_tau_ms: int,
    candidates: Sequence[SourceCandidate],
) -> tuple[SourceCandidate, ...]

def select_causal_source(
    agent: Agent,
    modality: Modality,
    n_t: int,
    target_tau_ms: int,
    delta_t_ms: int,
    history_limit: int,
    candidates: Sequence[SourceCandidate],
) -> BranchSelection

def age_decay(horizon: Tensor, alpha: float) -> Tensor

def branch_reliability(
    selection: BranchSelection,
    confidence: Tensor | None,
    alpha: float,
    batch_index: int,
) -> Tensor

def normalized_selected_rsu_delay(
    selections: Sequence[BranchSelection],
    history_limit: int,
    delta_t_ms: int,
) -> float
```

- [ ] **Step 1: Write protocol fixtures for the approved horizon table**

```python
import pytest

from transvision.models.resilient_v2x.causal_repair import select_causal_source
from transvision.models.resilient_v2x.contracts import Agent, Modality, SourceCandidate


def history(delay_ms: int, fault_target: bool) -> list[SourceCandidate]:
    return [
        SourceCandidate(
            packet_id=f"seq:{n}",
            n_s=n,
            tau_s_ms=n * 100,
            arrival_tau_ms=n * 100 + delay_ms,
            payload_valid=True,
            timestamp_valid=True,
            pose_valid=True,
            calibration_valid=True,
            faulted=fault_target and n == 10,
        )
        for n in range(7, 11)
    ]


@pytest.mark.parametrize(
    ("delay_ms", "fault_target", "expected_h"),
    [(0, False, 0), (0, True, 1), (100, False, 1), (100, True, 1),
     (200, False, 2), (200, True, 2), (300, False, 3), (300, True, 3)],
)
def test_rsu_global_target_fault_horizon_table(
    delay_ms: int,
    fault_target: bool,
    expected_h: int,
) -> None:
    result = select_causal_source(
        agent=Agent.RSU,
        modality=Modality.LIDAR,
        n_t=10,
        target_tau_ms=1000,
        delta_t_ms=100,
        history_limit=3,
        candidates=history(delay_ms, fault_target),
    )
    assert result.supported
    assert result.horizon == expected_h


def test_newest_invalid_candidate_falls_back() -> None:
    candidates = history(0, False)
    candidates[-1] = SourceCandidate(
        packet_id="seq:10",
        n_s=10,
        tau_s_ms=1000,
        arrival_tau_ms=1000,
        payload_valid=True,
        timestamp_valid=True,
        pose_valid=False,
        calibration_valid=True,
        faulted=False,
    )
    result = select_causal_source(
        Agent.EGO, Modality.CAMERA, 10, 1000, 100, 3, candidates
    )
    assert result.source is not None
    assert result.source.n_s == 9
    assert result.rejected[0].packet_id == "seq:10"
```

Add separate tests for:

- RSU empty arrived set gives `empty_arrival_set`;
- all candidates faulted gives `empty_modality_history`;
- `h=3` is supported and explicit `h=4` gives `unsupported_horizon`;
- ego ignores future candidates and uses the current cutoff;
- candidates supplied out of order still select the greatest causal valid `n_s`;
- a packet arriving at 1001 ms is unavailable at target 1000 ms and no feature interpolation occurs;
- when every arrived/unmasked candidate has invalid pose, calibration, timestamp, or missing payload, the final branch reason is the first failed check of the newest such candidate, not the generic empty-history reason;
- normalized RSU delay uses only selected arrived LiDAR/Camera packets, takes their maximum, divides by `k*Delta_t_ms`, clamps to `[0,1]`, and is zero when neither RSU branch is supported;
- a fixed condition label or the future target packet delay cannot enter normalized RSU delay;
- invalid `delta_t_ms`, inconsistent `tau_s_ms != n_s*delta_t_ms`, negative delay, or non-integral horizon raises `ProtocolInvariantError`.

- [ ] **Step 2: Run the causal tests and confirm failure**

Run: `python -m pytest tests/resilient_v2x/test_causal_repair.py -q`

Expected: FAIL because the selection functions are absent.

- [ ] **Step 3: Implement deterministic filter order and reason precedence**

Use this exact order:

1. reject future source tick;
2. for RSU reject packets with null/future arrival;
3. reject faulted payload;
4. reject invalid timestamp, payload, pose, calibration;
5. reject horizon beyond `history_limit`;
6. select the remaining candidate with greatest `(n_s, packet_id)`.

When none remains, reason precedence is:

```python
def unsupported_reason(
    agent: Agent,
    arrived_count: int,
    unmasked_count: int,
    within_horizon_count: int,
    newest_metadata_rejection: UnsupportedReason | None,
) -> UnsupportedReason:
    if agent is Agent.RSU and arrived_count == 0:
        return UnsupportedReason.EMPTY_ARRIVAL_SET
    if unmasked_count == 0:
        return UnsupportedReason.EMPTY_MODALITY_HISTORY
    if newest_metadata_rejection is not None:
        return newest_metadata_rejection
    if within_horizon_count == 0:
        return UnsupportedReason.UNSUPPORTED_HORIZON
    raise ProtocolInvariantError("unsupported reason requested with valid candidate")
```

Within one candidate, metadata checks use
`INVALID_TIMESTAMP → MISSING_PAYLOAD → INVALID_POSE → INVALID_CALIBRATION`.
Across candidates, `newest_metadata_rejection` comes from the greatest `(n_s,packet_id)`
after arrival/fault filtering. Thus a failed newest candidate is preserved in diagnostics,
but an older valid candidate still wins before the unsupported-reason function is called.

`observed` is true only for `agent is Agent.EGO and h == 0`; every other supported source is propagated.

`normalized_selected_rsu_delay()` reads only supported RSU selections:

```python
realized_delays = [
    selection.source.arrival_tau_ms - selection.source.tau_s_ms
    for selection in selections
    if selection.supported
    and selection.agent is Agent.RSU
    and selection.source is not None
    and selection.source.arrival_tau_ms is not None
]
if not realized_delays:
    return 0.0
denominator = history_limit * delta_t_ms
return min(max(max(realized_delays) / denominator, 0.0), 1.0)
```

- [ ] **Step 4: Run causal and contract tests**

Run: `python -m pytest tests/resilient_v2x/test_contracts.py tests/resilient_v2x/test_causal_repair.py -q`

Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add transvision/models/resilient_v2x/causal_repair.py transvision/models/resilient_v2x/__init__.py tests/resilient_v2x/test_causal_repair.py
git commit -m "feat: add causal source selection"
```

---

### Task 4: 实现 horizon-conditioned nonlinear/linear PTF

**Files:**

- Create: `transvision/models/resilient_v2x/ptf.py`
- Create: `tests/resilient_v2x/test_ptf.py`
- Modify: `transvision/models/resilient_v2x/__init__.py`

**Interfaces:**

```python
@dataclass(frozen=True)
class PTFOutput:
    displacement: Tensor
    confidence: Tensor

class HorizonConditionedPTF(nn.Module):
    def __init__(
        self,
        in_channels: int,
        history_positions: int,
        history_limit: int,
        projected_channels: int,
        context_channels: int,
        max_low_resolution_cells: float,
        mode: Literal["nonlinear", "linear"],
    ) -> None

    def build_context(
        self,
        aligned_history: Tensor,
        availability: Tensor,
    ) -> Tensor

    def query(
        self,
        context: Tensor,
        query_agent_index: Tensor,
        horizon: Tensor,
    ) -> PTFOutput
```

`aligned_history` shape 固定 `[B,2,4,256,288,288]`，`availability` 固定 `[B,2,4]`。

- [ ] **Step 1: Write exact shape, initialization, bounds, and masking tests**

```python
import torch

from transvision.models.resilient_v2x.ptf import HorizonConditionedPTF


def make_ptf(mode: str = "nonlinear") -> HorizonConditionedPTF:
    return HorizonConditionedPTF(
        in_channels=256,
        history_positions=4,
        history_limit=3,
        projected_channels=64,
        context_channels=128,
        max_low_resolution_cells=8.0,
        mode=mode,
    ).eval()


def test_ptf_initial_output_and_shape() -> None:
    model = make_ptf()
    history = torch.randn(1, 2, 4, 256, 32, 32)
    availability = torch.ones(1, 2, 4, dtype=torch.bool)
    context = model.build_context(history, availability)
    output = model.query(context, torch.tensor([0]), torch.tensor([3]))
    assert context.shape == (1, 128, 8, 8)
    assert output.displacement.shape == (1, 2, 32, 32)
    assert output.confidence.shape == (1, 1, 32, 32)
    torch.testing.assert_close(output.displacement, torch.zeros_like(output.displacement))
    torch.testing.assert_close(
        output.confidence,
        torch.full_like(output.confidence, 0.5),
    )


def test_horizon_zero_forces_zero_displacement() -> None:
    model = make_ptf()
    history = torch.randn(2, 2, 4, 256, 16, 16)
    availability = torch.ones(2, 2, 4, dtype=torch.bool)
    output = model.query(
        model.build_context(history, availability),
        torch.tensor([0, 1]),
        torch.tensor([0, 0]),
    )
    assert torch.count_nonzero(output.displacement).item() == 0


def test_unavailable_payload_cannot_change_context() -> None:
    model = make_ptf()
    first = torch.randn(1, 2, 4, 256, 16, 16)
    second = first.clone()
    second[:, 1, 3] = torch.randn_like(second[:, 1, 3]) * 1000
    availability = torch.ones(1, 2, 4, dtype=torch.bool)
    availability[:, 1, 3] = False
    torch.testing.assert_close(
        model.build_context(first, availability),
        model.build_context(second, availability),
        atol=0.0,
        rtol=0.0,
    )
```

Also add tests that:

- `model.projected_spatial_shape((288,288)) == (72,72)`，在不分配完整八路 tensor 的情况下锁定 production shape；
- invalid `h<0` and `h>3` are rejected before FiLM;
- every displacement component is within `[-32,32]`;
- every confidence value is within `[0,1]`;
- after setting non-zero final-head weights, nonlinear `D(2)` is not forced to `2*D(1)`;
- linear mode satisfies `D(2)=2*D(1)` and `D(3)=3*D(1)` within tolerance;
- another batch member's horizon cannot change sample 0 output.
- module-structure assertions prove both history-projection convolutions and the
  context-stem 1x1 convolution have `bias is None`.
- after projection plus agent/relative-time embeddings, every unavailable
  64-channel slot is still exactly zero; its separate mask plane is the only
  missingness signal entering the context stem.
- an exact module inventory freezes all remaining bias choices and asserts
  `sum(p.numel() for p in model.parameters() if p.requires_grad) == 1_577_923`
  for `make_ptf()`;

- [ ] **Step 2: Run PTF tests and confirm failure**

Run: `python -m pytest tests/resilient_v2x/test_ptf.py -q`

Expected: FAIL because `HorizonConditionedPTF` does not exist.

- [ ] **Step 3: Implement the approved network exactly**

Implementation requirements:

- shared projection `Conv3x3 256→128 stride2, bias=False, GN32, SiLU` then
  `Conv3x3 128→64 stride2, bias=False, GN16, SiLU`;
- add learned agent and relative-time embeddings, then multiply the resulting
  `(projection + agent_embedding + relative_time_embedding)` by the broadcast
  availability mask before context concatenation;
- concatenate eight projected features and eight mask planes to 520 channels;
- context stem `Conv1x1 520→128, bias=False, GN32, SiLU`;
- three independent FiLM residual blocks; each 3x3 convolution is followed by GN
  and therefore uses `bias=False`;
- FiLM MLP `Linear65→128, bias=True, SiLU, Linear128→768, bias=True`; final
  linear weight and bias are zero initialized;
- displacement/confidence first 3x3 convolutions use `bias=False` before GN;
  their final 1x1 convolutions use `bias=True`. Displacement final weight/bias
  and confidence final weight/bias are zero initialized;
- bilinear upsample with `align_corners=False`, displacement multiplied by four;
- linear mode queries FiLM with `1/k`, emits `(8/k)*tanh(raw)` and multiplies by `h`;
- explicit `torch.where(h==0, 0, D)` after upsampling.

The availability-safe projection must gather valid rows rather than convolve zeroed invalid rows:

```python
flat = aligned_history.reshape(batch * 8, channels, height, width)
flat_mask = availability.reshape(batch * 8)
projected_height, projected_width = self.projected_spatial_shape((height, width))
projected = flat.new_zeros(
    (batch * 8, 64, projected_height, projected_width)
)
valid_index = flat_mask.nonzero(as_tuple=False).flatten()
if valid_index.numel() > 0:
    projected_valid = self.history_projection(flat.index_select(0, valid_index))
    if projected_valid.shape[-2:] != (projected_height, projected_width):
        raise RuntimeError("PTF projection spatial shape mismatch")
    projected.index_copy_(0, valid_index, projected_valid)
projected = projected.view(
    batch,
    2,
    4,
    64,
    projected_height,
    projected_width,
)
```

`projected_spatial_shape()` rejects spatial dimensions not divisible by four and returns
`(height//4,width//4)`. Both heads upsample back to the input history spatial shape; the
displacement scale is exactly four for every accepted shape. Production remains
`288→72→288`, while unit fixtures use `16→4→16` and `32→8→32`.

- [ ] **Step 4: Run PTF tests**

Run: `python -m pytest tests/resilient_v2x/test_ptf.py -q`

Expected: all tests pass on CPU; tensor fixtures use `16x16`/`32x32` spatial sizes, while a shape-only contract locks production `288→72→288` behavior.

- [ ] **Step 5: Commit**

```bash
git add transvision/models/resilient_v2x/ptf.py transvision/models/resilient_v2x/__init__.py tests/resilient_v2x/test_ptf.py
git commit -m "feat: implement horizon conditioned ptf"
```

---

### Task 5: 组合选择、对齐、PTF、衰减和可序列化诊断

**Files:**

- Modify: `transvision/models/resilient_v2x/contracts.py`
- Modify: `transvision/models/resilient_v2x/causal_repair.py`
- Create: `tests/resilient_v2x/test_repair_pipeline.py`

**Interfaces:**

```python
@dataclass(frozen=True)
class BranchDiagnostics:
    agent: Agent
    modality: Modality
    supported: bool
    source_tick: int | None
    source_tau_ms: int | None
    horizon: int | None
    observed: bool
    propagated: bool
    gamma: float | None
    reliability: float
    ptf_queried: bool
    displacement: Tensor | None
    confidence: Tensor | None
    reason: UnsupportedReason | None

@dataclass
class RepairedBranch:
    feature: Tensor
    reliability: Tensor
    support: Tensor
    observed: Tensor
    propagated: Tensor
    normalized_age: Tensor
    displacement: Tensor | None
    confidence: Tensor | None
    diagnostics: tuple[BranchDiagnostics, ...]

class CausalBranchRepair(nn.Module):
    def __init__(
        self,
        grid_spec: BEVGridSpec,
        channels: int,
        alpha: float,
        history_limit: int,
    ) -> None

    def expected_output_shape(
        self,
        batch_size: int,
    ) -> tuple[int, int, int, int]

    def forward(
        self,
        selected_feature: Tensor,
        source_to_target: Tensor,
        ptf_context: Tensor,
        ptf: HorizonConditionedPTF,
        query_agent_index: Tensor,
        selections: Sequence[BranchSelection],
    ) -> RepairedBranch
```

- [ ] **Step 1: Write supported and unsupported pipeline tests**

All arithmetic fixtures use a `16x16` grid so this CPU suite never allocates a production
`[B,256,288,288]` tensor. The supported fixture must assert:

```python
SMALL_SPEC = BEVGridSpec(
    x_min=0.0,
    y_min=-2.56,
    resolution=0.32,
    height=16,
    width=16,
)
repair = CausalBranchRepair(
    grid_spec=SMALL_SPEC,
    channels=256,
    alpha=0.9,
    history_limit=3,
)

assert output.feature.shape == (2, 256, 16, 16)
torch.testing.assert_close(output.feature[0], aligned_warped[0])
assert output.diagnostics[0].gamma == 1.0
assert output.diagnostics[0].reliability == 1.0
assert output.diagnostics[1].gamma == pytest.approx(0.9 ** 3)
assert output.diagnostics[1].reliability == pytest.approx(
    (0.9 ** 3) * output.confidence[1].mean().item()
)
torch.testing.assert_close(
    output.normalized_age,
    torch.tensor([0.0, 1.0]),
)
```

Add this production shape-only contract; it must not allocate a feature tensor:

```python
def test_production_repair_shape_contract_without_feature_allocation() -> None:
    production = CausalBranchRepair(
        grid_spec=BEVGridSpec(
            x_min=0.0,
            y_min=-46.08,
            resolution=0.32,
            height=288,
            width=288,
        ),
        channels=256,
        alpha=0.9,
        history_limit=3,
    )
    assert production.expected_output_shape(batch_size=2) == (2, 256, 288, 288)
```

The unsupported fixture passes a spy PTF whose `query()` raises if called through the
explicit `ptf=` argument, then verifies:

```python
assert spy.query_count == 0
assert torch.count_nonzero(output.feature).item() == 0
assert output.displacement is None
assert output.confidence is None
assert output.diagnostics[0].ptf_queried is False
assert output.diagnostics[0].gamma is None
assert output.diagnostics[0].reliability == 0.0
assert output.diagnostics[0].displacement is None
assert output.diagnostics[0].confidence is None
```

Add JSON serialization assertion that unsupported `source_tick`, `source_tau_ms`, `horizon`, `gamma`, `displacement`, and `confidence` become JSON `null`, never a zero-valued prediction. Supported diagnostics hold the per-sample `D/Q` slices detached to CPU by the diagnostics exporter; internal scatter-zero tensors are not used as evidence values.
Add one exact fixture for supported horizons `h=(0,1,2,3)` and one unsupported
branch: `normalized_age==(0,1/3,2/3,1,0)`. Reject a horizon outside
`[0,history_limit]`, a non-finite age, or any supported age outside `[0,1]`.

Add a two-modality injection test that passes a LiDAR spy PTF and a Camera spy PTF with
different deterministic `D/Q` outputs through the same `CausalBranchRepair`. Assert each spy
is queried only for its own modality, the two outputs remain distinct, and the repair module
does not own or register either PTF's parameters.

- [ ] **Step 2: Run the pipeline test and confirm failure**

Run: `python -m pytest tests/resilient_v2x/test_repair_pipeline.py -q`

Expected: FAIL because `CausalBranchRepair` is absent.

- [ ] **Step 3: Implement branch-wise short-circuiting**

The forward path must iterate only to gather per-sample supported indices, execute alignment/PTF for that subset, scatter outputs into neutral tensors, and compute:

```python
gamma = torch.pow(
    selected_feature.new_tensor(self.alpha),
    horizon.to(dtype=selected_feature.dtype),
)
normalized_age = torch.where(
    support,
    horizon.to(dtype=selected_feature.dtype) / self.history_limit,
    torch.zeros_like(gamma),
)
reliability = torch.where(
    observed,
    torch.ones_like(gamma),
    gamma * confidence.mean(dim=(-3, -2, -1)),
)
feature = gamma[:, None, None, None] * warp_with_displacement(
    aligned_feature,
    displacement,
)
```

Reject non-finite displacement, confidence, feature, gamma, reliability or
normalized age with a runtime error before diagnostics are committed.

`CausalBranchRepair` is the sole owner of age normalization and owns
`grid_spec`, `channels`, `alpha`, and the frozen `history_limit=3`; it must not construct,
store, clone, or register a PTF. The detector owns the two independent modality modules and
injects them explicitly:

```python
lidar_repair = self.branch_repair(
    selected_feature=lidar_selected,
    source_to_target=lidar_source_to_target,
    ptf_context=lidar_context,
    ptf=self.lidar_ptf,
    query_agent_index=lidar_agent_index,
    selections=lidar_selections,
)
camera_repair = self.branch_repair(
    selected_feature=camera_selected,
    source_to_target=camera_source_to_target,
    ptf_context=camera_context,
    ptf=self.camera_ptf,
    query_agent_index=camera_agent_index,
    selections=camera_selections,
)
assert self.lidar_ptf is not self.camera_ptf
```

`expected_output_shape()` validates positive batch size and returns
`(batch_size,channels,grid_spec.height,grid_spec.width)` without allocating a tensor.

- [ ] **Step 4: Run repair and preceding core tests**

Run: `python -m pytest tests/resilient_v2x/test_causal_repair.py tests/resilient_v2x/test_geometry.py tests/resilient_v2x/test_ptf.py tests/resilient_v2x/test_repair_pipeline.py -q`

Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add transvision/models/resilient_v2x/contracts.py transvision/models/resilient_v2x/causal_repair.py tests/resilient_v2x/test_repair_pipeline.py
git commit -m "feat: compose causal branch repair"
```

---

### Task 6: 实现双端聚合、三专家和 783-D DER

**Files:**

- Create: `transvision/models/resilient_v2x/routing.py`
- Create: `tests/resilient_v2x/test_routing.py`
- Modify: `transvision/models/resilient_v2x/__init__.py`

**Interfaces:**

```python
class DepthwiseSeparableResidualBlock(nn.Module):
    def __init__(self, channels: int) -> None
    def forward(self, value: Tensor) -> Tensor

@dataclass
class ModalityAggregate:
    feature: Tensor
    support: Tensor
    reliability: Tensor

class ModalityAggregator(nn.Module):
    def __init__(self, channels: int) -> None
    def forward(
        self,
        ego_feature: Tensor,
        rsu_feature: Tensor,
        ego_support: Tensor,
        rsu_support: Tensor,
        ego_reliability: Tensor,
        rsu_reliability: Tensor,
    ) -> ModalityAggregate

@dataclass
class RoutingOutput:
    fused: Tensor
    weights: Tensor
    descriptor: Tensor
    expert_support: Tensor
    lidar_expert: Tensor
    camera_expert: Tensor
    synergy_expert: Tensor
    overall_support: Tensor

class DynamicExpertRouter(nn.Module):
    descriptor_dim: int = 783

    def __init__(self, channels: int, hidden_channels: int) -> None
    def forward(
        self,
        lidar_feature: Tensor,
        camera_feature: Tensor,
        lidar_branch_support: Tensor,
        camera_branch_support: Tensor,
        branch_reliability: Tensor,
        branch_observed: Tensor,
        branch_propagated: Tensor,
        branch_normalized_age: Tensor,
        normalized_rsu_delay: Tensor,
        routing_mode: Literal["dynamic", "static", "uniform"],
        use_reliability: bool,
        use_delay_metadata: bool,
    ) -> RoutingOutput
```

Branch order is fixed as `[L_E,L_R,C_E,C_R]`; flag order is
`[L_E_obs,L_E_prop,L_R_obs,L_R_prop,C_E_obs,C_E_prop,C_R_obs,C_R_prop]`.

- [ ] **Step 1: Write descriptor, expert mask, and all-neutral tests**

```python
import torch

from transvision.models.resilient_v2x.routing import DynamicExpertRouter


def test_descriptor_is_exactly_783_and_weights_are_normalized() -> None:
    router = DynamicExpertRouter(channels=256, hidden_channels=256).eval()
    output = router(
        lidar_feature=torch.randn(2, 256, 8, 8),
        camera_feature=torch.randn(2, 256, 8, 8),
        lidar_branch_support=torch.tensor([[True, False], [True, True]]),
        camera_branch_support=torch.tensor([[True, True], [False, False]]),
        branch_reliability=torch.tensor([[1.0, 0.0, 0.8, 0.7],
                                         [1.0, 0.9, 0.0, 0.0]]),
        branch_observed=torch.tensor([[True, False, True, False],
                                      [True, False, False, False]]),
        branch_propagated=torch.tensor([[False, False, False, True],
                                        [False, True, False, False]]),
        branch_normalized_age=torch.tensor([[0.0, 0.0, 0.0, 1.0 / 3.0],
                                            [0.0, 2.0 / 3.0, 0.0, 0.0]]),
        normalized_rsu_delay=torch.tensor([[1.0 / 3.0], [2.0 / 3.0]]),
        routing_mode="dynamic",
        use_reliability=True,
        use_delay_metadata=True,
    )
    assert output.descriptor.shape == (2, 783)
    torch.testing.assert_close(output.weights.sum(dim=1), torch.ones(2))
    assert output.weights[1, 1].item() == 0.0
    assert output.weights[1, 2].item() == 0.0


def test_all_invalid_returns_exact_zero_without_nan() -> None:
    router = DynamicExpertRouter(channels=256, hidden_channels=256).eval()
    zeros = torch.zeros(1, 256, 8, 8)
    output = router(
        lidar_feature=zeros,
        camera_feature=zeros,
        lidar_branch_support=torch.zeros(1, 2, dtype=torch.bool),
        camera_branch_support=torch.zeros(1, 2, dtype=torch.bool),
        branch_reliability=torch.zeros(1, 4),
        branch_observed=torch.zeros(1, 4, dtype=torch.bool),
        branch_propagated=torch.zeros(1, 4, dtype=torch.bool),
        branch_normalized_age=torch.zeros(1, 4),
        normalized_rsu_delay=torch.zeros(1, 1),
        routing_mode="dynamic",
        use_reliability=True,
        use_delay_metadata=True,
    )
    assert torch.count_nonzero(output.weights).item() == 0
    assert torch.count_nonzero(output.fused).item() == 0
    assert not torch.isnan(output.weights).any()
    assert output.overall_support.tolist() == [False]
```

Add exact behavior tests for:

- descriptor layout is frozen as expert GAP `[0:768]`, modality reliabilities
  `[768:770]`, eight observed/propagated flags `[770:778]`, four branch
  normalized ages `[778:782]`, and normalized RSU delay `[782]`. Feeding horizons
  `0,1,2,3` through `CausalBranchRepair(history_limit=3)` produces exactly
  `(0,1/3,2/3,1)` in that age slice; an unsupported branch contributes zero;
- modality reliability is exactly `(eta_ego+eta_rsu)/2`, including one supported/one unsupported branch；
- synergy support requires both modality aggregates;
- every modality/synergy projection is exactly `Conv2d(..., kernel_size=1,
  bias=False)`; a support-zeroing test sets later GN affine/residual parameters
  to produce nonzero pre-mask activations and still proves an invalid expert is
  exactly zero before GAP;
- `uniform` assigns `1/n_valid`;
- `static` depends only on final gate bias;
- `use_reliability=False` replaces reliability by modality support indicators;
- `use_delay_metadata=False` zeros four ages and delay but keeps observed/propagated flags.

- [ ] **Step 2: Run routing tests and confirm failure**

Run: `python -m pytest tests/resilient_v2x/test_routing.py -q`

Expected: FAIL because `routing.py` is absent.

- [ ] **Step 3: Implement the fixed residual block and safe masked softmax**

Use this safe softmax instead of filling an all-invalid row with `-inf`:

```python
def masked_softmax(logits: Tensor, support: Tensor) -> Tensor:
    masked = logits.masked_fill(~support, torch.finfo(logits.dtype).min)
    any_valid = support.any(dim=1, keepdim=True)
    safe_logits = torch.where(any_valid, masked, torch.zeros_like(masked))
    weights = torch.softmax(safe_logits, dim=1)
    weights = weights * support.to(dtype=weights.dtype)
    denominator = weights.sum(dim=1, keepdim=True).clamp_min(1.0)
    return torch.where(any_valid, weights / denominator, torch.zeros_like(weights))
```

Implement:

- modality aggregator `Conv1x1 512→256, bias=False, GN32, SiLU` plus one depthwise-separable residual block, support=`ego_support|rsu_support`, reliability=`(eta_ego+eta_rsu)/2`;
- LiDAR/Camera experts with two blocks each;
- synergy stem `Conv1x1 512→256, bias=False, GN32, SiLU` plus two blocks;
- gate `LayerNorm783, Linear783→256, SiLU, Linear256→3`;
- concatenate the descriptor in the exact slice order above and accept only
  `branch_normalized_age` already produced by `CausalBranchRepair`; the router
  never divides raw horizons a second time and rejects values outside `[0,1]`;
- feature zeroing before GAP and again before weighted sum;
- finite/range assertions on descriptor and weights.

All 256-channel depthwise-separable blocks use a `bias=False` 3x3 depthwise
convolution and a `bias=False` 1x1 pointwise convolution because each is followed
by GN. Both gate linears use `bias=True`. Structural/parameter-count golden tests
freeze one `ModalityAggregator(256)` at `200_448` trainable parameters and
`DynamicExpertRouter(256,256)` at `747_809`; changing a bias choice fails before
capacity-matched concat width is solved.

- [ ] **Step 4: Run routing tests**

Run: `python -m pytest tests/resilient_v2x/test_routing.py -q`

Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add transvision/models/resilient_v2x/routing.py transvision/models/resilient_v2x/__init__.py tests/resilient_v2x/test_routing.py
git commit -m "feat: implement dynamic expert routing"
```

---

### Task 7: 实现 frozen teacher 与 Bernoulli 蒸馏损失

**Files:**

- Create: `transvision/models/resilient_v2x/distillation.py`
- Create: `tests/resilient_v2x/test_distillation.py`
- Modify: `transvision/models/resilient_v2x/__init__.py`

**Interfaces:**

```python
@dataclass(frozen=True)
class DistillationLosses:
    feature: Tensor
    bernoulli: Tensor
    total: Tensor

def freeze_teacher(teacher: nn.Module) -> nn.Module

def assert_teacher_frozen(teacher: nn.Module) -> None

def bernoulli_kl_from_logits(
    teacher_logits: Tensor,
    student_logits: Tensor,
    temperature: float,
    epsilon: float,
) -> Tensor

def distillation_losses(
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

- [ ] **Step 1: Write numerical and gradient tests**

```python
import torch
from torch import nn

from transvision.models.resilient_v2x.distillation import (
    assert_teacher_frozen,
    bernoulli_kl_from_logits,
    distillation_losses,
    freeze_teacher,
)


def test_equal_logits_have_zero_bernoulli_kl() -> None:
    logits = torch.tensor([[-3.0, 0.0, 4.0]], dtype=torch.float64)
    loss = bernoulli_kl_from_logits(logits, logits, temperature=4.0, epsilon=1e-6)
    torch.testing.assert_close(loss, torch.zeros_like(loss), atol=1e-14, rtol=0.0)


def test_student_receives_gradient_and_teacher_does_not() -> None:
    teacher_feature = torch.randn(2, 4, 3, 3, requires_grad=True)
    student_feature = torch.randn(2, 4, 3, 3, requires_grad=True)
    teacher_logits = torch.randn(2, 1, 3, 3, requires_grad=True)
    student_logits = torch.randn(2, 1, 3, 3, requires_grad=True)
    losses = distillation_losses(
        teacher_feature=teacher_feature,
        student_feature=student_feature,
        teacher_logits=teacher_logits,
        student_logits=student_logits,
        temperature=4.0,
        lambda_feature=1.0,
        lambda_logit=1.0,
        valid_sample_mask=torch.tensor([True, False]),
    )
    losses.total.backward()
    assert teacher_feature.grad is None
    assert teacher_logits.grad is None
    assert student_feature.grad is not None
    assert student_logits.grad is not None


def test_freeze_teacher_sets_eval_and_disables_gradients() -> None:
    teacher = freeze_teacher(nn.Sequential(nn.Linear(4, 4), nn.BatchNorm1d(4)))
    assert teacher.training is False
    assert all(parameter.requires_grad is False for parameter in teacher.parameters())
    assert_teacher_frozen(teacher)
```

Add tests for extreme logits `±1000`, valid-mask empty error, shape mismatch, `T<=0`, and exact `T^2` multiplier in total loss.

- [ ] **Step 2: Run distillation tests and confirm failure**

Run: `python -m pytest tests/resilient_v2x/test_distillation.py -q`

Expected: FAIL because `distillation.py` is absent.

- [ ] **Step 3: Implement stop-gradient and stable Bernoulli KL**

Use the exact formula:

```python
def bernoulli_kl_from_logits(
    teacher_logits: Tensor,
    student_logits: Tensor,
    temperature: float,
    epsilon: float,
) -> Tensor:
    if teacher_logits.shape != student_logits.shape:
        raise ValueError("teacher and student logits must have identical shapes")
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    teacher_probability = torch.sigmoid(
        teacher_logits.detach().to(torch.float32) / temperature
    ).clamp(epsilon, 1.0 - epsilon)
    student_probability = torch.sigmoid(
        student_logits.to(torch.float32) / temperature
    ).clamp(epsilon, 1.0 - epsilon)
    positive = teacher_probability * (
        teacher_probability.log() - student_probability.log()
    )
    negative = (1.0 - teacher_probability) * (
        (1.0 - teacher_probability).log()
        - (1.0 - student_probability).log()
    )
    return (positive + negative).mean()
```

Feature loss uses `torch.nn.functional.mse_loss(..., reduction="none")`, averages `C*H*W` per sample, applies `valid_sample_mask`, then averages valid samples. Logit loss applies the same sample mask before flattening. Empty valid mask must raise `RuntimeError("no valid samples for distillation")`.

- [ ] **Step 4: Run distillation tests**

Run: `python -m pytest tests/resilient_v2x/test_distillation.py -q`

Expected: all tests pass with finite losses and gradients.

- [ ] **Step 5: Commit**

```bash
git add transvision/models/resilient_v2x/distillation.py transvision/models/resilient_v2x/__init__.py tests/resilient_v2x/test_distillation.py
git commit -m "feat: add resilient v2x distillation losses"
```

---

### Task 8: 核心包总回归和 public API 冻结

**Files:**

- Modify: `transvision/models/resilient_v2x/contracts.py`
- Modify: `transvision/models/resilient_v2x/__init__.py`
- Modify: `tests/resilient_v2x/test_contracts.py`
- Create: `docs/resilient_v2x/core-api.md`

**Interfaces:**

No new implementation symbol is introduced. This task freezes the exact
`transvision.models.resilient_v2x.__all__` tuple exercised below.

- [ ] **Step 1: Add a public API import test**

```python
CORE_PUBLIC_API = (
    "Agent",
    "Modality",
    "UnsupportedReason",
    "ProtocolInvariantError",
    "SourceCandidate",
    "RejectedCandidate",
    "BranchSelection",
    "BEVGridSpec",
    "compose_source_to_target",
    "build_backward_grid",
    "align_bev_to_target",
    "warp_with_displacement",
    "arrived_candidates",
    "select_causal_source",
    "age_decay",
    "branch_reliability",
    "normalized_selected_rsu_delay",
    "PTFOutput",
    "HorizonConditionedPTF",
    "BranchDiagnostics",
    "RepairedBranch",
    "CausalBranchRepair",
    "DepthwiseSeparableResidualBlock",
    "ModalityAggregate",
    "ModalityAggregator",
    "RoutingOutput",
    "DynamicExpertRouter",
    "RoutingDiagnostics",
    "SampleInferenceDiagnostics",
    "DistillationLosses",
    "freeze_teacher",
    "assert_teacher_frozen",
    "bernoulli_kl_from_logits",
    "distillation_losses",
    "ResilientFeatureBatch",
)


def test_public_core_api_imports() -> None:
    import transvision.models.resilient_v2x as core

    assert len(core.__all__) == len(set(core.__all__))
    missing = [name for name in CORE_PUBLIC_API if name not in core.__all__]
    assert missing == []
    for name in CORE_PUBLIC_API:
        assert getattr(core, name) is not None

    assert core.Agent.EGO.value == "ego"
    assert core.Modality.CAMERA.value == "camera"
    assert core.UnsupportedReason.UNSUPPORTED_HORIZON.value == (
        "unsupported_horizon"
    )
    assert core.DynamicExpertRouter.descriptor_dim == 783
    assert core.ResilientFeatureBatch.__name__ == "ResilientFeatureBatch"
    assert callable(core.compose_source_to_target)
    assert callable(core.distillation_losses)
```

- [ ] **Step 2: Run the new import test and confirm the export failure**

Run: `python -m pytest tests/resilient_v2x/test_contracts.py::test_public_core_api_imports -q`

Expected: FAIL because the final public symbols have not yet all been exported from `__init__.py`.

- [ ] **Step 3: Freeze exports and document tensor/error contracts**

Update `__init__.py` with explicit imports and an `__all__` containing every symbol in
`CORE_PUBLIC_API`; do not use wildcard imports. Preserve this complete core subset when later
plans append encoder/training APIs, so the core test remains valid after the package grows.
Implement `ResilientFeatureBatch.__post_init__()` to validate matching batch dimensions,
fused `[B,256,Y,X]`, support `[B]`, weights `[B,3]`, descriptor `[B,783]`, and one
diagnostics record per sample. Add one valid small-tensor fixture and one batch-mismatch
fixture to `test_contracts.py`.

```python
@dataclass(frozen=True)
class RoutingDiagnostics:
    descriptor: Tensor | None
    expert_support: Tensor | None
    weights: Tensor | None
    not_applicable_reason: str | None

@dataclass(frozen=True)
class SampleInferenceDiagnostics:
    schema_version: Literal[1]
    sample_id: str
    method: str
    branches: tuple[
        BranchDiagnostics,
        BranchDiagnostics,
        BranchDiagnostics,
        BranchDiagnostics,
    ]
    routing: RoutingDiagnostics

@dataclass
class ResilientFeatureBatch:
    fused: Tensor
    overall_support: Tensor
    routing_weights: Tensor
    routing_descriptor: Tensor
    branch_features: Mapping[str, Tensor]
    diagnostics: tuple[SampleInferenceDiagnostics, ...]

    def __post_init__(self) -> None:
        if self.fused.ndim != 4 or self.fused.shape[1] != 256:
            raise ProtocolInvariantError("fused feature must be [B,256,Y,X]")
        batch = self.fused.shape[0]
        if self.overall_support.shape != (batch,):
            raise ProtocolInvariantError("overall support must be [B]")
        if self.routing_weights.shape != (batch, 3):
            raise ProtocolInvariantError("routing weights must be [B,3]")
        if self.routing_descriptor.shape != (batch, 783):
            raise ProtocolInvariantError("routing descriptor must be [B,783]")
        if len(self.diagnostics) != batch:
            raise ProtocolInvariantError("diagnostics count must equal batch size")
        if any(value.shape[0] != batch for value in self.branch_features.values()):
            raise ProtocolInvariantError("branch feature batch mismatch")
```

`SampleInferenceDiagnostics.branches` is always ordered
`[L_E,L_R,C_E,C_R]`. Each supported Resilient routing record has detached
per-sample descriptor `[783]`, expert support `[3]` and weights `[3]`; an adapter
without DER sets all three to `None` plus a non-empty
`not_applicable_reason`. The validator rejects mixed present/missing routing
fields, non-finite values, wrong shapes/order/sample IDs or tensor values still
requiring gradients. A method without a modality still emits that branch with
`supported=False`, all source/tensor fields null and
`reason=METHOD_NOT_APPLICABLE`; a method without PTF/DER uses explicit null
algorithm fields and a non-empty routing reason rather than fabricating zeros.
The evidence plan owns durable serialization; this core type
owns the complete in-memory contract.

`core-api.md` must include:

- all public symbols and signatures;
- `[B,C,Y,X]` and displacement channel conventions;
- reason precedence;
- observed/propagated/reliability rules;
- PTF `h=0`, bounds, nonlinear/linear behavior;
- descriptor field order and 783-D arithmetic;
- which errors are protocol, unsupported, or runtime failures.

- [ ] **Step 4: Run the complete core suite**

Run:

```bash
python -m pytest tests/resilient_v2x/test_contracts.py tests/resilient_v2x/test_geometry.py tests/resilient_v2x/test_causal_repair.py tests/resilient_v2x/test_ptf.py tests/resilient_v2x/test_repair_pipeline.py tests/resilient_v2x/test_routing.py tests/resilient_v2x/test_distillation.py -q
python -m compileall -q transvision/models/resilient_v2x tests/resilient_v2x
git diff --check
```

Expected: every test passes, `compileall` and `git diff --check` produce no output.

- [ ] **Step 5: Verify no placeholders or hidden protocol defaults**

Run:

```bash
rg -n 'TODO|TBD|NotImplemented|pass$' transvision/models/resilient_v2x tests/resilient_v2x docs/resilient_v2x/core-api.md
rg -n 'history_limit: int =|delta_t_ms: int =|alpha: float =' transvision/models/resilient_v2x
```

Expected: both commands return no matches.

- [ ] **Step 6: Commit**

```bash
git add transvision/models/resilient_v2x/contracts.py transvision/models/resilient_v2x/__init__.py tests/resilient_v2x/test_contracts.py docs/resilient_v2x/core-api.md
git commit -m "docs: freeze resilient v2x core api"
```

## Completion Gate

This plan is complete only when:

- every listed CPU test passes in the pinned environment;
- unsupported branches never call PTF and serialize unavailable diagnostics as null;
- identity geometry is exact and all coordinate conventions are documented;
- nonlinear/linear PTF, total-age decay, 783-D descriptor, all-invalid routing, and Bernoulli KL are covered by tests;
- `git status --short` contains no unplanned files from this phase.
