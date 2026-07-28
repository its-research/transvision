# ResilientV2X 受控复现手册

本文档描述当前仓库能够执行和留证的复现流程。它不把论文中未公开的数值补成“论文默认值”，也不把单元测试、fixture、未训练权重或开发环境输出当作论文结果。所有命令都从仓库根目录运行；除构建容器的命令外，正式实验应在锁定的受控容器内执行。

## 1. 复现边界

当前主路径是 DAIR-V2X cooperative vehicle-infrastructure、一个 ego 与一个 RSU、LiDAR 与 camera、Car 检测。固定实现剖面为 `delta_t_ms=100`、`history_limit=3`，评测主表为延迟 0/100/200/300 ms 与 Full/L-Fail/C-Fail 的 12 个 E+R 条件。

以下内容不能从当前仓库声称：

- 论文表格中的任何数值已经复现；
- 与论文作者未发布参考实现逐位一致；
- 未经受控环境、官方 split、真实 checkpoint 和哈希证据支持的 AP、FLOPs、显存或时延；
- V2XSet 数值结果。Pair 只有严格序列级离线清单适配器；Standard 只有真实 source-loader trace 导出边界，且当前单 Ego/单 RSU 方法在 Standard 多智能体轨道固定为 N/A。详见 [论文覆盖矩阵](paper-coverage.md)、[Pair 边界](../../tools/resilient_v2x/V2XSET_PAIR_README.md) 和 [Standard 边界](../../tools/resilient_v2x/V2XSET_STANDARD_README.md)。

## 2. 锁定环境与环境证据

环境定义位于：

- `environments/resilient_v2x/environment.yml`
- `environments/resilient_v2x/environment-linux-64.lock.yml`
- `environments/resilient_v2x/bootstrap-linux-64.explicit.txt`
- `environments/resilient_v2x/Dockerfile`

构建并检查镜像：

```bash
docker build \
  -f environments/resilient_v2x/Dockerfile \
  -t resilient-v2x:locked \
  .

docker run --rm --gpus all resilient-v2x:locked
```

镜像默认命令等价于受控检查。进入实验环境后，也可显式运行：

```bash
python tools/resilient_v2x/check_environment.py \
  --mode controlled \
  --world-size 1 \
  --require-cuda \
  --require-custom-ops
```

只有输出中 `accepted` 为 `true` 的 controlled 检查才能支持正式复现声明。`--mode development` 只用于诊断；其中的 `not_executed_checks` 不能被解释为已验证。

环境 manifest 必须写在明确给定且已存在的绝对 evidence root 内：

```bash
export EVIDENCE_ROOT="$(pwd)/artifacts/resilient_v2x/evidence"
mkdir -p "$EVIDENCE_ROOT"

python tools/resilient_v2x/capture_environment.py \
  --world-size 1 \
  --classification controlled \
  --evidence-root "$EVIDENCE_ROOT" \
  --output "$EVIDENCE_ROOT/environment.json"
```

该命令同时写出 `environment.json.sha256`，并拒绝覆盖内容不同的既有 manifest。

## 3. 准备 DAIR-V2X 时间清单

官方 cooperative split 的固定 SHA-256 是：

```text
d048aeeca548fb194c548b798e6fc08488c4dd350ad223a154c028ed0a58de6c
```

设置路径并创建产物目录：

```bash
export DAIR_ROOT="$(pwd)/data/DAIR-V2X/cooperative-vehicle-infrastructure"
export DAIR_SPLIT="$(pwd)/data/split_datas/cooperative-split-data.json"
export SPLIT_SHA256="d048aeeca548fb194c548b798e6fc08488c4dd350ad223a154c028ed0a58de6c"
export DAIR_ARTIFACT_ROOT="$(pwd)/artifacts/resilient_v2x/dair"
mkdir -p "$DAIR_ARTIFACT_ROOT"
```

执行受控准备：

```bash
python tools/resilient_v2x/prepare_data.py \
  --data-root "$DAIR_ROOT" \
  --split-file "$DAIR_SPLIT" \
  --expected-split-sha256 "$SPLIT_SHA256" \
  --protocol-scope controlled \
  --output "$DAIR_ARTIFACT_ROOT/temporal_manifest.json" \
  --delta-t-ms 100 \
  --history-limit 3 \
  --interval-min-ms 50 \
  --interval-max-ms 150 \
  --max-capture-skew-ms 200
```

`controlled` 模式会拒绝其他 split 哈希或协议数值。公开的带标注 cooperative release 只覆盖官方 train/validation，因此 manifest 不要求未发布 GT 的 test 样本；论文结果也只从 validation 报告。命令验证原始 release inventory、标定、时间戳和文件哈希，将验证后的 PCD 转成 `prepared/resilient_v2x/.../*.bin`，并发布自哈希的时间 manifest。`--protocol-scope fixture` 只供测试，不能产生论文证据。

split 归属按 vehicle frame ID 查询；manifest 的 pair 身份使用
`dairc-v{vehicle_frame_id}-i{infrastructure_frame_id}`。当前官方元数据中同一
vehicle target 的 16 个重复配对变体全部保留，并按 LiDAR 时间差、RSU 时间
和 RSU ID 确定性分入相互隔离的 sequence lane，不做静默去重。

这里的 200 ms 是官方 multimodal pair 的四路 capture-time compatibility envelope，不是同步精度，也不计入通信延迟。`n×100 ms` 是 transport/fault 使用的逻辑协议网格；四路真实 `capture_timestamp_us` 独立保留。官方 release 中 Ego camera 相对 Ego LiDAR 存在约一帧的固定 phase，因此不能用 50 ms all-four skew 拒绝样本。

官方 infrastructure `virtuallidar_to_camera` 是可逆 affine 标定而非严格刚体
旋转。manifest 原样保留其数值供 LSS 投影使用，并强制有限、齐次、正定向及
无穷范数条件数不超过 4；agent/world 位姿和 LiDAR 外参仍必须是刚体，不能把 affine 松绑
扩散到时空对齐路径。

## 4. 生成 cohort、训练 overlay 与评测 overlay

### 4.1 冻结共享评测 cohort

先为 duration-1 评测矩阵在最坏 300 ms 延迟下冻结主共享 cohort。被排除样本及原因会进入 cohort 文档：

```bash
python tools/resilient_v2x/build_overlays.py cohort \
  "$DAIR_ARTIFACT_ROOT/temporal_manifest.json" \
  --expected-split-sha256 "$SPLIT_SHA256" \
  --split val \
  --max-delay-ms 300 \
  --max-duration 1 \
  --out "$DAIR_ARTIFACT_ROOT/validation_cohort.json"
```

下面生成的 36 个 duration-1 `(delay, condition, agent_scope)` 组合必须复用同一个 `validation_cohort.json` 和同一个 `sample_ids_sha256`；不能按条件重新筛样本。论文主表是其中 `agent_scope=E+R` 的 12 个组合。

### 4.2 训练 overlay

论文没有给出可核验的训练延迟采样 seed、LiDAR 故障概率或 camera 故障概率。下面的 `20250218`、`0.2`、`0.2` 是明确的 implementation choices，最终值会记录在 `training_overlays.json`，不是论文默认值：

```bash
python tools/resilient_v2x/build_overlays.py train \
  "$DAIR_ARTIFACT_ROOT/temporal_manifest.json" \
  --expected-split-sha256 "$SPLIT_SHA256" \
  --protocol-seed 20250218 \
  --epochs $(python -c 'print(*range(50))') \
  --p-lidar 0.2 \
  --p-camera 0.2 \
  --out-dir "$DAIR_ARTIFACT_ROOT"
```

产物包括 `train_transport.jsonl.zst`、`train_fault.jsonl.zst` 和 `training_overlays.json`。训练 overlay 按 epoch 和 sample ID 确定性生成；索引同时记录压缩与未压缩摘要。运行时配置使用索引中的 `uncompressed_sha256`。

### 4.3 12 个主条件

主表和 duration-1 agent-scope 诊断均由同一主 cohort 构建。下面命令生成 4 个 transport overlay，以及所有延迟、Full/L-Fail/C-Fail、E+R/E-only/R-only 的 duration-1 causal fault overlay：

```bash
python tools/resilient_v2x/build_overlays.py evaluation \
  "$DAIR_ARTIFACT_ROOT/temporal_manifest.json" \
  --expected-split-sha256 "$SPLIT_SHA256" \
  --cohort "$DAIR_ARTIFACT_ROOT/validation_cohort.json" \
  --delays 0 100 200 300 \
  --conditions Full L-Fail C-Fail \
  --agents E+R E-only R-only \
  --duration 1 \
  --out-dir "$DAIR_ARTIFACT_ROOT"
```

论文主表是该索引中 `agent_scope=E+R` 的 12 个 `(delay, condition)` 组合。故障顺序是：先按 transport overlay 为 ego/RSU 各自确定最新因果可用端点，再从该端点向历史方向掩码。非零延迟时，不能用统一的目标 tick mask 代替该过程。

运行时的 observed/propagated flag 也相对 branch endpoint 定义：最新已到达
RSU endpoint 的模态可用时，即使它相对 ego 决策时刻已经延迟，仍是
observed；只有 endpoint 缺失并回退到更早 source 时才是 propagated。该
observed RSU 仍使用 `gamma × trajectory confidence`，只有真正当前 Ego
观测的 reliability 固定为 1。

对应配置如下：

| 延迟 | Full | L-Fail | C-Fail |
|---:|---|---|---|
| 0 | `conditions/global_delay_000_full.py` | `conditions/causal_delay_000_l_fail.py` | `conditions/causal_delay_000_c_fail.py` |
| 100 | `conditions/global_delay_100_full.py` | `conditions/causal_delay_100_l_fail.py` | `conditions/causal_delay_100_c_fail.py` |
| 200 | `conditions/global_delay_200_full.py` | `conditions/causal_delay_200_l_fail.py` | `conditions/causal_delay_200_c_fail.py` |
| 300 | `conditions/global_delay_300_full.py` | `conditions/causal_delay_300_l_fail.py` | `conditions/causal_delay_300_c_fail.py` |

### 4.4 因果持续故障

论文没有规定持续故障与非零时延的笛卡尔积，本实现不补造这些组合。cohort 的联合历史契约是 `max_delay_ms / delta_t_ms + duration - 1 <= history_limit`；在当前 `delta_t_ms=100`、`history_limit=3` 下，300 ms 延迟已经消耗 3 个历史间隔，无法再容纳 duration 2 至 4。持续故障诊断因此单独冻结一个 0 ms、最长 4 tick 的共享 cohort：

```bash
python tools/resilient_v2x/build_overlays.py cohort \
  "$DAIR_ARTIFACT_ROOT/temporal_manifest.json" \
  --expected-split-sha256 "$SPLIT_SHA256" \
  --split val \
  --max-delay-ms 0 \
  --max-duration 4 \
  --out "$DAIR_ARTIFACT_ROOT/validation_duration_cohort.json"
```

持续 2、3、4 tick 的诊断复用该 cohort，但必须写到不同目录，因为每个目录的 `evaluation_overlays.json` 是不可变索引：

```bash
for duration in 2 3 4; do
  python tools/resilient_v2x/build_overlays.py evaluation \
    "$DAIR_ARTIFACT_ROOT/temporal_manifest.json" \
    --expected-split-sha256 "$SPLIT_SHA256" \
    --cohort "$DAIR_ARTIFACT_ROOT/validation_duration_cohort.json" \
    --delays 0 \
    --conditions L-Fail C-Fail \
    --agents E+R E-only R-only \
    --duration "$duration" \
    --out-dir "$DAIR_ARTIFACT_ROOT/continuous_d${duration}"
done
```

`conditions/causal_fault_diagnostic.py` 通过以下环境变量选择一个已生成条件：

- `RESILIENT_V2X_DIAGNOSTIC_DELAY_MS`
- `RESILIENT_V2X_DIAGNOSTIC_SCOPE`：`E+R`、`E-only` 或 `R-only`
- `RESILIENT_V2X_DIAGNOSTIC_MODALITY`：`lidar` 或 `camera`
- `RESILIENT_V2X_DIAGNOSTIC_DURATION_TICKS`：1 至 4
- 对应 transport/fault overlay 路径与未压缩 SHA-256

## 5. 将哈希绑定到运行时配置

通用数据环境：

```bash
export RESILIENT_V2X_DATA_ROOT="$DAIR_ROOT"
export RESILIENT_V2X_MANIFEST="$DAIR_ARTIFACT_ROOT/temporal_manifest.json"
export RESILIENT_V2X_SPLIT_SHA256="$SPLIT_SHA256"
```

训练 overlay 必须成对提供路径和摘要：

```bash
export RESILIENT_V2X_TRAIN_TRANSPORT_OVERLAY="$DAIR_ARTIFACT_ROOT/train_transport.jsonl.zst"
export RESILIENT_V2X_TRAIN_FAULT_OVERLAY="$DAIR_ARTIFACT_ROOT/train_fault.jsonl.zst"

export RESILIENT_V2X_TRAIN_TRANSPORT_SHA256="$(python -c 'import json,sys; d=json.load(open(sys.argv[1])); print(d["overlays"]["transport"]["uncompressed_sha256"])' "$DAIR_ARTIFACT_ROOT/training_overlays.json")"
export RESILIENT_V2X_TRAIN_FAULT_SHA256="$(python -c 'import json,sys; d=json.load(open(sys.argv[1])); print(d["overlays"]["fault"]["uncompressed_sha256"])' "$DAIR_ARTIFACT_ROOT/training_overlays.json")"
```

若只设置路径或只设置摘要，数据集会失败；不要使用全零或随意填写的摘要绕过检查。

主评测条件的环境变量可从索引确定。下例为 300 ms L-Fail：

0 ms Full 也必须绑定 `val_transport_delay_000.jsonl.zst` 及其未压缩摘要；
该零时延 overlay 不改变到达时刻，而是让运行时严格限制在同一个共享 cohort，
避免 0 ms 条件意外评测全量 val split。

```bash
export RESILIENT_V2X_TEST_TRANSPORT_DELAY_300_OVERLAY="$DAIR_ARTIFACT_ROOT/val_transport_delay_300.jsonl.zst"
export RESILIENT_V2X_TEST_CAUSAL_DELAY_300_L_FAIL_OVERLAY="$DAIR_ARTIFACT_ROOT/val_causal_delay_300_l_fail.jsonl.zst"

export RESILIENT_V2X_TEST_TRANSPORT_DELAY_300_SHA256="$(python -c 'import json,sys; d=json.load(open(sys.argv[1])); print(next(x["overlay"]["uncompressed_sha256"] for x in d["transport_overlays"] if x["delay_ms"]==300))' "$DAIR_ARTIFACT_ROOT/evaluation_overlays.json")"
export RESILIENT_V2X_TEST_CAUSAL_DELAY_300_L_FAIL_SHA256="$(python -c 'import json,sys; d=json.load(open(sys.argv[1])); print(next(x["overlay"]["uncompressed_sha256"] for x in d["fault_overlays"] if x["delay_ms"]==300 and x["condition"]=="L-Fail" and x["agent_scope"]=="E+R" and x["duration"]==1))' "$DAIR_ARTIFACT_ROOT/evaluation_overlays.json")"
```

其他条件使用各配置文件 `required_external_inputs` 中列出的同名变量；摘要仍从 `evaluation_overlays.json` 读取。

`tools/train.py` 和 `tools/test.py` 由子目录脚本启动，不会自行把仓库根目录加入模块搜索路径。进入两阶段训练前必须从仓库根目录显式绑定该路径；受控容器内对应路径是 `/workspace/transvision`：

```bash
export RESILIENT_V2X_REPO_ROOT="$(pwd)"
export PYTHONPATH="$RESILIENT_V2X_REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
```

## 6. 两阶段训练

### 6.1 Clean teacher

```bash
python tools/train.py \
  configs/resilient_v2x/dair_clean_teacher.py \
  --work-dir work_dirs/resilient_v2x_dair_clean_teacher
```

clean teacher 配置显式关闭 teacher、distillation 和训练 overlay，训练输入为零延迟、无注入故障。根据预先声明的 checkpoint 选择规则选择一个真实 checkpoint，并记录摘要：

```bash
export RESILIENT_V2X_TEACHER_CHECKPOINT="$(pwd)/work_dirs/resilient_v2x_dair_clean_teacher/teacher_epoch_50.pth"
sha256sum "$RESILIENT_V2X_TEACHER_CHECKPOINT"
```

不要把不存在的 `best_teacher.pth` 当成已生成文件；若选择 best checkpoint，应使用 CheckpointHook 实际写出的文件名。

### 6.2 Distilled student

```bash
python tools/train.py \
  configs/resilient_v2x/dair_resilient_v2x.py \
  --work-dir work_dirs/resilient_v2x_dair_student
```

模型会以 strict checkpoint load 构建冻结 clean teacher；训练 batch 同时包含受退化 student 输入和由同一 sample 生成的 clean teacher 输入。训练完成后选择并哈希实际 student checkpoint：

```bash
export STUDENT_CHECKPOINT="$(pwd)/work_dirs/resilient_v2x_dair_student/epoch_50.pth"
sha256sum "$STUDENT_CHECKPOINT"
```

路径只是按当前 `filename_tmpl` 的例子；应以真实工作目录中的文件为准。checkpoint 尚未生成时不能继续评测或填写结果。

## 7. 运行评测与保存 prediction evidence

先选择一个条件配置并确保其 overlay 环境变量已经绑定。预测证据由同一次 `ResilientV2XMetric` 评测写出，不进行第二次推理：

```bash
export RUN_ID="dair-test-delay-300-l-fail"
export RUN_DIR="$(pwd)/work_dirs/evaluation/$RUN_ID"
mkdir -p "$RUN_DIR"
export RESILIENT_V2X_PREDICTION_OUTPUT="$RUN_DIR/predictions.json"

python tools/test.py \
  configs/resilient_v2x/conditions/causal_delay_300_l_fail.py \
  "$STUDENT_CHECKPOINT" \
  --work-dir "$RUN_DIR"
```

`predictions.json` 是 canonical、自哈希文档，包含 ego LiDAR bottom-center box、score、label、GT 和分支/路由诊断。只有设置 `RESILIENT_V2X_PREDICTION_OUTPUT` 后由这次评测实际生成的文件，才能作为 `--predictions` 证据输入。

MMEngine 的 LocalVisBackend 将标量写入 `scalars.json` JSONL。下面命令找到本次 run 最新文件，并仅提取实际记录的 ResilientV2X 指标为 canonical JSON；找不到指标时会失败，而不是制造空结果：

```bash
export SCALARS_JSON="$(python -c 'from pathlib import Path; import sys; files=list(Path(sys.argv[1]).rglob("scalars.json")); assert files, "no scalars.json"; print(max(files,key=lambda p:p.stat().st_mtime))' "$RUN_DIR")"

python - "$SCALARS_JSON" "$RUN_DIR/metrics.json" <<'PY'
import json
import sys
from pathlib import Path

source = Path(sys.argv[1])
destination = Path(sys.argv[2])
rows = [json.loads(line) for line in source.read_text().splitlines() if line]
matches = [
    {key: value for key, value in row.items() if "resilient_v2x/" in key}
    for row in rows
]
matches = [row for row in matches if row]
if not matches:
    raise SystemExit("no ResilientV2X metrics in scalars.json")
destination.write_text(
    json.dumps({"metrics": matches[-1]}, sort_keys=True, separators=(",", ":")),
    encoding="utf-8",
)
PY
```

12 个主条件应分别使用表中配置、独立 `RUN_ID`、独立 prediction/metrics 文件，并复用完全相同的 cohort 与 student checkpoint。

## 8. 复杂度、显存和同步时延

`profile.py` 统计 student-only 参数，默认排除 training-only teacher；它对一个已 collate 的 test batch 测量 `model.test_step`，排除 dataloader I/O，并在计时边界进行 CUDA 同步：

```bash
python tools/resilient_v2x/profile.py \
  configs/resilient_v2x/conditions/causal_delay_300_l_fail.py \
  "$STUDENT_CHECKPOINT" \
  --out "$RUN_DIR/profile.json" \
  --device cuda:0 \
  --warmup 10 \
  --iterations 100 \
  --strict-checkpoint \
  --artifact environment_manifest="$EVIDENCE_ROOT/environment.json" \
  --artifact temporal_manifest="$DAIR_ARTIFACT_ROOT/temporal_manifest.json" \
  --artifact evaluation_overlays="$DAIR_ARTIFACT_ROOT/evaluation_overlays.json" \
  --artifact evaluation_cohort="$DAIR_ARTIFACT_ROOT/validation_cohort.json" \
  --artifact transport_overlay="$RESILIENT_V2X_TEST_TRANSPORT_DELAY_300_OVERLAY" \
  --artifact fault_overlay="$RESILIENT_V2X_TEST_CAUSAL_DELAY_300_L_FAIL_OVERLAY"
```

输出记录参数/缓冲区、FLOPs 支持状态、peak allocated/reserved GPU memory、每次同步 latency 样本、mean/median/p90/min/max、设备信息、输入 artifact 哈希和源码 Git 状态。若 profiler 不能支持某个算子，FLOPs 必须保持 `unavailable` 并附原因；不能猜测。`--skip-flops` 只用于显式声明不测 FLOPs。

该边界必须报告为 model-path latency，而不是包含 sensing ingestion 与
dataloader I/O 的最宽 end-to-end latency。`torch.profiler` 也只累计它能
识别的算子；自定义 CUDA op 等未覆盖算子不能被默认为零 FLOPs。

容量匹配 concat 比较使用 `configs/resilient_v2x/ablations/concat_capacity_matched.py`，并必须复用相同 profile 边界、batch、硬件和迭代数。

## 9. 构建最终 evidence bundle

evidence builder 不加载模型、不推理、不重算指标；它只验证并绑定已存在文件：

```bash
python tools/resilient_v2x/build_evidence.py \
  --run-id "$RUN_ID" \
  --metrics "$RUN_DIR/metrics.json" \
  --metrics-key metrics \
  --profile "$RUN_DIR/profile.json" \
  --predictions "$RUN_DIR/predictions.json" \
  --conditions "$DAIR_ARTIFACT_ROOT/evaluation_overlays.json" \
  --artifact environment_manifest="$EVIDENCE_ROOT/environment.json" \
  --artifact temporal_manifest="$DAIR_ARTIFACT_ROOT/temporal_manifest.json" \
  --artifact cohort="$DAIR_ARTIFACT_ROOT/validation_cohort.json" \
  --artifact evaluation_overlays="$DAIR_ARTIFACT_ROOT/evaluation_overlays.json" \
  --artifact transport_overlay="$RESILIENT_V2X_TEST_TRANSPORT_DELAY_300_OVERLAY" \
  --artifact fault_overlay="$RESILIENT_V2X_TEST_CAUSAL_DELAY_300_L_FAIL_OVERLAY" \
  --artifact condition_config=configs/resilient_v2x/conditions/causal_delay_300_l_fail.py \
  --artifact student_checkpoint="$STUDENT_CHECKPOINT" \
  --artifact runner_scalars="$SCALARS_JSON" \
  --out "$RUN_DIR/evidence.json"
```

最终文档包含 `content_sha256`，并明确 `inference_performed_by_evidence_builder=false`、`metric_recomputation_performed_by_evidence_builder=false`。任何源 artifact、指标或嵌套字段变动都会导致验证失败。

## 10. 哈希规则与禁止伪结果

- split 使用固定官方 SHA-256；不接受“相同文件名”作为身份。
- temporal manifest、cohort、overlay index、prediction、profile 与 evidence 都有 canonical content hash。
- `.jsonl.zst` 在索引中同时记录压缩与未压缩摘要；runtime 使用 `uncompressed_sha256` 解码验证。
- 环境 manifest 还有 detached `.sha256` sidecar。
- checkpoint、配置和额外 artifact 由 profile/evidence 逐文件哈希。
- immutable publisher 对相同内容幂等，对同路径不同内容拒绝覆盖。
- 禁止使用全零摘要、手工填 AP/时延/FLOPs、随机 teacher、重复推理后把预测与原指标拼接、或用 fixture 结果代表正式数据。
- 在 controlled 环境检查、官方数据准备、两阶段真实 checkpoint、12 条件推理和 evidence bundle 全部完成前，只能声明“实现/测试通过”，不能声明“复现论文数值”。

## 11. 相关校验

文档和配置静态检查：

```bash
python -m pytest -q \
  tests/resilient_v2x/test_reproduction_docs.py \
  tests/resilient_v2x/test_configs.py
```

协议、证据与 V2XSet 边界回归：

```bash
python -m pytest -q \
  tests/resilient_v2x/test_overlay_cli.py \
  tests/resilient_v2x/test_runtime_dataset.py \
  tests/resilient_v2x/test_detection_evaluation.py \
  tests/resilient_v2x/test_evidence_profile.py \
  tests/resilient_v2x/test_v2xset_pair.py \
  tests/resilient_v2x/test_v2xset_standard.py
```

完整 ResilientV2X 单元测试仍不能替代正式 GPU 训练、评测和证据产物。
本开发机上通过的真实 MMDetection3D/CUDA 合成与 fixture 集成 smoke 也只
用于证明路径可执行；不得把其峰值显存、随机损失、预测或指标写入论文结果。
