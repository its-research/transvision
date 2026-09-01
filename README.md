# Resilient V2X

**Temporally Valid Feature Repair and Reliability-Aware Routing for Collaborative 3D Detection**

Vehicle-to-everything (V2X) collaborative perception extends ego-vehicle sensing with roadside observations, but it is vulnerable to two coupled degradations: communication latency makes remote features stale, while short LiDAR or camera outages remove current evidence. Resilient V2X addresses both within an explicit decision-time information boundary. It repairs the newest valid historical features with a Perception Trajectory Field (PTF), then uses Dynamic Expert Routing (DER) to fuse LiDAR, camera, and cross-modal experts according to availability, latency, and reliability.

## Abstract

Existing collaborative perception methods typically address communication asynchrony and modality loss separately. Resilient V2X places both degradations within a common decision-time information boundary. For each agent–modality branch, it selects the newest valid BEV representation available by decision time; invalid candidates are excluded, and branches without a supported fallback return zero features. After geometric alignment, a horizon-conditioned PTF propagates valid historical features and estimates branch reliability. DER then combines LiDAR, camera, and cross-modal experts using expert-output summaries, modality availability, branch-source lag, modality-level reliability, and RSU endpoint lag.

On the fixed DAIR-V2X validation subset under the common controlled protocol, Resilient V2X achieves the highest observed BEV AP@0.7 among the compared methods in all 12 input-condition–delay combinations.

## Contributions

1. We formulate collaborative 3D detection with a single ego vehicle and a single RSU under short-duration modality loss and bounded communication latency, with explicit arrival cutoffs, fault masks, and behavior for unsupported inputs.
2. We couple horizon-conditioned PTF propagation and branch-reliability estimation with reliability-aware DER.
3. Under a common protocol, we evaluate individual and joint degradations, failure duration, component ablations, reliability and routing diagnostics, and deployment cost on a fixed DAIR-V2X validation subset.

## Evaluation Protocol

Results use the `DAIR-CAUSAL-1337-v1` protocol. Within the common experimental stack, all controlled methods share the samples, encoders, detection projection and head, initialization of shared modules, optimizer, training schedule, fault-injection protocol, and evaluator. Fusion modules are not parameter-matched. Resilient V2X additionally uses distillation from a full-input teacher, so the reported rankings compare complete training configurations.

| Protocol item | Definition |
| :--- | :--- |
| Dataset | DAIR-V2X cooperative vehicle–infrastructure data |
| Evaluation subset | 1,337 ego–RSU validation samples with 11,330 Car ground-truth boxes |
| Evaluation region | Current ego frame, `[0, 80] × [-40, 40] × [-3, 1] m` |
| Nominal added RSU delay | No added transport delay, 100 ms, 200 ms, and 300 ms; native capture-time skew is retained |
| Input conditions | `Full`, `L-Fail`, and `C-Fail`; the two failure conditions remove the corresponding modality from both the ego endpoint and the newest arrived RSU endpoint |
| Evaluation grid | 6 methods × 4 delays × 3 input conditions × 4 metrics, for 288 controlled result cells |
| Metrics | Car BEV/3D AP-R40 at IoU thresholds 0.5 and 0.7; the primary metric is BEV AP@0.7 |
| Training and checkpoint | Fixed seed `20250218`; the student uses the final epoch-50 checkpoint |
| Evidence status | The controlled result registry is `verified`. All 1,337 samples are retained in every controlled run (`unsupported_sample_count=0`); individual agent–modality branches may still be unsupported and are zero-masked |

The five comparison methods are protocol-aligned implementations based on published descriptions of FFNet, CoFormerNet, V2X-ViT, CoBEVT, and BEVFusion. They are not exact reproductions of the published code or original configurations. The results support only within-protocol comparisons and should not be compared directly with published values obtained using different data, modalities, backbones, or evaluators.

## Main Results

### Full Input, No Added Delay

Values are AP-R40 (%). Bold indicates the highest observed value in each column.

| Method | BEV AP@0.5 | BEV AP@0.7 | 3D AP@0.5 | 3D AP@0.7 |
| :--- | ---: | ---: | ---: | ---: |
| FFNet-style | **71.94** | 59.24 | 66.70 | 37.51 |
| CoFormerNet-style | 69.96 | 58.70 | 64.41 | 35.03 |
| V2X-ViT-style | 69.91 | 58.67 | 64.41 | 34.19 |
| CoBEVT-style | 67.89 | 59.11 | 64.41 | 35.76 |
| BEVFusion-style | 70.38 | 59.75 | 65.07 | 37.80 |
| **Resilient V2X** | 70.93 | **62.36** | **67.63** | **40.57** |

### Communication Latency and Modality Loss

Each cell reports “Resilient V2X / best-performing controlled comparator” in BEV AP@0.7 (%).

| Nominal added RSU delay | Full | L-Fail | C-Fail |
| ---: | ---: | ---: | ---: |
| 0 ms | **62.36** / 59.75 | **53.19** / 34.62 | **62.32** / 59.69 |
| 100 ms | **62.18** / 59.51 | **53.24** / 34.88 | **62.13** / 59.54 |
| 200 ms | **60.42** / 59.43 | **53.21** / 34.81 | **60.41** / 59.45 |
| 300 ms | **60.38** / 59.35 | **50.86** / 33.65 | **60.19** / 59.27 |

The results support the following observations:

- Resilient V2X achieves the highest observed BEV AP@0.7 in all 12 evaluated conditions.
- Under `Full` input, Resilient V2X ranks first in BEV AP@0.7 and 3D AP@0.7 at all four delay settings; it does not lead BEV AP@0.5 at every delay.
- Under both `L-Fail` and `C-Fail`, Resilient V2X ranks first in all four reported metrics at every delay.
- `L-Fail` is the more challenging fault condition under this protocol. At 300 ms, no earlier RSU observation remains within the supported window for the affected branch; Resilient V2X records 50.86 BEV AP@0.7.

### Consecutive Failures and Deployment Cost

Under the no-delay setting, BEV AP@0.7 decreases substantially as LiDAR loss extends from one to three frames, whereas it changes little over the corresponding camera-loss durations under this protocol.

| Consecutive-failure duration | LiDAR-loss BEV AP@0.7 | Camera-loss BEV AP@0.7 |
| :--- | ---: | ---: |
| 1 frame | 53.19 | 62.32 |
| 2 frames | 11.57 | 62.35 |
| 3 frames | 5.27 | 62.37 |

Deployment measurements exclude the teacher and all training-only computation. Model-path latency is the median of 100 GPU-synchronized iterations after 10 warm-up iterations on an RTX 5090 at batch size one. Timing starts with a collated `model.test_step` batch and covers preprocessing through decoded predictions; data loading and collation are excluded. FLOP counts include only operations supported by `torch.profiler`.

| Model | Parameters | `torch.profiler`-supported FLOPs | Peak allocated GPU memory | Model-path latency |
| :--- | ---: | ---: | ---: | ---: |
| Capacity-matched concatenation control | 35.076 M | 128.744 G | 0.833 GB | 52.341 ms |
| Resilient V2X student | 35.076 M | 128.745 G | 0.833 GB | 53.534 ms |

### Ablation Findings

- The zero-offset PTF retains fallback-source selection, ego-frame alignment, and source-lag decay, but lowers `L-Fail` BEV AP@0.7 from 53.19 to 32.81, the largest observed decrease among the single-factor ablations.
- Removing distillation lowers BEV AP@0.7 in all three reported settings: from 62.36 to 60.34 under `Full` at 0 ms, from 53.19 to 50.84 under `L-Fail` at 0 ms, and from 60.38 to 58.15 under `Full` at 300 ms.
- No variant is best in all three settings: linear PTF exceeds the complete configuration under `Full` at 300 ms, and binary support does so in both `Full` settings; the complete configuration achieves the highest observed `L-Fail` score.
- The failed modality's mean reliability decreases in all four displayed comparisons. These diagnostics are descriptive: they neither isolate the effect of reliability on routing nor evaluate calibration.

## Paper-to-Code Map

| Paper component | Implementation entry point |
| :--- | :--- |
| Resilient V2X detector | `transvision/models/detectors/resilient_v2x.py` |
| PTF, reliability, and DER | `transvision/models/resilient_v2x/` |
| Temporal manifests, scheduling, and runtime datasets | `transvision/dataset/resilient_v2x_*` |
| Controlled conditions and common evaluation | `transvision/evaluation/resilient_v2x_*`, `tools/resilient_v2x/evaluate_controlled_baselines.py` |
| Main model, ablations, and protocol-aligned methods | `configs/resilient_v2x/` |
| Data preparation and condition overlays | `tools/resilient_v2x/prepare_data.py`, `tools/resilient_v2x/build_overlays.py` |
| Training | `tools/train.py`, `configs/resilient_v2x/dair_clean_teacher.py`, `configs/resilient_v2x/dair_resilient_v2x.py` |
| Profiling and evidence packaging | `tools/resilient_v2x/profile.py`, `tools/resilient_v2x/build_evidence.py` |

## Reproduction

The locked environment targets Linux/amd64, CUDA 11.8, Python 3.10, PyTorch 2.0.1, MMEngine 0.10.7, MMCV 2.1.0, MMDetection 3.2.0, and MMDetection3D 1.3.0. Use the provided container:

```bash
docker build \
  -f environments/resilient_v2x/Dockerfile \
  -t resilient-v2x:locked \
  .

docker run --rm --gpus all resilient-v2x:locked
```

Run the regression suite in an environment with the development dependencies installed:

```bash
python -m pytest -q tests/resilient_v2x
```

Raw DAIR data, `artifacts/`, `models/`, and `work_dirs/` are local or external artifacts and are not stored in Git. The corresponding commands for data preparation, two-stage training, 12-condition evaluation, profiling, and evidence packaging are under `tools/resilient_v2x/`.

## Documentation

- [Configuration and training](configs/resilient_v2x/README.md)
- [Core API](docs/resilient_v2x/core-api.md)

## Upstream Projects

- [MMDetection3D v1.3.0](https://github.com/open-mmlab/mmdetection3d/tree/v1.3.0)
- [DAIR-V2X](https://github.com/AIR-THU/DAIR-V2X)
- [FFNet-VIC3D](https://github.com/haibao-yu/FFNet-VIC3D)
