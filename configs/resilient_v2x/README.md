# ResilientV2X reproduction configurations

These configurations implement the paper's causal four-branch
LiDAR/camera pipeline on DAIR-V2X.  They do not claim unpublished paper
hyperparameters or numerical results.  Every unreported numeric setting is
listed under an `implementation_choices_*` field in the merged config.

Use the [controlled reproduction guide](../../docs/resilient_v2x/reproduction.md)
for executable commands and the
[paper-to-code coverage matrix](../../docs/resilient_v2x/paper-coverage.md)
for implementation, test, and evidence boundaries.

## Two-stage training

1. Prepare the strict temporal manifest and protocol overlays.  Keep their
   uncompressed SHA-256 values; the runtime verifies them before reading data.
2. Export `RESILIENT_V2X_DATA_ROOT`, `RESILIENT_V2X_MANIFEST`, and
   `RESILIENT_V2X_SPLIT_SHA256`.
3. Train `dair_clean_teacher.py`.  Put the selected clean checkpoint at
   `work_dirs/resilient_v2x_dair_clean_teacher/best_teacher.pth`, or override
   `model.teacher_checkpoint`.
4. For the distilled student, also export the train transport/fault overlay
   paths and hashes named in `dair_resilient_v2x.py`, then train that config.

The main student deliberately has real overlay paths but no default hashes.
This makes an omitted external protocol artifact fail fast instead of silently
running a different experiment.  To intentionally train without one overlay,
set both its path and SHA-256 fields to `None` with `--cfg-options`.

## Evaluation matrix

`conditions/global_delay_*_full.py` provides the Full rows at 0/100/200/300
ms.  The matching `causal_delay_*_{l,c}_fail.py` files provide L-Fail and
C-Fail.  Their overlays must select each ego/RSU endpoint under the transport
schedule first and apply the fault second; a target-tick mask is not a valid
substitute at non-zero delay.  `causal_fault_diagnostic.py` covers E+R,
E-only, and R-only scopes with continuous outages of one to four ticks.
Non-zero delay/fault configs require the exact path and digest environment
variables documented in each file.

`ablations/` contains PTF, routing, reliability, delay-metadata, and
distillation ablations.  Use the same dataset/protocol artifacts and checkpoint
selection across a comparison.

`ablations/concat_capacity_matched.py` is the deployment-cost comparison
requested by the paper.  It has exactly the student's trainable parameter
count and does not execute DER.  Because the manuscript does not define the
concat topology, the chosen masked modality-expert concat projection is an
explicit implementation choice rather than a claimed paper hyperparameter.

## Controlled comparison baselines

`baselines/` contains V2X-ViT-, CoBEVT-, CoFormerNet-, BEVFusion-, and
FFNet-style
fusion modules behind one detector interface:

- `baselines/v2x_vit.py`
- `baselines/cobevt.py`
- `baselines/coformernet.py`
- `baselines/bevfusion.py`
- `baselines/ffnet.py`

All five reuse the main experiment's DAIR-V2X manifest, causal transport and
fault overlays, shared LiDAR/camera encoders, detection head, optimizer, and
evaluator. They train directly against detection targets with
`include_clean_teacher=False`; no teacher, distillation, PTF, dynamic router,
or DER path is present in their model configs.

These configs are controlled multimodal adaptations. They are not claimed as
bit-exact reproductions of source-paper systems whose published dataset,
modality, delay/noise, bandwidth, or metric protocol differs. Do not copy such
published numbers into the controlled DAIR-V2X result cells. Train each config
with the same external split and train-overlay identities.

The FFNet paper estimates a first-order derivative from two consecutive RSU
frames in a separate self-supervised stage. The common selected-branch contract
contains only one causal source per branch; `baselines/ffnet.py` therefore uses
the disclosed single-frame, age-conditioned L+C adaptation and must not be
reported as the official two-frame FFNet implementation.

Resolve and validate one sealed training run before allocating the GPU:

```bash
python tools/resilient_v2x/train_controlled_baseline.py \
  --baseline v2x_vit \
  --training-index artifacts/resilient_v2x/dair_v2/training_overlays.json \
  --work-dir work_dirs/controlled_train/v2x_vit_seed_20250218 \
  --seed 20250218 \
  --dry-run
```

The command requires `RESILIENT_V2X_DATA_ROOT`, `RESILIENT_V2X_MANIFEST`, and
`RESILIENT_V2X_SPLIT_SHA256`. It writes a sealed `training_plan.json` and a
fully resolved config with the train transport/fault overlay identities. Remove
`--dry-run` to train. Use `--resume` only in the same work directory with its
matching existing plan and checkpoint; an orphan or identity-mismatched
checkpoint is rejected.

Evaluate a selected checkpoint through the unified entry point rather than by
invoking the condition files directly:

```bash
python tools/resilient_v2x/evaluate_controlled_baselines.py \
  --baseline v2x_vit \
  --checkpoint work_dirs/v2x_vit/best.pth \
  --overlay-index artifacts/resilient_v2x/dair_v2/evaluation_overlays.json \
  --work-dir work_dirs/controlled_eval/v2x_vit \
  --dry-run
```

The entry point verifies that `RESILIENT_V2X_MANIFEST` and the overlay index
name the same manifest content hash, injects the exact E+R one-tick overlays,
and replaces only the condition config's model with the selected baseline.
Remove `--dry-run` to run the 12 conditions sequentially. Each condition gets a
resolved config, checkpoint digest, and `predictions.json`; aggregate metrics
are written atomically to `metrics.json`. Use `--delays` or `--conditions` only
for an explicitly documented subset run.
