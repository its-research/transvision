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
