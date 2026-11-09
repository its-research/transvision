# V2V4Real causal motion and covariance dependency

This is an implementation dependency audit, not a fitted model or acceptance.
The three-seed post-head numerical scope completed on 2026-10-05. The next 9D
cache gate remains closed because its motion/covariance producer and independent
admission have not been implemented or executed.

`build_v2v4real_raw_native_cache.py` consumes a contract; it performs no velocity
estimation or covariance fitting. Its exact envelope binds raw manifest/index/
archive, projection frames, nominal clock, detector config, method implementation,
model, train-fit receipt and separate independent admission. Every source row,
including empty rows, must have a unique SHA-bound NPZ containing `velocity_source`
(N×2, original state dtype) and full `covariances` (N×9×9). Dependencies must refer
to bound rows from the same sequence and side with no future ordinal. The consumer
cannot supply this missing evidence or manufacture zero velocity/diagonal noise.

Before fitting, the producer must admit actual train detector rows with source
poses, ordinals, PCD and checkpoint identity, plus independently accepted native
vehicle train GT. Existing existence-calibration receipt covers binary existence
probabilities; it provides no velocity residuals or full covariance parameters.
The historical detector train/test overlap must remain disclosed. A train-only
fit does not establish physical session independence.

The producer needs an explicit method for each of these cases before execution:

- Past-only source-local detection association and velocity estimation in a
  moving lidar frame, with source-to-world transforms and actual source-time
  differences derived from the frozen nominal clock.
- First observation, empty frames, missing predecessor, unmatched candidate,
  duplicate frame and nonpositive time difference. Initial-state uncertainty
  must come from the admitted training method; unavailable motion cannot be
  relabelled as measured zero velocity.
- Train-only target construction using past identity observations and pose
  transforms. Future GT and all official-test GT are excluded from fit/model
  selection and online inference.
- Nine-dimensional residual layout, yaw wrapping, cross terms, positive
  definiteness, sample sufficiency and any regularization. The estimator and its
  fitting rule must be named and frozen; full covariance must not be invented by
  filling diagonal placeholders.

Independent admission must reconstruct causal dependencies and train examples,
recompute model parameters and full covariance, and compare every produced state
and covariance using the frozen numerical comparison policy. A producer-generated
boolean or a bridge persistence readback alone cannot establish that admission.
The evaluator must reject future/cross-source rows, modified input hashes,
changed coordinate conventions, train/test leakage and inadmissible initial
states. Real-data training, inference, independent acceptance and bridge execution
are separate stages; none is complete at the time of this audit.

Next executable preparation: inspect the existing native train export and
training-side calibration recipes, freeze the explicit estimator/initial policy
and evidence schema, then implement producer and independent oracle separately.
No accepted detector/raw/numerical task should be repeated to prepare this stage.

## Train supervision primitive prepared at 13:17 UTC

`v2v4real_motion_train_targets.py` now prepares past-only source-local native
identity velocity targets from explicitly supplied train frames. It rejects
non-train manifests before iterating labels, uses column-vector source-to-world
poses and nominal ordinal-derived 10 Hz time, keeps first observations invalid
(NaN plus mask), isolates sequence/source histories, retains empty frames, and
breaks ambiguous identity histories. Five new synthetic boundary/pose checks
passed, including a static world object observed by a translating sensor and a
rotated source. Initial synthetic side names were corrected to the existing
vehicle-side/infrastructure-side cache contract before final checks.
This is a supervision helper only: no actual train input byte admission, native
GT loading, model fitting, covariance estimation, online use or 9D acceptance
has occurred. The existing SPD residual covariance recipe is a candidate
reference, not inherited V2V acceptance or a chosen fitted model.


## Native source-label byte admission at 13:32 UTC

The previously accepted 1017-frame calibration GT is an ego-frame union: it
selects the first CAV identity before final ROI and is not a per-source motion
supervision cache. Do not inverse-transform that union and call it independently
observed source GT. The three original train volumes train_04/train_05/train_08
retain the native YAML labels for both CAVs. The new admission primitive verifies
pinned train volume receipts before label reads, resolves both source labels for
every explicitly selected frame, rejects symlinks and malformed inventories, and
checks each original byte size and SHA. Three new boundary tests passed.
This step only admits bytes for a new dependency; label parsing, per-source native
geometry/identity policy, pose/clock binding, independent target oracle, actual
velocity targets and covariance fitting remain unaccepted.


## Source geometry adapter prepared at 13:47 UTC

`v2v4real_motion_source_frame.py` parses an explicitly SHA-bound training label
through the existing restricted numeric YAML decoder, native vehicle selector,
source-local upright geometry and native float32 source ROI. It requires exact
agreement with the separately admitted raw source pose. It preserves raw classes
and duplicate mapped IDs for history invalidation; it applies no cross-CAV union
or final ego ROI. Empty source rows remain explicit. Three new synthetic checks
passed for local coordinates despite a translated lidar pose, duplicate IDs,
Pedestrian exclusion before geometry, empty ROI and split/byte/pose rejection.
This is adapter preparation: the admitted real 2034 labels have not yet been
parsed through it, raw pose/clock lineage has not been admitted for this stage,
and source geometry/target mathematics still require a separate independent
oracle before any claim of motion or full-covariance acceptance.


## First real train target candidate at 14:02 UTC

The first candidate command exited before label reads because its check compared
a historical frozen documentation SHA to the later updated workspace document.
The failed command/log remain preserved. The corrected command verifies every
historical frozen file and current executable Python source instead; numerical
or source geometry checks were not relaxed.

The create-once candidate at
`artifacts/v2v4real-motion-train-source-target-candidate-20261005T140507076495Z`
consumed the original 2034 admitted source labels. All three train native raw
indexes and their prior independent admission pins were tied to the original
projection pose/PCD/ordinal rows, with exact label-to-raw poses. It produced
21953 source objects and 21427 valid past-identity velocity targets, preserving
invalid first/ambiguous observations with NaN plus a validity mask. The candidate
receipt SHA is `92d9389ea3fa798d26045de4acd197278ec5c87c16b624fda1d6bfcaaa006ac0`.
This is actual candidate construction, not independent geometry or target-math
acceptance. The next gate is an independent per-source original geometry oracle
and separately implemented past-history velocity oracle over every candidate
row, including missing/ambiguous/empty cases. No online detector motion estimator,
initial-state model, residual covariance fitting or 9D cache has been accepted.


## Independent train source geometry and target mathematics at 14:17 UTC

The new independent checker executes SHA-pinned original `project_world_objects`
and its original geometry/ROI definitions separately for each source, preserving
mapped duplicate identities instead of merging CAVs. A separate scalar `fsum`
past-history oracle reconstructs world displacement, source velocity, validity
and predecessor ordinals without importing either producer helper. Two synthetic
history tests and a pinned-original source geometry smoke check passed.

The actual complete 2034 source rows / 21953 objects / 21427 valid targets passed
at the unchanged absolute and relative tolerances 1e-4. Maximum absolute geometry
error was 2.220446049250313e-15 and velocity error 4.576228285202433e-12.
The receipt is
`artifacts/v2v4real-motion-train-target-independent-20261005T142111884030Z/acceptance-receipt.json`
with SHA `047d9a5ead1f333a6fd9b4e7542776af32568a254bcc7280407a9a93259302ef`.
Its scope is offline train native source geometry and past-identity supervision.
It does not verify physical identity continuity, measured time, online detector
association/velocity, initial-state uncertainty, fitted covariance, 9D cache,
tracking performance or paper metrics. The next implementation dependency is the
causal detector motion estimator and explicit initial policy, followed by admitted
train residual fitting and independent full-covariance verification.

## GT-free detector motion preparation after the status audit

`v2v4real_causal_detector_motion.py` now estimates velocity from the adjacent
available source detector row, with same sequence/side, strictly earlier nominal
ordinal and original row SHA dependencies. It retains every all-class candidate.
Matching uses the unique mutual nearest 3D world centre among same-class edges
within explicit maximum-speed and ordinal-gap gates. Exact ties and non-mutual
assignments remain unavailable; no score/appearance/GT identity enters matching.
An empty row breaks the adjacent detection history. Poses compensate moving and
rotating lidar frames before the displacement is expressed in the current source
frame. Inputs are not modified and stored history snapshots are independent.

The gates have no defaults. Their numerical values still need train-development
selection and freezing before any official-test execution. First/unmatched
observations expose NaN plus a validity mask, not measured zero velocity. This
preparation output cannot be admitted to the finite 9D bridge until a train-fitted
initial-state model and full covariance producer are independently accepted.
The matching rule is an explicit implementation choice, not a claim that it is
an independently verified physical identity assignment.

A separate scalar `math.fsum` oracle imports no producer functions and rebuilds
world coordinates, allowed edges, unique mutual assignments and velocities.
Six new synthetic checks passed, including static world objects under translated
and rotated sensors, all-class competition, one-to-one conflicts, exact ties,
speed/gap gates, empty/source-isolated histories, forbidden future/duplicate rows,
invalid poses, and 40 seeded differential cases at the unchanged 1e-4 tolerances.
These checks establish software preparation only. No real detector-row motion,
gate selection, initial-state fitting, covariance fitting, bridge execution or
tracking/paper metric has been accepted in this step.

## Actual three-seed train detector input consumption

The new motion input consumer binds the already accepted 2034 source train rows
to the original raw indexes, checkpoint seeds, source poses, PCDs and nominal
ordinals. It consumes the existing local original archives without running a
detector. All selected original NPZ payload hashes are checked, and candidate
order, raw legacy boxes, classes and scores are preserved in separate motion
input arrays. The three seeds contain 129286 / 128791 / 128161 candidates.
The total 386238 agrees with the selected training detector input scope.

The first attempt failed at an exact member inventory check because the original
archive additionally includes `predictions.jsonl`. Inspection confirmed that
this member is byte-identical to the separately bound `native-feature-index`.
The consumer now explicitly requires that original index member with its exact
SHA and parsed rows, and the original anchor member with its pinned SHA. Four
consumer boundary checks passed after this correction. The original failed
command/log/output directory remain preserved; the corrected command exited 0.

The input receipt is
`artifacts/v2v4real-motion-detector-train-input-admission-20261005T143945333369Z/input-admission-receipt.json`
with SHA `7e174f1860e0de24e5258c5e38ed45e2a81379aaf407977e51850a76a3833b72`.
Its source/test hashes bind the corrected consumer, but its static test summary
still records the three pre-correction checks. The separate execution-history
receipt records the actual corrected four-check result; the original receipt is
not overwritten. This step admits inputs for motion fitting, not a new raw
detector numerical acceptance, actual velocity computation, gate selection,
initial-state model, covariance fitting, 9D bridge or paper performance result.

## Real train fitting and independent full-split motion on 2026-10-06

The dependency above has advanced beyond software preparation. The frozen recipe
uses the two original `calibration_fit` name groups, leaving each group out in
turn. Candidate speed gates are 5/10/20/40 m/s and ordinal gaps 1/3/10. Every gate
scores the same offline, score-ordered, unique 2 m XY detector/GT matches with
valid past-identity velocity. Held-out initial priors use only the other group.
Selection is the mean vector velocity MSE across seeds and held-out groups, with
smaller speed/gap breaking exact ties. Test does not participate.

The selected gate is 40 m/s and one nominal frame. Per-side population priors use
valid past training GT velocities. Unavailable detector velocity retains its
flag and receives this prior; it is not a measured zero. First observations
transfer the unavailable population model without future first-frame targets.
This explicitly assumes transfer of that population distribution; it does not
prove the physical velocity of a first observation.

Full9 residuals retain the raw legacy seven-column state, convert GT l/w/yaw to
the same convention, and wrap yaw residuals. Each seed/side has separate matched
and unavailable models: the uncentered full second moment, 5% diagonal shrinkage
and a frozen 1e-6 identity floor. Cross terms and spatial bias are retained because
the observed seven columns are not corrected. Only matches with valid past GT
velocity enter the fit. False-positive uncertainty and physical identity
continuity are not claimed to be calibrated or independently verified.

The fitting candidate is
`artifacts/v2v4real-motion-train-fit-candidate-20261006T152728173245Z/candidate-receipt.json`
(SHA `04a704aa1c0c3c74fcdbfc083a355148d5c77f0724638d4515f4e08178569ee1`).
Separate scalar code recomputed all 12 gate scores, matching, priors, full
residual moments and every motion/covariance for the original 386238 candidates.
Its receipt is
`artifacts/v2v4real-motion-train-fit-independent-20261006T153123039889Z/acceptance-receipt.json`
(SHA `99a38ceb4c04f4d17a42faa6743c3c75979a153165970bb9620ba9d99174d31e`).
Maximum velocity difference is 9.094058839309582e-12 and covariance difference
5.684341886080802e-14, at unchanged 1e-4 absolute/relative tolerances.

GT-free application completed three seeds on full train (14210 source rows /
32 sequences each) and official test (3986 source rows / 9 sequences each).
Candidates are in `artifacts/v2v4real-full-motion-campaign-20261006T154211951265Z`.
All six scopes passed scalar independent reconstruction in
`artifacts/v2v4real-full-motion-independent-campaign-20261006T154611112121Z`:
54588 source rows, 3060073 candidates, maximum velocity difference
9.160672220787092e-12 and applied covariance difference 0. Every actual payload
is finite, symmetric and positive definite. Test GT was not opened and test did
not select or fit the model. These receipts verify the motion policy's numerical
application, not tracking accuracy or independence of the overlapping lineage.

The new motion chain passed 31 targeted software checks, including rehashed
wrong velocity, removed cross terms and false dependencies. The existing bridge
passed a separate 34-check regression suite. These are software checks, not
experiment counts or paper completion.

Packaging and actual 9D conversion started in
`artifacts/v2v4real-motion-contract-and-9d-bridge-20261006T155046228454Z`.
The original config is copied from its pinned archive; nominal 10 Hz time uses
explicit zero relative origin per sequence; admitted NPZ bytes are unchanged.
Completion, independent cache readback and identity-model qualification remain
separate gates.

Concurrent A/D continuation: Top1/K4 have all 276 sequence receipts and six
whole-cohort acceptances. Forest seeds 1337/3407 have complete independent scope
receipts, including capacity-undecided semantics rather than full-optimality
claims. Their final-model witness teachers are ClearML tasks
`b1122e3f236a4d498aaf4ffc2c7dd81e` (8-card 3090) and
`59d822eaf3d248afac762f3b5ddf2471` (4-card 2080Ti), both observed `in_progress`.
Teacher target acceptance is not inherited from that status. Original seed2027
rank5 and previously rejected CUDA/Linux priority assets retain their gates.
No existing replay was restarted; no commit or push was performed.

## Actual 9D cache independent readback completed on 2026-10-07

Both the six-scope bridge and the separate cache oracle exited zero. The oracle
opened every original raw payload and cache array, independently recalculated
existence probabilities, reconstructed frame digests, and verified exact
seven-state, appearance, raw-score, admitted velocity and full covariance bytes.
All six cache manifests and bridge receipts match the independent receipt SHA.
Coverage is 54588 source rows and 3060073 candidates, retaining empty source rows.

Independent receipts: `/Volumes/Data/test/recover-before-fuse/artifacts/
v2v4real-9d-cache-independent-campaign-20261006T155812142274Z`.
The milestone receipt is `/Volumes/Data/test/recover-before-fuse/receipts/v2v4real-full-nine-state-input-milestone-20261006T160226874633Z.json` (SHA `573d830578f889ba44ccf7a38e1372b97aa941001c6ffd2a526aae0e34ab78a9`).
This closes the numerical input/cache gate in B; it does not qualify an identity
model, verify tracking metrics or erase train/test overlap. The accepted prior
SPD all-class checkpoints are dataset-specific and require a separate V2V
training/supervision contract. The original seed2027 archive and rejected
CUDA/Linux-priority packages retain their stated gates.

## V2V identity qualification and full native supervision continuation

The historical independent model cohort totals 677744 rows across three seeds,
46 SPD train sequences, 7445 pairs and 16338 source frames. The weights and data
manifest SHA were read back against their original byte receipts on 2026-10-07.
They do not qualify V2V native BEV128 inputs. This finding is recorded in
`/Volumes/Data/test/recover-before-fuse/receipts/rbf-joint-identity-dataset-qualification-20261006T161342368029Z.json`.
No accepted SPD model was retrained.

Full V2V train supervision now uses all 32 sequences, 14210 source frames and
137134 native objects. The separate original AST geometry/ID checker passed all
rows with maximum absolute geometry error 3.552713678800501e-15 m at the unchanged
1e-4 m tolerance. Candidate and independent receipts are in
`/Volumes/Data/test/recover-before-fuse/artifacts/v2v4real-full-train-identity-label-signed-native-campaign-20261006T161036701541Z`.

The first strict preparation failed after 5521 source rows because one native
Truck width extent was negative. Its output and failure remain in
`v2v4real-full-train-identity-label-campaign-20261006T160717620170Z`.
The author geometry retains signed corner operands and measures positive edge
lengths; the new full-train adapter implements those same operations, without
absolute-value repair. All full-train rows were then independently compared.
The earlier strict motion/source parser was not changed; zero dimensions still
raise a real failure. A preflight pose checker also rejected the simulation
six-vector pose function on native 4x4 matrices; the corrected checker binds the
original native matrix directly and does not relax a numerical threshold.

Temporal identity targets use original per-source object IDs. Cross-source
links require equal explicit nonmissing association IDs within one paired frame.
The fallback object-ID namespace does not create cross-source links. Repeated
native IDs, changed association IDs and ambiguous joined components remain
unknown targets. The common neural feature slots are retained, with native
merged vehicle explicitly occupying car slot zero; this is not an assertion of
three native categories or physical identity continuity.

Three-seed row preparation is running in
`/Volumes/Data/test/recover-before-fuse/artifacts/v2v4real-three-seed-native-identity-rows-campaign-20261006T161442413941Z`.
An earlier attempt used the strict SPD cache loader and was rejected before row
creation; that failure is retained. The running producer uses the existing
separate NativePaperCache loader, preserving the SPD envelope's rejection.
Independent row verification is waiting for each complete seed receipt in
`/Volumes/Data/test/recover-before-fuse/artifacts/v2v4real-three-seed-identity-row-independent-20261006T162006440452Z`.
The checker imports no producer row, cache, world transform or identity-index
functions. It separately reconstructs the full source-component graph, raw
score matching, prediction-only world features/covariance, all causal gated
parents, and partial birth/parent targets. The first 500 real seed1337 nodes
passed a diagnostic feature/context comparison with maximum difference zero;
this limited diagnostic is not full row admission or checkpoint acceptance.

Two final-model witness teachers and three recovery-off replays still report
in_progress without artifacts as of the metadata snapshot
`/Volumes/Data/test/recover-before-fuse/receipts/rbf-current-GPU-and-rank5-metadata-only-20261006T162246429513Z.json`.
Seed2027 original rank5 still registers the previous size/SHA; its archive was
not opened again and no good replacement was claimed. All stated CUDA/priority
and MHT authorization gates remain unchanged. Goal execution remains active;
no experiment or paper completion percentage is inferred from software checks.

## Current SPD final-refit qualification and prepared native training path

The current SPD final-refit weights were separately checked against all actual
training/checkpoint/forward byte proofs and full float64 numerical receipts.
The current forest dispatches bind those exact weights. Seeds 1337, 2027 and
3407 selected epochs 3, 9 and 1 and cover 229212, 225177 and 223355 rows,
respectively (677744 total). This is recorded in
`/Volumes/Data/test/recover-before-fuse/receipts/rbf-current-SPD-final-refit-identity-qualification-20261006T162942908750Z.json`
(SHA `5ade0f6eaaf13a9b18db61ae4d13bb0a1730913c6347d8652c008f1fff61c87a`).
These applicable SPD models are reused. This receipt does not qualify V2V or
complete the online method or paper evaluation.

The separate native training runner now requires the complete independent
32-sequence real row proof before updates. It retains the original filename
name-group fit/selection roles, selects the earliest minimum two-head sequence
macro loss among ten epochs, then initializes the same seed anew and refits
all original train sequences for the selected epoch count. Cross-source and
temporal heads use their own global informative-example denominators under
DDP, including unequal rank/head counts and empty rank tails. A real four-rank
CPU/Gloo gradient comparison passed against the direct complete-batch network.
Zero-informative global batches skip optimizer updates. No strict upstream or
physical-session isolation is claimed, and known train/test overlap remains
explicit.

The final candidate exports every original row, including unknown-target rows,
without using target masks to select forward coverage. Weights are saved with
CPU storage so the unchanged independent NumPy reader can read them directly.
The separate checkpoint checker independently recomputes original group and
epoch-selection metadata, reads initial/epoch/final weight bytes, checks
sequence/rank/tail coverage, and applies the original SHA-bound full neural and
motion reference to all final rows at unchanged atol/rtol 1e-4. Separate cloud
registered-byte readback is still required; local numeric acceptance cannot
replace it.

The source preparation is
`/Volumes/Data/test/recover-before-fuse/source-freezes/v2v4real-native-identity-training-and-independent-admission-20261006T165321764645Z/preparation.json`
(SHA `f9b9180bc87c23e08c2250b1f9a1fbe92be889a0bd1689b06a973302525cbfab`).
Twenty-seven targeted software checks passed. An untrained 128-hidden/4-head
network on 127 real seed1337 rows covered both sources and context lengths
1/2/3/4/5/9; independent logits had zero failed rows and maximum absolute
difference 3.4874503853643546e-6. The independent weight digest also matched.
That diagnostic is
`/Volumes/Data/test/recover-before-fuse/artifacts/v2v4real-native-identity-NN-and-weight-diagnostic-20261006T164622567783Z/receipt.json`.
It is neither full native row admission nor a trained checkpoint qualification.

An input-preparation coordinator is live at
`/Volumes/Data/test/recover-before-fuse/artifacts/v2v4real-native-identity-admitted-GPU-input-preparation-20261006T165428896945Z`.
It waits for each existing full independent seed proof, then creates and
reopens a training-only input bundle from frozen sources. It does not upload
or dispatch. Actual GPU work additionally requires independent cloud input
readback and disjoint physical 4/8-GPU admission; the consumer rejects L40S,
duplicate UUIDs, unsupported native CUDA architectures and pre-existing
compute processes on assigned UUIDs before launch. It excludes the previously
rejected priority and CUDA branch-state packages.

As of 2026-10-07 00:47 CST, both witness teachers and all three recovery-off
tasks remained in_progress with no artifacts, as recorded in
`/Volumes/Data/test/recover-before-fuse/receipts/rbf-goal-five-live-task-metadata-20261006T164741101603Z.json`.
At 00:55 CST the existing native row producer and independent checker remained
live; seed1337 still had three completed shards and no complete seed proof.
Native training, full numerical model qualification, cloud input publication
and formal paper metrics therefore remain incomplete.
