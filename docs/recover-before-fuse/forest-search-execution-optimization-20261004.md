# Forest execution optimization candidate, 2026-10-04

This is a distinct execution candidate. Existing frozen sources, active CPU
acceptors, remote jobs, checkpoints and failure records are unchanged. It is
not a completed experiment, GPU qualification, or full Stage2 acceptance.

## Measured bottleneck and implementation

A cProfile replay of the first 12 original events of SPD train sequence 0000
used the accepted serial database in
`/Volumes/Data/test/recover-before-fuse/artifacts/rbf-batched-full-real-sequence-CPU-pair-v1-20261004/serial-replay`.
The immutable input receipt SHA-256 is
`d30d1879c65ee84cfbbefd04eabbf1acd22d7954e30f5811779530955cd307de`.
The profile recorded about 157 seconds, of which about 108 seconds were in
`residual_regions`; it recorded 11.85 million ancestor lookups and 69.89 million
prefix lookups. Profile overhead and concurrent host work preclude interpreting
these times as isolated production throughput.

`transvision/models/event_track_v2x/batched_branch_states.py` provides:

- A bounded per-call `(retained leaf, frontier depth)` ancestor cache. All
  frontier regions, exclusions, weights, ordering and hashes are still emitted.
- A shared 256-entry SQL translation cache. Namespace separation and transaction
  rejection remain enforced; values remain bound SQL parameters.
- An event-local, shared raw-observation cache bounded by the original prefix
  cache entry limit, cleared after successful and failed transactions.
- Float64 batches of independent root-state CI updates, using NumPy on CPU or
  explicit `cuda:N` with TF32 disabled. History order is unchanged within each
  root; births and late observations use the original implementation.

The scheduler is bounded by the original prefix-cache entry limit. If the
dependency graph exceeds that limit, it uses serial replay. State work is still
charged against the existing cap; exhaustion rolls the event back. No cap,
search budget, candidate support, model, score threshold or numerical tolerance
is relaxed. Branches retain separate state histories. Additional execution
cache limits are declared in the candidate audit and must count toward future
same-resource comparisons.

`BatchedStateExclusiveTracker` has a distinct database schema. It supports bound
allocation only and refuses resume pending separate admission. It is not the
default paper runtime. CUDA support here covers state arithmetic, not SQLite
transactions or the entire irregular identity search.

## Reproducible qualification

`tools/event_track_v2x/qualify_batched_branch_states.py` snapshots sources before
import, verifies the input receipt/database hashes, and replays actual original
events with their original observations, appended factors, rescored factors,
empty events and decision indices. It never substitutes final factor rows or
constructs a schedule from nonempty query rows.

The paired check compares selected actions, active/frontier structures, factor
hashes, allocation trace, work counts, output identities and continuous states.
It also compares every persisted branch state, including unselected branches,
with fixed `atol=rtol=1e-8`. Profiling and throughput runs are separate options.
Commands, source hashes, per-event progress, profiles, SQLite results and the
candidate receipt are retained under the experiment artifact directory.

CUDA qualification uses the same CLI with `--device cuda:0`; device identity is
recorded and no fallback silently labels CPU execution as GPU work. This must run
on a collision-free assigned physical device and pass independent numerical
checks before promotion. The current code does not claim 75–80% device memory
utilization. Occupancy alone is not evidence of faster useful forest work.

## Remaining gates

The final CPU candidate was checked on the 12-event original prefix without
profiling: serial 69.758666 seconds, candidate 40.974669 seconds (observed ratio
1.702483). Search decisions, active/frontier structures, factor hashes,
allocation traces and work counts matched. All 1,013 materialized branch states
matched at fixed `atol=rtol=1e-8`; maximum absolute difference was
`1.7763568394002505e-15`. These are a single ordered pair on a shared host, not
isolated or full-cohort throughput claims. The first state-only prototype was
also retained; its profile showed why the ancestor lookup optimization was
necessary. A nonprofile-report export failure and a restricted-sandbox
subprocess SIGABRT are retained with their corrected successful attempts.

Final regression: 58 passed, one real-CUDA check skipped because the local
runtime has no CUDA device. Tests cover full covariances and yaw wrapping,
late arrivals, batch sizes one/four, work-cap rollback, namespace-safe bounded
SQL caching and the complete nonprofile qualification CLI.

Evidence:
`/Volumes/Data/test/recover-before-fuse/artifacts/rbf-independent-root-state-CPU-pair-v3-20261004/candidate-check.json`.

Real CUDA execution, independent full-cohort validation, throughput/memory
measurement on assigned hardware, learned-allocation integration and resumable
execution remain pending. This work must not restart accepted inputs, completed
training, or running acceptance jobs. Overall experiment ETA remains unknown.

## Measured CUDA entry point

`tools/event_track_v2x/qualify_batched_branch_states_gpu.py` requires an explicit
`cuda:N`, a complete independently accepted CPU reference and a new output path.
It reuses that reference instead of executing the serial replay again. The
candidate runs the same original events and compares every materialized state
and the search/work outputs at the existing fixed tolerance. TF32 is disabled.

The entry point records device UUID/name, one-second whole-device memory samples,
process tensor allocation/reservation peaks, failures and observed wall time.
Whole-device memory includes other processes; elapsed time includes CPU forest
work and final numerical comparisons. Neither is an isolated performance claim.
Launch code hashes and reference identity are retained even if qualification
fails after creating its output directory. Existing output directories are
never reused. No artificial memory allocation is used to meet the 75–80% target.

Launcher and state regression: 10 passed, one actual CUDA check skipped locally.
These are software checks only. The CUDA variant has not run. Dispatch requires
collision-free capacity and gives the preceding core experiments priority.

## Complete CPU sequence and independent admission progress

The optimized CPU candidate completed all 195 original events in 2469.480574
seconds, with a verified process exit code of zero. Its comparison checked
56,389 materialized branch states at unchanged `atol=rtol=1e-8`; the maximum
absolute difference from the admitted serial reference was
`3.907985046680551e-14`. It reused the completed serial reference, so this run
does not establish a full-sequence throughput ratio. Evidence is in
`artifacts/rbf-independent-root-state-full-sequence-CPU-v1-20261004` beneath the
external experiment root.

The independent checker now binds all 390 request commitments and 5,178 branch
state commitments for the two databases. It verifies the original default
decision scope and the candidate's explicit scope are identical. It reproduces
each branch commitment from the corresponding stored states before comparing
134,693 branch prediction pairs; full event audits, structural/raw tables,
component summaries and each database's own commit chain are also checked.
The maximum branch projection difference is `3.907985046680551e-14`.
The complete correspondence report is at
`artifacts/rbf-optimized-state-single-sequence-independent-v3-numpy126-20261004/structural-and-event-correspondence.json`.

The separate frozen fresh-history NumPy oracle completed all 56,389 states,
195 events and 57,297 chosen predictions from original observations. Its maximum
absolute error was `7.275957614183426e-12` at unchanged `atol=rtol=1e-8`.
The independent checker exited zero. The acceptance receipt is
`artifacts/rbf-optimized-state-single-sequence-independent-v3-numpy126-20261004/acceptance.json`,
SHA-256 `a26c27c62f3eba4b76640d2ece2cbee5d5e1a151242da59e8271da79170069f1`.
This accepts the optimized single CPU sequence, not CUDA or the full cohort.
The checker has no production tracker imports. Its 13 software tests include rejection of changed
hashes, states, prefix identities, expiry, future information, output identity,
and default decision scopes. They are not experiment completion evidence.

Failed checker attempts remain preserved. The first compared float-derived
state hashes as if they were invariant; the second treated default and explicit
request scopes as identical bytes; the third attempted byte-hash reproduction
under NumPy 2.5.3 instead of the producer's 1.26.4 runtime. The current attempt
uses the same frozen v3 checker under NumPy 1.26.4 for exact commitments and
retains the independent fresh-history arithmetic and fixed numerical tolerance.
No experimental replay was restarted and no comparison tolerance was relaxed.

`tools/event_track_v2x/prepare_branch_state_cuda_bundle.py` packages only the
frozen CUDA entry/source, accepted serial database, and associated independent
receipts. It refuses to package without complete optimized CPU admission,
checks source identity against the actual candidate, and independently hashes
every archive member after packing. The intended state-arithmetic measurement
requires one assigned CUDA device; a GPU4/GPU8 worker binding must not be
misreported as multi-device computation. Remote publication, collision-safe
single-device scheduling, actual CUDA execution and independent CUDA output
admission are still pending.

The CUDA transport code now also includes
`publish_branch_state_cuda_bundle.py` and `run_branch_state_cuda_candidate.py`.
The publisher checks the package and its complete local stream readback, creates
one deduplicated ClearML data task, then independently reads each cloud artifact
before admitting the publication. A failed or interrupted publication is
preserved for inspection. The producer verifies every extracted member and
the literal permitted command, requires real CUDA work and nonzero state
batches, records one actual device separately from the worker's visible cards,
and uploads partial outputs on failure. Fourteen input/publication tests passed
in the existing ClearML runtime; this is not a CUDA run.

`export_branch_state_commitment_witness.py` adds all branch projection values to
the candidate artifacts. It independently reproduces every branch hash in the
producer's NumPy runtime and records the values, so the receiving host can
verify those exact bytes and separately use the unchanged numerical tolerance.
It also verifies request and event commit chains, includes nonselected branches,
and imports no production tracker. Fourteen witness/correspondence software
tests passed. This witness does not replace fresh-history numerical validation.

Current freezes are `rbf-branch-state-CUDA-input-bundle-preparation-v2-witness-20261004`,
`rbf-branch-state-CUDA-transport-producer-v2-witness-20261004`, and
`rbf-branch-state-CUDA-commitment-witness-v1-20261004` beneath `source-freezes`.
The immutable CUDA execution source prepared earlier remains unchanged. The
complete CPU admission and local package creation/readback have completed.
The input archive has 156 members and 167,431,250 bytes, with SHA-256
`e73b7f3a968f2a3a0bc1db7f17b6a22d0cf11dae9fb8b5655646ceaf53010ac1`.
Automatic approval review rejected its proposed upload: it requires explicit
authorization for this code/database payload and the destination ClearML file
server `http://10.100.34.118:8081`. A concrete authorization question is pending;
no publication or GPU task was created by that rejected command.
Safe dispatch and the full independent CUDA receiver remain under preparation. No GPU
optimization experiment or 75–80% memory result has been claimed.

## Portable CUDA admission and dispatch preparation, 2026-10-05

`read_branch_state_cuda_outputs.py` reads the completed task's two artifacts,
checks the registered plan and producer bytes, and extracts exactly the declared
regular files. Partial output, duplicate members, path escapes, links, changed
bytes and conflicting device metadata fail admission. Its receipt grants only
cloud byte acceptance.

`accept_branch_state_cuda_outputs.py` requires that receipt and binds the exact
frozen implementation, input manifest, reference and actual CUDA device. It
checks every witness commitment, recomputes every selected and unselected
projection from the persisted states, compares complete event/audit structures
and invokes the unchanged fresh-history oracle for all states. It uses the
existing exact score/identity comparison and `atol=rtol=1e-8`; it does not permit
a cross-platform discrepancy to relax tolerance. Nineteen targeted receiver and
archive checks passed, including a wrong unselected branch whose producer hash
was self-consistent. Actual CUDA output has not yet been received.

`submit_branch_state_cuda_candidate.py` prefers the smallest eligible worker,
allows every GPU model, excludes L40S and checks whole-worker physical overlap
against running and queued work. It declares one actual compute device even
when the available worker binds four or eight cards. It preserves the existing
nine core experiment dispatches' priority, checks independently published input
bytes, deduplicates by the full input/producer plan, saves the created task before
configuration, and checks resource bindings again before enqueue. Ten scheduling
checks passed. Input publication authorization and free capacity are pending;
the dispatcher has created no task.
