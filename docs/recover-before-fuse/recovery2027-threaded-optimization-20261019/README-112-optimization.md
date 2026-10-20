# SSH112 recovery-off optimization candidates

Isolated candidates; frozen production sources and ClearML tasks are unchanged.

- `incremental_projection_candidate.py`: reuse correlated variants along one immutable prefix path; rebuild on branch changes.
- `prepared_threaded_factors*_candidate.py`, `threaded_factors.py`, `restricted_children.cpp`: packed preparation and native children computation, tested at 1/4/8 threads.
- `indexed_observation_slots_candidate.py`: equivalent slot lookup using existing indexes.
- `serial_lookup_candidate.py`: bounded SQL translation cache and prefix cache hit fast path preserving global and namespace recency.

The replay entrypoints require the admitted frozen `/runtime`, `/inputs`, `/raw`, `/context`, `/native` and read-only `/baseline` mounts. Single-event entrypoints also require `/sealed`. They are deployment diagnostics, not standalone training commands. `full-indexed-v2-8-112-submission.json` records the exact image, mounts and entry SHA. Do not rerun into an existing output directory. Build the native shared library from `restricted_children.cpp` using the existing frozen runtime contract; do not change BLAS thread settings or numeric tolerances.

## Verified scope

The full v2 run finished 452 events, exit 0. Events 0–313 match baseline prediction and full audit bytes. Events 314–451 have no baseline and remain unverified. See `full-indexed-v2-8-112-result.json`. This run did not include the later serial lookup candidate.

The serial lookup candidate passed 24 paired merge/rescore/reopen/empty-event controls. A single matched event313 measurement changed wall time from 17.106 to 15.598 seconds and CPU time from 16.556 to 14.175 seconds. Prediction, audit and final database SHA match. These are single-run measurements under shared host load, not a stable speedup claim. See `serial-lookup-evidence/`.

Event313 performs no native children calls. Eight workers are observed in other calls, but overall replay remains mostly serial. No full-cohort scientific acceptance or eight-core saturation is claimed.

Container validation scripts use the frozen mounted runtime. `test_incremental_projection_candidate.py` additionally requires the preserved frozen source inspection and runtime package referenced in that script; those archived experiment inputs are not bundled in this code commit.
