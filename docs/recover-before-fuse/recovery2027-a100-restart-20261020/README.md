# A100 rerun against the designated 112 result

The user authorized stopping the two old A100 validations, preserving outputs, and rerunning the 112-validated version with eight native threads. The subsequent instruction designates the completed 112 result as the reference. No slow original baseline arm is recomputed.

- Old projection: `7c9beb09c18e48999fd94e0047c81b74`, stopped.
- Old combined: `4623f764376546d7a82794405321e66a`, stopped.
- Preservation: `2696dd26573840eba324952e1eee1195`, completed. Both archives include raw SQLite and journals; member hashes were checked, then complete uploaded archives were independently read back on SSH112. See `preservation-readback.json`. This preserves partial results and is not SQLite recovery or scientific acceptance.
- New projection: `85a82005271a4faeb2f01a9346736c42`, A100 GPU0–3.
- New combined: `d3c4da00634f4d6089fd21d26f779b0a`, A100 GPU4–7.

Both reruns use the same actual 112 runtime (130 files) and six optimization files. `112-source-readback.json` records no mismatches. Original container, model, admitted inputs, plan, budgets and tolerances are retained. The child uses eight native threads and CUDA is disabled; OMP and OpenBLAS remain one thread as in the 112 run. Eight configured workers do not imply eight-core saturation. The later serial lookup experiment is excluded because it was not part of the completed 112 full run.

Before replay, each task verifies the immutable package, compressed and uncompressed reference hashes, physical A100 binding and absence of compute processes, then runs the native 1/4/8-thread equivalence controls. The fresh replay compares prediction and full audit bytes at all 452 events, reports wall time and process CPU time, and uploads its outputs after child exit. Original numerical tolerance is not relaxed; byte equality is required for this reference comparison.

The 112 run completed all 452 events. Events 0–313 matched the original baseline; the remaining 138 events are user-designated reference results. Passing this rerun means consistency with 112, not independent original-baseline qualification for all 452 events, GPU model-forward acceptance, or full-cohort scientific acceptance.

The authoritative deployed package is `reference-package.json` / `package-upload-final.json`, with bootstrap `reference_bootstrap.py`. Earlier paired package preparations and intermediate uploaded packages were not enqueued. `submission.json` preserves old/new identity and exact package provenance. `latest-live-progress.json` is a timestamped snapshot, not a permanent status.

The initial preservation task `a208c2cb9824482fb2052a063be13de2` failed before execution because its entry source was not embedded; its failed record is retained. The corrected preservation task above completed successfully before the old tasks were stopped.

## Independent terminal readback

Wait for each exact new task to reach `completed` and finish artifact uploads.
Record the task ID, terminal state, registered artifact sizes/hashes and download
time before consuming `candidate.sqlite` and `result.json`. Use authenticated
full-byte downloads, preferably on freshly admitted SSH112. Do not consume the
partial local `reference452.jsonl.gz`; use the pinned full reference described in
`submission.json` and verify its compressed SHA before comparison.

Run `verify_downloaded_reference_parity.py` with `--database`, `--reference`,
`--producer-result`, and `--artifact-sha256` taken from that task's registered
artifact. It opens SQLite read-only, rejects pending transaction sidecars,
checks integrity, compares every stored event identity, prediction and full
audit byte string, rejects missing or extra events, checks the pinned plain
reference SHA, and rechecks that the database did not change. Save its output
with the task-specific download receipt. A producer's success report alone
does not satisfy this gate.

The six synthetic controls in `test_reference_readback.py` pass, including
corruption at events 400 and 451, missing/extra rows and wrong event identity.
These controls test the reader and do not constitute experimental acceptance.
The two real task readbacks remain pending until terminal artifacts exist.

`collect_terminal_reference.py` supplies a local-disk fallback when SSH112
storage admission is unavailable. It refuses either task until `completed`,
requires all registered artifact sizes and SHA256 hashes, preserves failed
downloads as `.partial`, reserves an 8 GiB local free-space floor, checks that
the terminal artifact manifest remains unchanged, and invokes the independent
reader above. Use a fresh output directory for each attempt. It does not
delete caches or alter the SSH112 resource contract. Four download controls
(valid, truncated, oversized, corrupt) passed; a live running task was rejected
before creating its output directory. Real terminal readback is still pending.

Performance conclusions require a separate isolated measurement: the existing
event313 measurements are about 37 seconds and about one CPU core, and another
training workload was observed on this host. Reference parity does not establish
eight-core utilization or a same-resource speed comparison.
