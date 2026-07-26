# Task 2 Completion Report

## Status

`COMPLETE` — the unified Linux runtime lock, bootstrap lock, pinned Docker
definition, environment capture/validation tools, custom-op build check, recovery
adapter, and regression coverage are complete.

The Linux CUDA runtime and custom extensions were not executed on this macOS
development host. Those controlled-runtime checks remain explicitly
`not_executed`; they were not reported as passed.

## Delivered files

- `environments/resilient_v2x/environment-linux-64.lock.yml`
- `environments/resilient_v2x/bootstrap-environment.yml`
- `environments/resilient_v2x/bootstrap-linux-64.explicit.txt`
- `environments/resilient_v2x/Dockerfile`
- `environments/resilient_v2x/environment-manifest.schema.json`
- `environments/resilient_v2x/runtime-pip-report.json`
- `tools/resilient_v2x/capture_environment.py`
- `tools/resilient_v2x/check_environment.py`
- `tools/resilient_v2x/seed_runtime_lock.py`
- `scripts/build_resilient_v2x_ops.sh`
- `tests/resilient_v2x/test_environment.py`

## Runtime-lock result

The approved adapter ran the normal pinned child command:

```text
python -m conda_lock lock
  --file environments/resilient_v2x/environment.yml
  --platform linux-64
  --lockfile <temporary-seed>
  --micromamba --no-mamba
  --with-cuda 11.8
  --log-level INFO
```

The child command overwrote the temporary seed with the complete unified lock.
The outer adapter only audited and atomically materialized the result; it did not
modify the generated final lock.

Independent audit results:

```text
runtime lock SHA-256:
d8137aeaf3aa216691938f487c70b0c90ff047a8d52a4a6ba1575fb9e4ebdf9f

runtime artifacts: 338
pip artifacts:     190
conda artifacts:   148
defects:           0

runtime content hash:
3c298ab9a83173125a41160c1cf049ec06790ee4cd7d7ef75a4681796c89f3e4
```

All 190 pip records exactly match the committed native Linux report by canonical
name, version, HTTPS URL, and SHA-256. Every conda record has cryptographic
hashes. The required conda pins resolve exactly:

```text
python=3.10.14
pip=23.3.2
cuda=11.8.0
cuda-toolkit=11.8.0
```

The committed pip report SHA-256 is:

```text
657151ac1fa38d384c9afaf8e61774263af41aa134deea6f476455281fdf014d
```

## Recovery evidence

The adapter is pinned to Python 3.10 and `conda-lock==2.5.7`. Focused RED tests
reproduced each observed conda-lock 2.5.7 compatibility defect before its bounded
child-side fix:

1. the stale manylinux ceiling rejected three report-selected wheels using
   `manylinux_2_31` or `manylinux_2_34`;
2. Poetry discarded 17 audited seed artifacts through skipped/uninstall or
   conda-installed operations;
3. the environment parser appended `pip=*` after explicit `pip=23.3.2`, and
   last-wins aggregation resolved pip 26.1.2.

The final child compatibility shim:

- permits only the two audited newer manylinux tags;
- adds only seed-snapshot records entirely missing from resolver output;
- fails on any same-name version, URL, or hash drift instead of overwriting it;
- preserves the one explicit conda-side pip pin from the read-only committed
  environment snapshot;
- keeps the final content-hash model identical to the repaired child spec.

No package pin, artifact URL, or artifact hash was synthesized or changed.
Failure/conflict tests verify that no final output is published before the full
audit and that conflicting existing outputs are never overwritten.

## Bootstrap and container audit

Bootstrap lock:

```text
SHA-256:
02dff88aeaedf3416efd8877c08fa28edf90ff8cf648dbcc1dab06d10bd29461

artifacts: 101
defects:   0
```

Every bootstrap artifact has an HTTPS URL and cryptographic hash. The direct
pins are exactly Python 3.10.14, pip 23.3.2, and conda-lock 2.5.7.

The Dockerfile uses the verified immutable linux/amd64 manifests:

```text
mambaorg/micromamba:2.8.1@
sha256:79284aa2949ac9555eca7e975ceb4ecefefc5964a2d4fdd4e181ca8eaccf347e

nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04@
sha256:bd746eb3b9953805ebe644847a227e218b5da775f47007c69930569a75c9ad7d
```

`docker build --check` completed with no warnings. The Docker check directive
suppresses only `FromPlatformFlagConstDisallowed`, because the Task 2 contract
requires both stages to be fixed explicitly to `linux/amd64`.

## Verification

Full Task 1 and Task 2 suite:

```text
98 passed
```

The suite initially exposed conda-lock 2.5.7 polluting the shared test process
with Poetry's vendored `jsonschema` and subprocess backport. The test module now
loads the runtime validator first and restores the standard-library
`subprocess.run`; production behavior and dependency pins are unchanged.

Other checks:

```text
final runtime-lock audit: PASS
independent runtime/bootstrap audit: PASS
development checker: exit 0, no blocking mismatches
bash -n scripts/build_resilient_v2x_ops.sh: PASS
docker build --check -f environments/resilient_v2x/Dockerfile .: PASS
git diff --check: PASS
```

The development checker reported these checks as unavailable rather than passed:

```text
controlled_linux_runtime
runtime_packages
cuda_runtime
custom_ops
determinism_runtime
container_image_digest
dair_v2x_data
```

## Controlled-runtime handoff

The following remain for an approved Linux host with Docker, NVIDIA Container
Toolkit, CUDA hardware, and the DAIR-V2X data:

```bash
docker build --platform linux/amd64 \
  -f environments/resilient_v2x/Dockerfile \
  -t resilient-v2x:test .

docker run --rm --platform linux/amd64 --gpus all \
  resilient-v2x:test \
  python tools/resilient_v2x/check_environment.py \
  --mode controlled --require-cuda --require-custom-ops
```

Until both commands pass, the controlled reproduction gate remains open.
