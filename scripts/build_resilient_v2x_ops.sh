#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
build_root="$(mktemp -d "${TMPDIR:-/tmp}/resilient-v2x-ops.XXXXXX")"

cleanup() {
  rm -rf "${build_root}"
}
trap cleanup EXIT

mkdir -p "${build_root}/source/environments/resilient_v2x"
cp "${repo_root}/setup.py" "${repo_root}/setup.cfg" "${repo_root}/MANIFEST.in" "${build_root}/source/"
cp "${repo_root}/environments/resilient_v2x/constraints.txt" "${build_root}/source/environments/resilient_v2x/"
cp -a "${repo_root}/transvision" "${build_root}/source/transvision"
cd "${build_root}/source"

export FORCE_CUDA=1
python setup.py build_ext --inplace

python - <<'PY'
import importlib

for module_name in (
    "transvision.models.voxel.voxel_layer",
    "transvision.models.bev_pool.bev_pool_ext",
):
    module = importlib.import_module(module_name)
    if module.__name__ != module_name:
        raise SystemExit(f"custom op import mismatch: {module.__name__!r} != {module_name!r}")
PY

find transvision/models -type f -name '*.so' -print0 |
  while IFS= read -r -d '' extension; do
    destination="${repo_root}/${extension}"
    mkdir -p "$(dirname "${destination}")"
    cp "${extension}" "${destination}"
  done

export RESILIENT_V2X_REPO_ROOT="${repo_root}"
python - <<'PY'
import importlib
import os
import sys

sys.path.insert(0, os.environ["RESILIENT_V2X_REPO_ROOT"])
for module_name in (
    "transvision.models.voxel.voxel_layer",
    "transvision.models.bev_pool.bev_pool_ext",
):
    module = importlib.import_module(module_name)
    if module.__name__ != module_name:
        raise SystemExit(f"installed custom op mismatch: {module.__name__!r} != {module_name!r}")
PY
