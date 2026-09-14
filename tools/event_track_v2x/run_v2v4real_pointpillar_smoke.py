#!/usr/bin/env python3
"""Isolated official PointPillar/voxel/NMS smoke on four A100s, without GT.

This is a runtime prerequisite, not a pretrained detector or paper result.
Official sources are downloaded into this task's private directory, hash-checked
and left unchanged. They are not embedded in the ClearML script or artifacts.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
from urllib.request import urlopen
import venv

COMMIT = "5a821e13753bafc611f95c47bc1a306acdcb0f7c"
BASE_URL = f"https://raw.githubusercontent.com/ucla-mobility/V2V4Real/{COMMIT}/"
SOURCE_PINS = {
    "LICENSE": "054296a83ccba3cb068644f6258191112a72387d3edaeda7743f13bb2469ebd8",
    "opencood/models/point_pillar.py": "21f36cce8ed105c8a125d147534c3d85cb7d95f92a46f1e8f1041a40fb821af5",
    "opencood/hypes_yaml/point_pillar_late_fusion.yaml": "138c4ad3508fdd7061f5b290c83ad8f0772fea33f0af0e2be56330423a3c92d1",
    "opencood/models/sub_modules/pillar_vfe.py": "38c94b45f1c7d3e10be4c0d927a5296106667e9908660cb5e202f80f72345fdc",
    "opencood/models/sub_modules/point_pillar_scatter.py": "43631cabbc046c1812f8e101675dd6ca825ad6bfc0e6a863291c0fe61c9d5ea9",
    "opencood/models/sub_modules/base_bev_backbone.py": "ee89f2501c83c355aea09066f5f41b509317a9f40f8b6aa2276289cf9c4c9223",
    "opencood/models/sub_modules/downsample_conv.py": "043711a14256db94b733d04bdc8839362c09fabff4f5383041db14a71a86de09",
    "opencood/data_utils/pre_processor/sp_voxel_preprocessor.py": "2b9a20da2138d3fc2c37fb98900ebbdc31739fc8b39135162532ef4e64b974d0",
    "opencood/data_utils/pre_processor/base_preprocessor.py": "1234e467c3dc16d7712fd414291024262b04fb6e7ddaf174f1be94d2529761a2",
    "opencood/utils/pcd_utils.py": "8d5f25b47bafab367d688deb9ba2a4251a37a6e5d4a3531e78ed8828d0cf19ab",
    "opencood/utils/box_utils.py": "961914a1291f57e86d9a7416e7318a3e911f0de8fb98a0c4492e2a9d5785de86",
    "opencood/utils/transformation_utils.py": "09a5b0f3ce37a95d9d084de2edb5ea2c72263da0cbe965344416e0cb04a89741",
    "opencood/utils/common_utils.py": "12d89510b0e8f2c29def3b2cf1963ae790e226fa056d8d7cc0b0f70cd9d5be21",
    "opencood/hypes_yaml/yaml_utils.py": "2321d582a737659c4642c96a73f13b06678134ad136519959ec6292670213b6e",
}
# Direct runtime dependencies only; pip's report seals the resolved transitives.
PACKAGES = ("numpy==1.26.4", "scipy==1.14.1", "PyYAML==6.0.2",
            "spconv-cu126==2.3.8", "cumm-cu126==0.7.11",
            "open3d==0.19.0", "shapely==2.0.7")


def four_a100(torch):
    if torch.cuda.device_count() != 4:
        raise ValueError("exactly four assigned A100 GPUs required")
    names = [torch.cuda.get_device_name(i) for i in range(4)]
    if any("A100" not in name for name in names):
        raise ValueError("non-A100 device assignment rejected")
    return names


def verify_sources(root):
    root = Path(root)
    for name, digest in SOURCE_PINS.items():
        path = root / name
        if not path.is_file() or any(p.is_symlink() for p in (path, *path.parents)):
            raise ValueError(f"ordinary pinned source required: {name}")
        if path.stat().st_size > 2 * 1024 * 1024 or hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise ValueError(f"source SHA-256 mismatch: {name}")


def fetch_sources(root, *, opener=urlopen):
    root = Path(root)
    root.mkdir(exist_ok=False)
    for name, digest in SOURCE_PINS.items():
        with opener(BASE_URL + name, timeout=45) as response:
            payload = response.read(2 * 1024 * 1024 + 1)
        if len(payload) > 2 * 1024 * 1024 or hashlib.sha256(payload).hexdigest() != digest:
            raise ValueError(f"downloaded source SHA-256 mismatch: {name}")
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as stream:
            stream.write(payload)
    verify_sources(root)


def fixture_cloud():
    """Fixed no-data fixture: truncation, repeated cells and three distinct cells."""
    import numpy as np
    close = [[0.05, 0.05, 0.0, i / 40.0] for i in range(35)]
    return np.asarray(close + [[1.0, 1.0, 0.1, 0.2], [11.0, 2.0, 0.2, 0.4]], dtype=np.float32)


def check_fixture_voxels(result, cloud):
    import numpy as np
    coords = np.asarray([[0, 100, 176], [0, 102, 178], [0, 105, 203]], dtype=np.int32)
    counts = np.asarray([32, 1, 1], dtype=np.int32)
    features = np.zeros((3, 32, 4), dtype=np.float32)
    features[0] = cloud[:32]
    features[1, 0] = cloud[35]
    features[2, 0] = cloud[36]
    for key, expected in (("voxel_coords", coords), ("voxel_num_points", counts),
                          ("voxel_features", features)):
        if not np.array_equal(result[key], expected):
            raise ValueError(f"fixture voxel result differs: {key}")


def child_run(root):
    import numpy as np
    import torch
    import yaml
    root = Path(root).resolve()
    if Path(sys.prefix).resolve() != root / "venv" or sys.prefix == sys.base_prefix:
        raise ValueError("smoke must run inside its task-private venv")
    names = four_a100(torch)
    for spec in PACKAGES:
        package, expected = spec.split("==")
        if importlib.metadata.version(package) != expected:
            raise ValueError(f"runtime version mismatch: {package}")
    source = root / "official-source"
    verify_sources(source)
    if "opencood" in sys.modules:
        raise ValueError("refuse a pre-imported, unverified OpenCOOD")
    sys.path.insert(0, str(source))
    from opencood.hypes_yaml.yaml_utils import load_point_pillar_params
    from opencood.data_utils.pre_processor.sp_voxel_preprocessor import SpVoxelPreprocessor
    from opencood.models.point_pillar import PointPillar
    from opencood.utils import box_utils
    import opencood.models.point_pillar as model_module
    if Path(model_module.__file__).resolve() != source / "opencood/models/point_pillar.py":
        raise ValueError("unexpected model import origin")
    # Safe YAML plus an explicitly selected fixed helper, never YAML-driven eval.
    config = yaml.safe_load((source / "opencood/hypes_yaml/point_pillar_late_fusion.yaml").read_text())
    config = load_point_pillar_params(config)
    preprocessor = SpVoxelPreprocessor(config["preprocess"], train=False)
    cloud = fixture_cloud()
    processed = preprocessor.preprocess(cloud)
    check_fixture_voxels(processed, cloud)
    batch = preprocessor.collate_batch([processed])
    # Same geometry, lower score -> suppressed; distant geometry -> kept.
    square = [[-1., -1.], [-1., 1.], [1., 1.], [1., -1.]]
    boxes = torch.tensor([square, square, [[x + 10., y] for x, y in square]])
    keep = box_utils.nms_rotated(boxes, torch.tensor([0.9, 0.8, 0.7]), 0.15).tolist()
    if keep != [0, 2]:
        raise ValueError("official rotated-NMS fixture mismatch")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    reports, reference = [], None
    for index, name in enumerate(names):
        torch.manual_seed(1337)
        model = PointPillar(config["model"]["args"]).to(f"cuda:{index}").eval()
        parameter_count = sum(p.numel() for p in model.parameters())
        device_batch = {k: v.to(f"cuda:{index}") for k, v in batch.items()}
        begin = time.perf_counter()
        with torch.inference_mode():
            outputs = model({"processed_lidar": device_batch})
        torch.cuda.synchronize(index)
        arrays = {key: value.cpu().numpy() for key, value in outputs.items()}
        expected_shapes = {"psm": (1, 2, 50, 88), "rm": (1, 14, 50, 88)}
        if set(arrays) != set(expected_shapes) or any(
                arrays[k].shape != shape or not np.isfinite(arrays[k]).all()
                for k, shape in expected_shapes.items()):
            raise ValueError("official PointPillar head shape or finite-value check failed")
        if reference is None:
            reference = arrays
        differences = {k: float(np.max(np.abs(arrays[k] - reference[k]))) for k in arrays}
        if any(not np.allclose(arrays[k], reference[k], atol=1e-6, rtol=1e-5) for k in arrays):
            raise ValueError("cross-device smoke mismatch")
        reports.append(dict(index=index, name=name, parameter_count=parameter_count,
            elapsed_seconds=time.perf_counter() - begin, max_abs_difference_to_first=differences,
            head_shapes={k: list(a.shape) for k, a in arrays.items()},
            head_sha256={k: hashlib.sha256(a.tobytes()).hexdigest() for k, a in arrays.items()}))
        del model, device_batch, outputs
        torch.cuda.empty_cache()
    verify_sources(source)
    versions = {}
    for dist in importlib.metadata.distributions():
        if dist.metadata.get("Name"):
            versions.setdefault(dist.metadata["Name"], dist.version)
    report = dict(kind="v2v4real_official_pointpillar_a100_smoke_v1", official_commit=COMMIT,
        source_pins=SOURCE_PINS, source_edits=False, devices=reports,
        python=sys.version.split()[0], torch=str(torch.__version__), cuda=torch.version.cuda,
        direct_requirements=list(PACKAGES), installed_distributions=versions,
        voxel_fixture=dict(points=len(cloud), voxels=3, counts=[32, 1, 1], exact_match=True),
        rotated_nms_keep=keep, random_weight_seed=1337, random_weights=True,
        dataset_read=False, checkpoint_loaded=False, detector_forward_verified=True,
        ddp_verified=False, parameter_training=False, paper_eligible=False,
        full_detection_pipeline_verified=False, original_paper_runtime_reproduced=False)
    with (root / "smoke-report.json").open("x") as stream:
        json.dump(report, stream, sort_keys=True)
    print("RBF_POINTPILLAR_CHILD_COMPLETE", flush=True)


def bootstrap():
    from clearml import Task
    import torch
    task = Task.init(project_name="Thesis/Recover-Before-Fuse/Training",
                     task_name="V2V4Real isolated PointPillar smoke",
                     reuse_last_task_id=False, auto_connect_frameworks=False,
                     auto_connect_arg_parser=False)
    four_a100(torch)
    watched = ("numpy", "torch", "PyYAML", "scipy")
    before = {p: importlib.metadata.version(p) for p in watched}
    root = Path(tempfile.mkdtemp(prefix="rbf-v2v4real-isolated-", dir=Path.cwd())).resolve()
    print("RBF_POINTPILLAR_BOOTSTRAP " + json.dumps(dict(runtime_root=str(root), stage="create-venv")), flush=True)
    venv.EnvBuilder(with_pip=True, system_site_packages=True).create(root / "venv")
    python = str(root / "venv/bin/python")
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", PYTHONNOUSERSITE="1",
               MPLCONFIGDIR=str(root / "matplotlib"), XDG_CACHE_HOME=str(root / "cache"),
               OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
    pip_env = {k: v for k, v in env.items() if not k.startswith("PIP_")}
    pip_env["PIP_CONFIG_FILE"] = os.devnull
    subprocess.run([python, "-m", "pip", "install", "--disable-pip-version-check",
                    "--index-url", "https://pypi.org/simple", "--only-binary=:all:",
                    "--report", str(root / "pip-report.json"), *PACKAGES],
                   env=pip_env, check=True, timeout=1200)
    print("RBF_POINTPILLAR_BOOTSTRAP " + json.dumps(dict(stage="fetch-pinned-official-source")), flush=True)
    fetch_sources(root / "official-source")
    subprocess.run([python, str(Path(__file__).resolve()), "--child-root", str(root)],
                   env=env, check=True, timeout=300)
    after = {p: importlib.metadata.version(p) for p in watched}
    if before != after:
        raise RuntimeError("base environment changed during isolated setup")
    report = json.loads((root / "smoke-report.json").read_text())
    report["base_watched_versions_unchanged"] = before == after
    report["base_watched_versions"] = before
    report["pip_report_sha256"] = hashlib.sha256((root / "pip-report.json").read_bytes()).hexdigest()
    pip_report = json.loads((root / "pip-report.json").read_text())
    report["pip_resolution"] = [dict(name=item["metadata"]["name"], version=item["metadata"]["version"],
                                     download_info=item["download_info"])
                                for item in pip_report["install"]]
    print("RBF_POINTPILLAR_SMOKE " + json.dumps(report, sort_keys=True), flush=True)
    task.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--child-root", type=Path)
    args = parser.parse_args()
    if args.child_root:
        child_run(args.child_root)
    else:
        bootstrap()
