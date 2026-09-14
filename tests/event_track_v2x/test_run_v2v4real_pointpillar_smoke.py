from io import BytesIO
import hashlib
from pathlib import Path
import os
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from tools.event_track_v2x import run_v2v4real_pointpillar_smoke as smoke
from tools.event_track_v2x import submit_v2v4real_a100_runtime as submit


@pytest.mark.parametrize("count,name", [(0, "A100"), (3, "A100"), (8, "A100"), (4, "RTX 5090"), (4, "V100")])
def test_four_a100_gate(count, name):
    fake = SimpleNamespace(cuda=SimpleNamespace(device_count=lambda: count, get_device_name=lambda _: name))
    with pytest.raises(ValueError):
        smoke.four_a100(fake)


def test_four_a100_is_allowed():
    fake = SimpleNamespace(cuda=SimpleNamespace(device_count=lambda: 4, get_device_name=lambda _: "NVIDIA A100-PCIE-40GB"))
    assert len(smoke.four_a100(fake)) == 4


def test_fixture_includes_point_truncation():
    cloud = smoke.fixture_cloud()
    assert cloud.shape == (37, 4) and cloud.dtype == np.float32
    expected = {"voxel_coords": np.array([[0, 100, 176], [0, 102, 178], [0, 105, 203]]),
                "voxel_num_points": np.array([32, 1, 1]),
                "voxel_features": np.zeros((3, 32, 4), dtype=np.float32)}
    expected["voxel_features"][0] = cloud[:32]
    expected["voxel_features"][1, 0] = cloud[35]
    expected["voxel_features"][2, 0] = cloud[36]
    smoke.check_fixture_voxels(expected, cloud)
    expected["voxel_num_points"][0] = 35
    with pytest.raises(ValueError, match="voxel_num_points"):
        smoke.check_fixture_voxels(expected, cloud)


def test_pins_cover_model_and_real_preprocessor_without_dataset_loader():
    assert len(smoke.SOURCE_PINS) == 14
    assert "LICENSE" in smoke.SOURCE_PINS
    assert "opencood/models/point_pillar.py" in smoke.SOURCE_PINS
    assert "opencood/data_utils/pre_processor/sp_voxel_preprocessor.py" in smoke.SOURCE_PINS
    assert not any("/datasets/" in path for path in smoke.SOURCE_PINS)
    assert "spconv-cu126==2.3.8" in smoke.PACKAGES
    assert "cumm-cu126==0.7.11" in smoke.PACKAGES


def test_download_failure_does_not_leave_executable_source(tmp_path, monkeypatch):
    monkeypatch.setattr(smoke, "SOURCE_PINS", {"module.py": hashlib.sha256(b"expected").hexdigest()})
    def opener(url, timeout):
        assert url.startswith(smoke.BASE_URL) and timeout == 45
        return BytesIO(b"bad response")
    with pytest.raises(ValueError, match="downloaded source"):
        smoke.fetch_sources(tmp_path / "source", opener=opener)
    assert not (tmp_path / "source/module.py").exists()


def test_download_verify_and_refuse_overwrite(tmp_path, monkeypatch):
    monkeypatch.setattr(smoke, "SOURCE_PINS", {"opencood/model.py": hashlib.sha256(b"expected").hexdigest()})
    output = tmp_path / "source"
    smoke.fetch_sources(output, opener=lambda *a, **k: BytesIO(b"expected"))
    smoke.verify_sources(output)
    with pytest.raises(FileExistsError):
        smoke.fetch_sources(output, opener=lambda *a, **k: BytesIO(b"expected"))
    (output / "opencood/model.py").write_bytes(b"changed")
    with pytest.raises(ValueError, match="SHA-256"):
        smoke.verify_sources(output)


def test_source_symlink_refused(tmp_path, monkeypatch):
    monkeypatch.setattr(smoke, "SOURCE_PINS", {"model.py": hashlib.sha256(b"expected").hexdigest()})
    original = tmp_path / "original.py"; original.write_bytes(b"expected")
    (tmp_path / "model.py").symlink_to(original)
    with pytest.raises(ValueError, match="ordinary"):
        smoke.verify_sources(tmp_path)


def test_arbitrary_probe_cannot_be_deployed():
    with pytest.raises(ValueError, match="predeclared"):
        submit.deploy(None, None, b"", "", probe_kind="arbitrary")


def test_pointpillar_submission_is_four_a100_no_training():
    calls = {}
    class Task:
        TaskTypes = SimpleNamespace(testing="testing")
        id = "fixture"; status = "created"
        @classmethod
        def get_tasks(cls, **kwargs): return []
        @classmethod
        def create(cls, **kwargs): calls["create"] = kwargs; return cls()
        def add_tags(self, tags): calls["tags"] = tags
        def set_script(self, **kwargs): calls["script"] = kwargs
        def set_packages(self, packages): calls["packages"] = packages
        def set_base_docker(self, image, **kwargs): calls["image"] = image
        def set_parameters(self, params): calls["parameters"] = params
        @classmethod
        def enqueue(cls, task, queue_name): calls["queue"] = queue_name; task.status = "queued"
        def get_status(self): return self.status
    source = b"fixture"
    ready = lambda: [{"family": "A100", "queue": "GPU4-A100"}]
    result = submit.deploy(Task, ready, source, hashlib.sha256(source).hexdigest(), probe_kind="pointpillar")
    assert calls["script"]["entry_point"] == Path(smoke.__file__).name
    assert calls["parameters"]["required_gpu_count"] == 4
    assert calls["parameters"]["parameter_training"] is False
    assert calls["queue"] == "GPU4-A100" and result["status"] == "queued"
    assert result["pointpillar_smoke_submitted"] and not result["runtime_inventory_submitted"]
    assert "pointpillar-smoke" in calls["create"]["task_name"]


def test_hash_pinned_official_pointpillar_cpu_forward():
    """Optional source integration check, explicitly not the A100 acceptance."""
    location = os.environ.get("V2V4REAL_POINTPILLAR_REFERENCE_ROOT")
    if not location:
        pytest.skip("set the explicitly downloaded official reference source root")
    import torch
    import yaml
    root = Path(location).resolve()
    smoke.verify_sources(root)
    assert not any(k == "opencood" or k.startswith("opencood.") for k in sys.modules)
    original_path = list(sys.path)
    sys.path.insert(0, str(root))
    try:
        from opencood.models.point_pillar import PointPillar
        from opencood.hypes_yaml.yaml_utils import load_point_pillar_params
        config = yaml.safe_load((root / "opencood/hypes_yaml/point_pillar_late_fusion.yaml").read_text())
        config = load_point_pillar_params(config)
        cloud = smoke.fixture_cloud()
        features = np.zeros((3, 32, 4), dtype=np.float32)
        features[0] = cloud[:32]; features[1, 0] = cloud[35]; features[2, 0] = cloud[36]
        processed = {"voxel_features": torch.from_numpy(features),
                     "voxel_coords": torch.tensor([[0, 0, 100, 176], [0, 0, 102, 178], [0, 0, 105, 203]], dtype=torch.int32),
                     "voxel_num_points": torch.tensor([32, 1, 1], dtype=torch.int32)}
        original = {key: value.clone() for key, value in processed.items()}
        torch.manual_seed(1337)
        model = PointPillar(config["model"]["args"]).cpu().eval()
        with torch.inference_mode():
            first = model({"processed_lidar": processed})
            repeated = model({"processed_lidar": processed})
        assert set(first) == {"psm", "rm"}
        for key, shape in {"psm": (1, 2, 50, 88), "rm": (1, 14, 50, 88)}.items():
            assert tuple(first[key].shape) == shape
            assert torch.isfinite(first[key]).all()
            assert torch.equal(first[key], repeated[key])
        assert all(torch.equal(value, original[key]) for key, value in processed.items())
        smoke.verify_sources(root)
    finally:
        sys.path[:] = original_path
        for key in list(sys.modules):
            if key == "opencood" or key.startswith("opencood."):
                del sys.modules[key]
