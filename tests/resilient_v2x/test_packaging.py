import hashlib
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
CONSTRAINTS = ROOT / "environments" / "resilient_v2x" / "constraints.txt"


def test_setup_metadata_does_not_require_torch_import(tmp_path: Path) -> None:
    result = subprocess.run(
        [sys.executable, "setup.py", "--name"],
        cwd=ROOT,
        env={"PATH": str(Path(sys.executable).parent), "PYTHONPATH": ""},
        check=True,
        capture_output=True,
        text=True,
    )
    assert result.stdout.strip() == "transvision"


def test_setup_metadata_does_not_rewrite_tracked_version_file() -> None:
    version_file = ROOT / "transvision/version.py"
    before = hashlib.sha256(version_file.read_bytes()).hexdigest()
    subprocess.run(
        [sys.executable, "setup.py", "--version"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    after = hashlib.sha256(version_file.read_bytes()).hexdigest()
    assert after == before


def test_core_import_does_not_import_custom_ops() -> None:
    code = (
        "import sys; "
        "import transvision; "
        "assert 'transvision.models.bev_pool' not in sys.modules; "
        "assert 'transvision.models.voxel.voxel_layer' not in sys.modules"
    )
    subprocess.run([sys.executable, "-c", code], cwd=ROOT, check=True)


def test_constraints_pin_every_approved_runtime_package() -> None:
    expected = {
        "torch": "2.0.1",
        "torchvision": "0.15.2",
        "numpy": "1.24.4",
        "mmengine": "0.10.7",
        "mmcv": "2.1.0",
        "mmdet": "3.2.0",
        "mmdet3d": "1.3.0",
        "fvcore": "0.1.5.post20221221",
        "zstandard": "0.22.0",
        "pypcd4": "1.4.3",
        "pytest": "7.4.4",
        "jsonschema": "4.23.0",
        "PyYAML": "6.0.2",
        "conda-lock": "2.5.7",
    }
    pinned = {
        name: version
        for line in CONSTRAINTS.read_text().splitlines()
        if line and not line.startswith("#")
        for name, version in [line.split("==", maxsplit=1)]
    }
    assert pinned == expected
