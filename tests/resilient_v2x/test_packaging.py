import hashlib
import importlib.util
import shutil
import subprocess
import sys
import tarfile
import types
from pathlib import Path

import pytest
import yaml
from setuptools import Distribution, Extension
from setuptools.command.build_ext import build_ext


ROOT = Path(__file__).resolve().parents[2]
CONSTRAINTS = ROOT / "environments" / "resilient_v2x" / "constraints.txt"
ENVIRONMENT = ROOT / "environments" / "resilient_v2x" / "environment.yml"
DEV_REQUIREMENTS = ROOT / "environments" / "resilient_v2x" / "dev-requirements.txt"


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


def test_clean_sdist_egg_info_does_not_require_torch(tmp_path: Path) -> None:
    source_tree = tmp_path / "source"
    shutil.copytree(
        ROOT,
        source_tree,
        ignore=shutil.ignore_patterns(".git", ".venv", "build", "dist", "*.egg-info", "__pycache__"),
    )
    dist_dir = tmp_path / "dist"
    subprocess.run([sys.executable, "setup.py", "sdist", "--dist-dir", str(dist_dir)], cwd=source_tree, check=True, capture_output=True, text=True)
    with tarfile.open(next(dist_dir.glob("*.tar.gz"))) as tar:
        tar.extractall(tmp_path / "extracted")

    extracted = tmp_path / "extracted" / "transvision-0.1.0"
    egg_base = tmp_path / "egg-info"
    egg_base.mkdir()
    subprocess.run(
        [sys.executable, "setup.py", "egg_info", "--egg-base", str(egg_base)],
        cwd=extracted,
        env={"PATH": str(Path(sys.executable).parent), "PYTHONPATH": ""},
        check=True,
        capture_output=True,
        text=True,
    )
    assert (egg_base / "transvision.egg-info" / "PKG-INFO").is_file()


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


def test_environment_and_dev_inputs_have_exact_pins_and_parse_with_conda_lock() -> None:
    expected_environment_pip = {
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
    environment = yaml.safe_load(ENVIRONMENT.read_text())
    environment_pip = {
        name: version
        for entry in environment["dependencies"]
        if isinstance(entry, dict)
        for line in entry["pip"]
        for name, version in [line.split("==", maxsplit=1)]
    }
    dev_pins = {
        name: version
        for line in DEV_REQUIREMENTS.read_text().splitlines()
        if line
        for name, version in [line.split("==", maxsplit=1)]
    }

    parser_code = (
        "from pathlib import Path; "
        "from conda_lock.src_parser import pyproject_toml; "
        "pyproject_toml.get_lookup = lambda: {}; "
        "from conda_lock.src_parser.environment_yaml import parse_environment_file; "
        f"specification = parse_environment_file(Path({str(ENVIRONMENT)!r}), ['linux-64']); "
        "assert any(dependency.name == 'pypcd4' and dependency.version == '=1.4.3' "
        "for dependency in specification.dependencies['linux-64'])"
    )

    assert environment["channels"] == ["pytorch", "nvidia", "conda-forge"]
    assert "python=3.10.14" in environment["dependencies"]
    assert "cuda=11.8.0" in environment["dependencies"]
    assert "cuda-toolkit=11.8.0" in environment["dependencies"]
    assert environment_pip == expected_environment_pip
    assert dev_pins == {
        "pip": "23.3.2",
        "setuptools": "68.2.2",
        "wheel": "0.41.3",
        "packaging": "23.2",
        "numpy": "1.24.4",
        "pytest": "7.4.4",
        "jsonschema": "4.23.0",
        "PyYAML": "6.0.2",
        "zstandard": "0.22.0",
        "pypcd4": "1.4.3",
        "conda-lock": "2.5.7",
    }
    subprocess.run([sys.executable, "-c", parser_code], check=True)


def test_explicit_registration_loads_formatting_transform_once() -> None:
    code = r'''
import sys
import types
from pathlib import Path

root = Path.cwd()


class Registry:
    def __init__(self):
        self.module_dict = {}

    def register_module(self):
        def register(cls):
            self.module_dict[cls.__name__] = cls
            return cls
        return register


registry = Registry()


def package(name, path=None):
    module = types.ModuleType(name)
    if path is not None:
        module.__path__ = [str(path)]
    sys.modules[name] = module
    return module


package("transvision.dataset", root / "transvision" / "dataset")
package("transvision.dataset.transforms", root / "transvision" / "dataset" / "transforms")
package("transvision.dataset.v2x_dataset")
package("transvision.evaluation.metrics")
package("transvision.evaluation.metrics.dair_v2x_metric")
package("transvision.models")
package("transvision.models.data_preprocessors")
package("transvision.models.data_preprocessors.data_preprocessor")
package("transvision.models.dense_heads")
package("transvision.models.detectors")
package("transvision.models.hooks")
package("transvision.models.necks")
package("mmcv").BaseTransform = object
package("mmdet3d")
package("mmdet3d.datasets")
package("mmdet3d.datasets.transforms")
package("mmdet3d.datasets.transforms.formating").to_tensor = lambda value: value
package("mmdet3d.registry").TRANSFORMS = registry
structures = package("mmdet3d.structures")
structures.BaseInstance3DBoxes = object
structures.Det3DDataSample = object
structures.PointData = object
package("mmdet3d.structures.points").BasePoints = object
package("mmengine")
package("mmengine.structures").InstanceData = object

from transvision.register import register_resilient_v2x_modules

assert "Pack3DDetDAIRInputs" not in registry.module_dict
register_resilient_v2x_modules()
assert "Pack3DDetDAIRInputs" in registry.module_dict
registered = registry.module_dict["Pack3DDetDAIRInputs"]
register_resilient_v2x_modules()
assert registry.module_dict == {"Pack3DDetDAIRInputs": registered}
'''
    subprocess.run([sys.executable, "-c", code], cwd=ROOT, check=True)


def _load_setup_module(monkeypatch: pytest.MonkeyPatch):
    captured = {}
    monkeypatch.setattr("setuptools.setup", lambda **kwargs: captured.update(kwargs))
    module_name = "transvision_setup_under_test"
    sys.modules.pop(module_name, None)
    spec = importlib.util.spec_from_file_location(module_name, ROOT / "setup.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module, captured


def test_lazy_build_delegates_to_pytorch_compiler_lifecycle(monkeypatch: pytest.MonkeyPatch) -> None:
    module, _ = _load_setup_module(monkeypatch)
    calls = []
    instances = []

    class FakeBuildExtension(build_ext):
        def initialize_options(self) -> None:
            super().initialize_options()
            self.use_ninja = None
            instances.append(self)
            calls.append("delegate_initialize")

        def finalize_options(self) -> None:
            super().finalize_options()
            self.use_ninja = True
            if self.use_ninja:
                self.force = True
            calls.append("delegate_finalize")

        def _check_abi(self) -> None:
            calls.append("delegate_check_abi")

        def build_extensions(self) -> None:
            assert self.use_ninja is True
            calls.append("delegate_build_extensions")

        def run(self) -> None:
            assert self.force is True
            assert self.parallel == 7
            assert self.debug is True
            self._check_abi()
            self.build_extensions()
            calls.append("delegate_run")

    torch = types.ModuleType("torch")
    torch_utils = types.ModuleType("torch.utils")
    torch_cpp_extension = types.ModuleType("torch.utils.cpp_extension")
    torch_cpp_extension.BuildExtension = FakeBuildExtension
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "torch.utils", torch_utils)
    monkeypatch.setitem(sys.modules, "torch.utils.cpp_extension", torch_cpp_extension)
    extension = Extension("transvision.native", ["native.cpp"])
    monkeypatch.setattr(module, "build_extensions", lambda: [extension])

    distribution = Distribution(
        {
            "name": "transvision",
            "ext_modules": [],
            "cmdclass": {"build_ext": module.LazyBuildExtension},
        }
    )
    command = distribution.get_command_obj("build_ext")
    assert type(command) is module.LazyBuildExtension
    assert instances == []

    command.parallel = 7
    command.debug = True
    command.finalize_options()
    assert command.force is None
    assert command.parallel == 7
    assert command.debug is True
    command.run()

    assert len(instances) == 1
    assert type(instances[0]) is FakeBuildExtension
    assert instances[0].extensions == [extension]
    assert instances[0].distribution.ext_modules == [extension]
    assert calls == [
        "delegate_initialize",
        "delegate_finalize",
        "delegate_check_abi",
        "delegate_build_extensions",
        "delegate_run",
    ]


def test_sdist_contains_runtime_metadata_and_all_native_sources(tmp_path: Path) -> None:
    source_tree = tmp_path / "source"
    shutil.copytree(
        ROOT,
        source_tree,
        ignore=shutil.ignore_patterns(".git", ".venv", "build", "dist", "*.egg-info", "__pycache__"),
    )
    dist_dir = tmp_path / "dist"
    subprocess.run([sys.executable, "setup.py", "sdist", "--dist-dir", str(dist_dir)], cwd=source_tree, check=True, capture_output=True, text=True)
    archive = next(dist_dir.glob("*.tar.gz"))
    expected_files = {
        "transvision-0.1.0/environments/resilient_v2x/constraints.txt",
        "transvision-0.1.0/transvision/models/voxel/src/voxelization.cpp",
        "transvision-0.1.0/transvision/models/voxel/src/scatter_points_cpu.cpp",
        "transvision-0.1.0/transvision/models/voxel/src/scatter_points_cuda.cu",
        "transvision-0.1.0/transvision/models/voxel/src/voxelization_cpu.cpp",
        "transvision-0.1.0/transvision/models/voxel/src/voxelization_cuda.cu",
        "transvision-0.1.0/transvision/models/bev_pool/src/bev_pool.cpp",
        "transvision-0.1.0/transvision/models/bev_pool/src/bev_pool_cuda.cu",
    }
    with tarfile.open(archive) as tar:
        assert expected_files <= set(tar.getnames())
        tar.extractall(tmp_path / "extracted")

    extracted = tmp_path / "extracted" / "transvision-0.1.0"
    result = subprocess.run([sys.executable, "setup.py", "--name"], cwd=extracted, check=True, capture_output=True, text=True)
    assert result.stdout.strip() == "transvision"
