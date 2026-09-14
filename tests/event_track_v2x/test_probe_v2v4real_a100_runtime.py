"""Hardware/import gates; no actual ClearML submission or CUDA use."""
import subprocess
from types import SimpleNamespace

import pytest

from tools.event_track_v2x import probe_v2v4real_a100_runtime as probe


@pytest.mark.parametrize("names", [[], ["A100"] * 3, ["A100"] * 8,
                                  ["A100"] * 3 + ["RTX 5090"],
                                  ["V100"] * 4, ["A100"] * 3 + [None]])
def test_reject_other_allocations(names):
    with pytest.raises(ValueError, match="four worker-assigned A100"):
        probe.validate_devices(names)


def test_accepts_four_a100():
    probe.validate_devices(["NVIDIA A100-PCIE-40GB"] * 4)


def test_bad_allocation_rejected_before_tensor_operations():
    cuda = SimpleNamespace(device_count=lambda: 4,
                           get_device_name=lambda i: "A100" if i < 3 else "V100")
    with pytest.raises(ValueError):
        probe.collect(SimpleNamespace(cuda=cuda))


@pytest.mark.parametrize("stdout,code,expected", [
    ('RBF_IMPORT_PROBE {"importable":true,"error_type":null}\n', 0, True),
    ('RBF_IMPORT_PROBE {"importable":false,"error_type":"ModuleNotFoundError"}\n', 0, False),
    ("", -11, False),
    ("unrelated output", 0, False),
])
def test_import_results(stdout, code, expected):
    def run(args, **kwargs):
        assert args[1] == "-c" and kwargs["timeout"] == 20
        return SimpleNamespace(stdout=stdout, returncode=code)
    result = probe.inspect_import("spconv.pytorch", None, run=run)
    assert result["importable"] is expected


def test_timeout_is_diagnostic_failure():
    def run(*args, **kwargs):
        raise subprocess.TimeoutExpired("probe", 20)
    assert probe.inspect_import("opencood", None, run=run)["error_type"] == "ImportTimeout"


def test_no_arbitrary_module_execution():
    with pytest.raises(ValueError, match="predeclared"):
        probe.inspect_import("untrusted_plugin", None)
