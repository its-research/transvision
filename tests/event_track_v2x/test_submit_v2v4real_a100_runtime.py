import hashlib
from types import SimpleNamespace

import pytest

from tools.event_track_v2x import submit_v2v4real_a100_runtime as submit


def test_source_mismatch_prevents_even_network_lookup():
    with pytest.raises(ValueError, match="SHA-256"):
        submit.deploy(None, None, b"fixture", "0" * 64)


def test_existing_completed_or_active_probe_is_not_resubmitted():
    task = SimpleNamespace(get_tasks=lambda **kwargs: [SimpleNamespace(id="existing")])
    result = submit.deploy(task, None, b"fixture", hashlib.sha256(b"fixture").hexdigest())
    assert result["duplicate_not_created"] and result["task_ids"] == ["existing"]


@pytest.mark.parametrize("ready", [[], [{"family": "5090", "queue": "GPU4-5090"}],
                                    [{"family": "A100", "queue": "GPU8-A100"}]])
def test_no_task_created_without_four_a100_capacity(ready):
    task = SimpleNamespace(get_tasks=lambda **kwargs: [])
    with pytest.raises(ValueError, match="four-A100"):
        submit.deploy(task, lambda: ready, b"fixture", hashlib.sha256(b"fixture").hexdigest())
