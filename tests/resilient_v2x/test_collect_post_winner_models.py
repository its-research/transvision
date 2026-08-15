from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from tools.resilient_v2x import collect_post_winner_models as collector


def _evidence() -> dict[str, object]:
    value = {
        "schema_version": 1,
        "table_vi": [],
    }
    detached = dict(value)
    import hashlib

    value["seal_sha256"] = hashlib.sha256(
        collector._canonical(detached).encode("utf-8")
    ).hexdigest()
    return value


def test_read_accepts_sealed_regular_json(tmp_path) -> None:
    path = tmp_path / "evidence.json"
    value = _evidence()
    path.write_text(json.dumps(value), encoding="utf-8")

    assert collector._read(path) == value


def test_read_rejects_mutated_sealed_json(tmp_path) -> None:
    path = tmp_path / "evidence.json"
    value = _evidence()
    value["schema_version"] = 2
    path.write_text(json.dumps(value), encoding="utf-8")

    with pytest.raises(RuntimeError, match="seal mismatch"):
        collector._read(path)


def test_model_requires_unique_task_bound_output_model() -> None:
    model = SimpleNamespace(id="a" * 32, task="b" * 32, url="http://model")
    task = SimpleNamespace(
        id="b" * 32,
        get_models=lambda: {"output": [model]},
    )

    assert collector._model(task, "a" * 32, "http://model") is model

    model.task = "c" * 32
    with pytest.raises(RuntimeError, match="identity drifted"):
        collector._model(task, "a" * 32, "http://model")
