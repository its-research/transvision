from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
MODULE_NAMES = (
    "clearml_1337_leaderboard",
    "clearml_formal_comparability_audit",
    "clearml_formal_candidate_selector",
)


def _load(name: str) -> object:
    path = ROOT / "tools/resilient_v2x" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"{name}_pin_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("module_name", MODULE_NAMES)
def test_successor_pins_are_all_or_none_and_typed(module_name: str) -> None:
    module = _load(module_name)
    empty = argparse.Namespace()
    assert module._sealed_successor_pins(empty) == {}

    partial = argparse.Namespace(
        evaluation_plan_amendment_producer_task_id="a" * 32
    )
    with pytest.raises(ValueError, match="all amended plan"):
        module._sealed_successor_pins(partial)

    complete = argparse.Namespace()
    for name, kind in module.SEALED_SUCCESSOR_PIN_ARGS:
        setattr(complete, name, "a" * (32 if kind == "task_id" else 64))
    pins = module._sealed_successor_pins(complete)
    assert set(pins) == {
        f"Args/{name}" for name, _kind in module.SEALED_SUCCESSOR_PIN_ARGS
    }
    assert pins["Args/evaluation_plan_amendment_producer_task_id"] == "a" * 32


@pytest.mark.parametrize("module_name", MODULE_NAMES)
def test_successor_parser_accepts_every_sealed_pin(module_name: str) -> None:
    module = _load(module_name)
    options = {action.dest for action in module._parser()._actions}
    assert {name for name, _kind in module.SEALED_SUCCESSOR_PIN_ARGS} <= options
