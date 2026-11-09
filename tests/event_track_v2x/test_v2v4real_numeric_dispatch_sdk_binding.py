import ast
from pathlib import Path
import pytest
from tools.event_track_v2x.submit_v2v4real_numeric_gpu import verify_uploaded_parameters


def test_sdk_exact_serialized_parameters():
    verify_uploaded_parameters({'General/seed':'1337','General/gt_read':'False','General/sha':'abc'},dict(seed=1337,gt_read=False,sha='abc'))

@pytest.mark.parametrize('value',['True','false','0',0,None])
def test_false_not_replaced_by_truthiness(value):
    with pytest.raises(ValueError):verify_uploaded_parameters({'General/gt_read':value},dict(gt_read=False))


def test_task_name_is_not_artifact_loop_variable():
    p=Path(__file__).resolve().parents[2]/'tools/event_track_v2x/submit_v2v4real_numeric_gpu.py'
    tree=ast.parse(p.read_bytes())
    calls=[n for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='create']
    keyword=next(k for k in calls[0].keywords if k.arg=='task_name')
    assert isinstance(keyword.value,ast.Name) and keyword.value.id=='task_name'
    loops=[n for n in ast.walk(tree) if isinstance(n,ast.For)]
    assert not any(isinstance(n,ast.Name) and n.id=='task_name' for l in loops for n in ast.walk(l.target))
