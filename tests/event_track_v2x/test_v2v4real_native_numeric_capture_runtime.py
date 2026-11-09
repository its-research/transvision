"""Fail-before-runtime contract checks, not CUDA qualification."""
import pytest

from tools.event_track_v2x.run_v2v4real_native_numeric_capture import run


@pytest.mark.parametrize('case', ['cpu', 'budget', 'source'])
def test_invalid_runtime_inputs_do_not_create_candidate_output(tmp_path, case):
    source=tmp_path/'source'; source.write_bytes(b'not the original published archive')
    args=dict(source_archive=source, inputs_root=tmp_path/'inputs', admission=tmp_path/'admission',
        checkpoint=tmp_path/'checkpoint', seed=1337, output=tmp_path/'output',
        accepted_raw_index=tmp_path/'raw.jsonl', accepted_raw_sha='a'*64,
        accepted_raw_root=tmp_path/'raw', max_trace_bytes=1024, device='cuda:0')
    if case=='cpu': args['device']='cpu'
    if case=='budget': args['max_trace_bytes']=0
    with pytest.raises(ValueError): run(**args)
    assert not args['output'].exists()
