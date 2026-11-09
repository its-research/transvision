"""Synthetic transport inventory checks; no numerical or GPU acceptance."""
import hashlib
import json
from pathlib import Path

import pytest

from tools.event_track_v2x.v2v4real_numeric_trace_archive import pack,readback
from tools.event_track_v2x.verify_v2v4real_native_numeric_traces import digest


@pytest.fixture(scope='module')
def candidate(tmp_path_factory):
    root=tmp_path_factory.mktemp('trace-candidate');(root/'payloads').mkdir()
    payload=b'synthetic transport bytes, not numeric arrays';sha=hashlib.sha256(payload).hexdigest()
    with (root/'traces.jsonl').open('w') as f:
        for n in range(3986):
            name=f'payloads/{n:04d}.npz';(root/name).write_bytes(payload)
            f.write(json.dumps(dict(sequence_id='synthetic',frame_ordinal=n,side='vehicle-side',
                frame_id=str(n),payload=name,payload_sha256=sha,raw_rows_sha256='a'*64,
                TF32_enabled=False,trace_layout='compact-cells-v1'))+'\n')
    (root/'capacity-preflight.json').write_text(json.dumps(dict(rows=3986,within_budget=True)))
    return root


def test_create_once_lossless_full_inventory_readback(candidate,tmp_path):
    out=tmp_path/'package'
    envelope=pack(candidate,out,expected_rows=3986,raw_index_sha='a'*64,max_archive_bytes=10**7)
    assert len(envelope['inventory'])==3988 and not envelope['independent_numeric_acceptance']
    proof=readback(out/'numeric-traces.tar',out/'trace-manifest.json',tmp_path/'readback',
        manifest_sha=digest(out/'trace-manifest.json'),max_total_bytes=10**7)
    assert proof['independent_transport_bytes_verified'] and not proof['independent_numeric_acceptance']
    with pytest.raises(ValueError):pack(candidate,out,expected_rows=3986,raw_index_sha='a'*64,max_archive_bytes=10**7)


@pytest.mark.parametrize('case',['cohort','binding','budget'])
def test_failed_admission_cannot_create_package(candidate,tmp_path,case):
    args=dict(expected_rows=3986,raw_index_sha='a'*64,max_archive_bytes=10**7)
    if case=='cohort':args['expected_rows']=1
    if case=='binding':args['raw_index_sha']='b'*64
    if case=='budget':args['max_archive_bytes']=1
    with pytest.raises(ValueError):pack(candidate,tmp_path/'out',**args)
    assert not (tmp_path/'out').exists()


def test_corrupted_archive_rejected_before_extracting(candidate,tmp_path):
    out=tmp_path/'package';pack(candidate,out,expected_rows=3986,raw_index_sha='a'*64,max_archive_bytes=10**7)
    archive=out/'numeric-traces.tar'
    with archive.open('r+b') as f:f.write(b'changed')
    with pytest.raises(ValueError):readback(archive,out/'trace-manifest.json',tmp_path/'extracted',
        manifest_sha=digest(out/'trace-manifest.json'),max_total_bytes=10**7)
    assert not (tmp_path/'extracted').exists()
