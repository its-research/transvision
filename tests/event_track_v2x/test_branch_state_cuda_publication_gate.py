import importlib.util
import json
from pathlib import Path

import pytest


@pytest.fixture
def publication(monkeypatch,tmp_path):
    tools=Path(__file__).resolve().parents[2]/'tools/event_track_v2x'
    frozen=Path('/Volumes/Data/test/recover-before-fuse/source-freezes/rbf-final-refit-Top1-independent-output-reader-v1-20261004')
    monkeypatch.syspath_prepend(str(frozen));monkeypatch.syspath_prepend(str(tools))
    spec=importlib.util.spec_from_file_location('branch_state_publication',tools/'publish_branch_state_cuda_bundle.py')
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    monkeypatch.setattr(m,'R',tmp_path)
    p=tmp_path/'artifacts/package';p.mkdir(parents=True)
    (p/'inputs.tar.gz').write_bytes(b'software-gate-only')
    manifest=dict(kind='rbf_branch_state_CUDA_exact_source_and_admitted_reference_bundle_v1',
        CPU_acceptance_sha256='cpu-proof',GPU_devices_required=1,actual_GPU_execution=False,files={'x':{}})
    (p/'manifest.json').write_text(json.dumps(manifest))
    proof=dict(kind='rbf_branch_state_CUDA_input_package_independent_stream_readback_v1',full_archive_stream_hashes_match=True,
        archive_sha256=m.sha(p/'inputs.tar.gz'),archive_bytes=(p/'inputs.tar.gz').stat().st_size,
        manifest_sha256=m.sha(p/'manifest.json'),CPU_acceptance_sha256='cpu-proof',members=2)
    (p/'package-readback.json').write_text(json.dumps(proof))
    return m,p,proof


def test_publication_identity_binds_all_three_exact_files(publication):
    m,p,_=publication
    files,specs,identity=m.specification(p)
    assert set(files)==set(specs)=={'inputs','manifest','local-package-readback'}
    assert len(identity)==64 and m.specification(p)[2]==identity


@pytest.mark.parametrize('mutation',['archive_bytes','manifest','CPU_proof','incomplete_readback','member_count'])
def test_input_mismatch_rejected_before_external_task_creation(publication,mutation):
    m,p,proof=publication
    if mutation=='archive_bytes':(p/'inputs.tar.gz').write_bytes(b'changed')
    elif mutation=='manifest':(p/'manifest.json').write_text('{}')
    elif mutation=='CPU_proof':proof['CPU_acceptance_sha256']='different-proof'
    elif mutation=='incomplete_readback':proof['full_archive_stream_hashes_match']=False
    else:proof['members']=99
    (p/'package-readback.json').write_text(json.dumps(proof))
    with pytest.raises(AssertionError):m.specification(p)
