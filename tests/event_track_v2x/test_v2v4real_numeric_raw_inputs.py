import hashlib,io,json,tarfile
from pathlib import Path
import pytest
from tools.event_track_v2x.admit_v2v4real_numeric_raw_inputs import admit,sha

@pytest.fixture(scope='module')
def inputs(tmp_path_factory):
    root=tmp_path_factory.mktemp('raw-admission');payload=b'synthetic';psha=hashlib.sha256(payload).hexdigest()
    index=root/'index';index.write_bytes(b''.join(json.dumps(dict(payload=f'rows/{i}.npz',payload_sha256=psha)).encode()+b'\n' for i in range(3986)))
    manifest=root/'manifest';manifest.write_text(json.dumps(dict(rows_sha256=sha(index),checkpoint_seed=1337,gt_read=False,anchor_file_sha256=psha)))
    archive=root/'archive'
    with tarfile.open(archive,'w') as f:
        for name,data in [('anchors.npy',payload),('manifest.json',manifest.read_bytes()),('predictions.jsonl',index.read_bytes()),*[(f'rows/{i}.npz',payload) for i in range(3986)]]:
            m=tarfile.TarInfo(name);m.size=len(data);f.addfile(m,io.BytesIO(data))
    proof=root/'proof';proof.write_text(json.dumps(dict(task_id='synthetic',checkpoint_seed=1337,independent_byte_schema_coverage_accepted=True,input_sha256={'native-features':sha(archive),'native-feature-manifest':sha(manifest),'native-feature-index':sha(index)})))
    return dict(proof=proof,proof_sha=sha(proof),task_id='synthetic',seed=1337,archive=archive,manifest=manifest,index=index,max_unpacked_bytes=1000000)

def test_existing_proof_bounds_transport_without_new_acceptance(inputs,tmp_path):
    result=admit(**inputs,output=tmp_path/'inputs')
    assert result['source_rows']==3986 and not result['new_raw_acceptance'] and not result['independent_numeric_acceptance']
    assert sha(tmp_path/'inputs/predictions.jsonl')==sha(inputs['index'])

@pytest.mark.parametrize('change',[dict(seed=2027),dict(proof_sha='0'*64),dict(max_unpacked_bytes=1)])
def test_wrong_seed_proof_and_capacity_fail_before_extract(inputs,tmp_path,change):
    with pytest.raises(ValueError):admit(**(inputs|change),output=tmp_path/'inputs')
    assert not (tmp_path/'inputs').exists()
