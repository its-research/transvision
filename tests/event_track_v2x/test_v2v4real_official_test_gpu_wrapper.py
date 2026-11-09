"""No-data boundary checks for the original-input Linux/CUDA candidate wrapper."""
import ast
import base64
import hashlib
import json
from pathlib import Path
import tarfile
import types
import sys
import pytest
from tools.event_track_v2x import run_v2v4real_official_test_gpu_shards as gpu
from tools.event_track_v2x import prepare_v2v4real_official_test_gpu_source as prepare


def test_embedded_bundle_reconstructs_exact_minimal_source_bytes():
    directory=Path(gpu.__file__).parent;script,m=prepare.build_bootstrap(directory)
    tree=ast.parse(script)
    files=ast.literal_eval(next(n.value for n in tree.body if isinstance(n,ast.Assign) and n.targets[0].id=='FILES'))
    assert len(files)==4
    for name,row in files.items():
        raw=base64.b64decode(row['base64'],validate=True)
        assert raw==(directory/Path(name).name).read_bytes()
        assert hashlib.sha256(raw).hexdigest()==row['sha256'] and len(raw)==row['bytes']
    assert hashlib.sha256(script.encode()).hexdigest()==m['bootstrap_sha256']
    assert m['actual_GPU_execution_verified'] is False


@pytest.mark.parametrize('count',[0,3,5,8])
def test_four_card_execution_rejects_wrong_actual_CUDA_card_count(tmp_path,monkeypatch,count):
    monkeypatch.setitem(sys.modules,'torch',types.SimpleNamespace(cuda=types.SimpleNamespace(is_available=lambda:True,device_count=lambda:count)))
    with pytest.raises(ValueError,match='CUDA card count'):gpu.launch(tmp_path,None,None,None,None,1337,4)


@pytest.mark.parametrize('world',[0,1,2,True])
def test_unsupported_worker_size_never_launches(tmp_path,world):
    with pytest.raises(ValueError,match='admitted GPU'):gpu.launch(tmp_path,None,None,None,None,1337,world)


def test_projection_extraction_requires_original_inventory_and_frame_assets(tmp_path,monkeypatch):
    root=tmp_path/'source';(root/'inputs').mkdir(parents=True)
    for name in ('inputs/manifest.json','inputs/frames.jsonl','split-admission.json'):
        (root/name).write_text(name)
    inventory=[dict(path=str(p.relative_to(root)),bytes=p.stat().st_size,sha256=gpu.export.sha(p)) for p in root.rglob('*') if p.is_file()]
    publication=dict(kind='v2v4real_formal_projection_publication_v1',split='official_test',gt_in_projection=False,inventory=inventory,
        projection_manifest_sha256=gpu.export.sha(root/'inputs/manifest.json'),split_admission_sha256=gpu.export.sha(root/'split-admission.json'))
    receipt=tmp_path/'publication-receipt.json';receipt.write_text(json.dumps(publication))
    monkeypatch.setattr(gpu,'PUBLICATION_SHA',gpu.export.sha(receipt))
    archive=tmp_path/'projection.tar.gz'
    with tarfile.open(archive,'w:gz') as stream:
        for p in sorted(root.rglob('*')):
            if p.is_file():stream.add(p,arcname='projection/'+str(p.relative_to(root)),recursive=False)
        stream.add(receipt,arcname='projection/publication-receipt.json',recursive=False)
    monkeypatch.setattr(gpu,'PROJECTION_ARCHIVE_SHA',gpu.export.sha(archive))
    for constant,name in [('PROJECTION_SHA','inputs/manifest.json'),('FRAME_SHA','inputs/frames.jsonl'),('ADMISSION_SHA','split-admission.json')]:
        monkeypatch.setattr(gpu.export,constant,gpu.export.sha(root/name))
    inputs,admission=gpu.unpack_projection(archive,tmp_path/'unpacked',receipt)
    assert inputs.name=='inputs' and admission.name=='split-admission.json'
    import io
    for mutation in ('missing-self','extra-member','changed-self'):
        bad=tmp_path/(mutation+'.tar.gz')
        with tarfile.open(archive) as original,tarfile.open(bad,'w:gz') as altered:
            for member in original:
                if mutation=='missing-self' and member.name=='projection/publication-receipt.json':continue
                data=original.extractfile(member).read()
                if mutation=='changed-self' and member.name=='projection/publication-receipt.json':data=b'!'+data[1:]
                info=tarfile.TarInfo(member.name);info.size=len(data);altered.addfile(info,io.BytesIO(data))
            if mutation=='extra-member':
                info=tarfile.TarInfo('projection/unregistered.bin');info.size=1;altered.addfile(info,io.BytesIO(b'x'))
        monkeypatch.setattr(gpu,'PROJECTION_ARCHIVE_SHA',gpu.export.sha(bad))
        with pytest.raises(ValueError):gpu.unpack_projection(bad,tmp_path/('reject-'+mutation),receipt)
    monkeypatch.setattr(gpu,'PROJECTION_ARCHIVE_SHA',gpu.export.sha(archive))
    receipt.write_text(receipt.read_text()+' ')
    with pytest.raises(ValueError,match='asset bytes differ'):gpu.unpack_projection(archive,tmp_path/'bad-receipt',receipt)
    assert not (tmp_path/'bad-receipt').exists()
    with archive.open('ab') as f:f.write(b'changed')
    with pytest.raises(ValueError,match='asset bytes differ'):gpu.unpack_projection(archive,tmp_path/'mutated',receipt)


def test_registered_download_checks_bytes_and_disallows_credential_URL(tmp_path,monkeypatch):
    import requests
    session=types.ModuleType('clearml.backend_api.session')
    session.Session=lambda:types.SimpleNamespace(token='fixture-token')
    monkeypatch.setitem(sys.modules,'clearml.backend_api.session',session)
    payload=b'fixture-original';digest=hashlib.sha256(payload).hexdigest()
    item=types.SimpleNamespace(hash=digest,url='http://10.100.34.118:8081/fixture',size=len(payload))
    owner=types.SimpleNamespace(id='fixture-task',status='completed',artifacts={'projection':item})
    class Response:
        status_code=200
        def __enter__(self):return self
        def __exit__(self,*args):pass
        def iter_content(self,chunk_size):return iter([payload[:4],b'',payload[4:]])
    def get(url,**kwargs):
        assert url=='http://10.100.34.118:8081/fixture'
        assert kwargs['headers']=={'Authorization':'Bearer fixture-token'}
        assert kwargs['allow_redirects'] is False and kwargs['timeout']==(10,120)
        return Response()
    monkeypatch.setattr(requests,'get',get)
    assert gpu.download(owner,'projection',digest,tmp_path/'good')['sha256']==digest
    item.url='http://10.100.35.118:8081/fixture'
    admitted=gpu.download(owner,'projection',digest,tmp_path/'LAN-routed')
    assert admitted['registered_file_host']=='10.100.35.118:8081'
    assert admitted['transport_file_host']=='10.100.34.118:8081'
    item.url='http://user:secret@10.100.34.118:8081/fixture'
    with pytest.raises(ValueError,match='file service'):gpu.download(owner,'projection',digest,tmp_path/'bad')
    assert not (tmp_path/'bad').exists()
    item.url='http://10.100.34.118:8081/fixture'
    for status in (301,302,401,500):
        Response.status_code=status
        with pytest.raises(ValueError,match='HTTP status'):gpu.download(owner,'projection',digest,tmp_path/'bad')
        assert not (tmp_path/'bad').exists()
    Response.status_code=200
    item.size+=1
    with pytest.raises(ValueError,match='bytes differ'):gpu.download(owner,'projection',digest,tmp_path/'short')


@pytest.mark.parametrize('mutation',['completed','running','has-output','wrong-checkpoint','wrong-frames','ambiguous'])
def test_retry_never_duplicates_live_or_successful_or_mismatched_predecessor(mutation):
    from tools.event_track_v2x import submit_v2v4real_official_test_raw_gpu as dispatch
    params={'General/checkpoint_sha256':dispatch.CHECKPOINTS[1337],'General/projection_frames_sha256':dispatch.FRAME_SHA}
    task=types.SimpleNamespace(status='failed',artifacts={},get_parameters=lambda:params)
    dispatch.validate_failed_predecessors([task],1337)
    if mutation=='completed':task.status='completed'
    elif mutation=='running':task.status='in_progress'
    elif mutation=='has-output':task.artifacts={'native-features':object()}
    elif mutation=='wrong-checkpoint':params['General/checkpoint_sha256']='0'*64
    elif mutation=='wrong-frames':params['General/projection_frames_sha256']='0'*64
    with pytest.raises(ValueError):dispatch.validate_failed_predecessors([task,task] if mutation=='ambiguous' else [task],1337)


def test_projection_retry_queries_newest_existing_predecessor_first():
    from tools.event_track_v2x.submit_v2v4real_official_test_raw_gpu import predecessor_versions
    assert predecessor_versions('original-publication-envelope-v4')==('authenticated-LAN34-v3','authenticated-download-v2','v1')
    assert predecessor_versions('original-publication-self-member-v5')[0]=='original-publication-envelope-v4'
    with pytest.raises(KeyError):predecessor_versions('invented-retry')
