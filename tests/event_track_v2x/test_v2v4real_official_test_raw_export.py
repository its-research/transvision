"""No-data software gates for the separately named official-test exporter."""
import ast
import hashlib
import importlib
import json
from pathlib import Path
import sys
import tarfile
import types

import numpy as np
import pytest
import torch

from tools.event_track_v2x import run_v2v4real_official_test_raw_features as export

ARCHIVE=Path('/Volumes/Data/test/recover-before-fuse/historical/live-audit-20260926/fulltrain-native-export-source.tar.gz')
HISTORY=ARCHIVE.parent


@pytest.fixture(scope='module')
def source(tmp_path_factory):
    out=tmp_path_factory.mktemp('official_native')/'source'
    copied=export.extract_source(ARCHIVE,out)
    assert len(copied)==403
    return out


@pytest.fixture(scope='module')
def core(source):
    result=export.frozen_core(source)
    yield result
    for name in list(sys.modules):
        if name==export.PACKAGE or name.startswith(export.PACKAGE+'.'):del sys.modules[name]


def test_frozen_core_and_raw_export_inference_AST_unchanged(source,core):
    path=source/'project/transvision/models/event_track_v2x/paper_pointpillar.py'
    assert export.sha(path)==export.CORE_SHA
    assert Path(core.__file__).resolve()==path.resolve()
    with tarfile.open(ARCHIVE) as t:
        original=t.extractfile('project/transvision/models/event_track_v2x/paper_pointpillar.py').read()
    assert ast.dump(ast.parse(path.read_bytes()))==ast.dump(ast.parse(original))
    tree=ast.parse(original)
    fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='native_arrays')
    assert not any(isinstance(n,ast.Name) and n.id=='rotated_nms' for n in ast.walk(fn))
    assert b"PaperProtocol('v2v4real', 'train').select" in original


def heads(count=70,logit=0.):
    psm=torch.full((1,1,1,count),logit,dtype=torch.float64)
    regression=torch.zeros((1,7,1,count),dtype=torch.float64)
    anchors=np.zeros((1,count,1,7));anchors[...,3:6]=[1.5,2.,4.]
    anchors[0,:,0,0]=np.arange(count)/100.
    bev=torch.ones((1,128,1,count),dtype=torch.float64)
    return psm,regression,anchors,bev


@pytest.mark.parametrize('count,logit,expected',[(70,0.,64),(5,-100.,0),(1,20.,1)])
def test_unchanged_core_raw_top64_keeps_overlap_ties_and_empty(core,count,logit,expected):
    arrays=core.native_arrays(*heads(count,logit),lidar_range=[-5,-5,-5,5,5,5],
        covariance_diagonal=np.ones(9),calibration=export.IdentityCalibration())
    assert arrays['states'].shape==(expected,9)
    np.testing.assert_allclose(arrays['states'][:,0],np.arange(expected)/100.)
    assert arrays['appearance'].shape==(expected,128)
    assert arrays['appearance_valid'].all()


def test_unchanged_protocol_selects_all_classes_with_stable_threshold(core):
    protocol=importlib.import_module(export.PACKAGE+'.paper_protocol').PaperProtocol('v2v4real','official_test')
    assert protocol.select(np.array([.05,.2,.2,.049]),np.array([2,1,0,0])).tolist()==[1,2,0]


def test_checkpoint_seeds_bind_original_native_exports():
    for seed,task in ((1337,'d9876e8f09f644bc90dc09195a997ad7'),(2027,'24079dc97841492392e32fcb92b9cf40'),(3407,'0cb4b70b004946c9b3ed2d6c34829d28')):
        m=json.loads((HISTORY/task/'native-feature-manifest').read_bytes())
        assert m['checkpoint_seed']==seed and m['checkpoint_sha256']==export.CHECKPOINTS[seed]
        assert m['partition_sha256']==export.PARTITION_SHA


def test_original_projection_admission_and_raw_scope():
    root=HISTORY/export.PROJECTION_TASK
    projection=json.loads((root/'projection-manifest').read_bytes())
    admission=json.loads((root/'split-admission').read_bytes())
    assert export.sha(root/'projection-manifest')==export.PROJECTION_SHA
    assert export.sha(root/'split-admission')==export.ADMISSION_SHA
    records=[dict(sequence_id=s) for s in projection['ego_agents']]
    records += [dict(sequence_id=next(iter(projection['ego_agents'])))]*(3986-len(records))
    export.validate_scope(projection,admission,records)
    for key,value in (('dataset_split','train'),('gt_in_projection',True),('frames_sha256','a'*64)):
        with pytest.raises(ValueError):export.validate_scope(dict(projection,**{key:value}),admission,records)
    with pytest.raises(ValueError):export.validate_scope(projection,dict(admission,split='official_train'),records)
    with pytest.raises(ValueError):export.validate_scope(projection,admission,records[:-1])


def raw_fixture(core,tmp_path):
    pose=np.eye(4).tolist();pose[0][3]=2.125
    rows=[dict(sequence_id='scene',frame_ordinal=i,frame_key=f'{i:06d}',is_ego=ego,
        source_to_world=pose,pcd_path=f'{i}-{ego}.pcd',pcd_sha256='f'*64) for i in range(2) for ego in (True,False)]
    calls=[]
    class Detector:
        def predict(self,points,**kwargs):
            return core.native_arrays(*heads(2,-100. if points[0,0] else 0.),**kwargs)
    def read(path,expected_sha256):
        assert expected_sha256=='f'*64
        calls.append(path.name)
        return types.SimpleNamespace(xyzi=np.array([[int(path.name[0]),0,0,1.]]))
    progress=[]
    result,valid=export.export_rows(rows,tmp_path,Detector(),[-5,-5,-5,5,5,5],read,
        lambda p:p,lambda p,b:p,tmp_path/'raw',progress=progress.append)
    return rows,result,valid,calls,progress


def test_source_pose_identity_empty_rows_and_BEV_payloads(core,tmp_path):
    records,rows,valid,calls,progress=raw_fixture(core,tmp_path)
    assert len(rows)==len(calls)==4 and valid==4
    assert [r['candidates'] for r in rows]==[2,2,0,0]
    assert all(a['source_to_world']==b['source_to_world'] for a,b in zip(records,rows))
    assert progress[-1]['completed_rows']==progress[-1]['total_rows']==4
    assert progress[-1]['ETA_seconds']==0
    for row in rows:
        with np.load(tmp_path/'raw'/row['payload']) as arrays:
            assert set(arrays.files)=={'states_source_legacy7','raw_scores','class_indices','appearance','appearance_valid'}
            assert arrays['states_source_legacy7'].shape==(row['candidates'],7)
            assert arrays['appearance'].shape==(row['candidates'],128)


def test_consumer_envelope_packs_exact_indexed_bytes_and_no_GT(core,tmp_path):
    _,rows,_,_,_=raw_fixture(core,tmp_path)
    out=tmp_path/'out';out.mkdir()
    work=tmp_path/'raw';np.save(work/'anchors.npy',np.zeros((1,1,1,7)))
    a=tmp_path/'projection';a.write_text('{}')
    b=tmp_path/'admission';b.write_text('{}')
    raw=b''.join(export.canonical(r)+b'\n' for r in rows)
    manifest=dict(kind=export.KIND,rows_sha256=hashlib.sha256(raw).hexdigest())
    archive_hash=export.pack(work,out,manifest,rows,a,b)
    assert export.sha(out/'native-features')==archive_hash
    assert (out/'native-feature-index').read_bytes()==raw
    assert json.loads((out/'native-feature-manifest').read_bytes())['kind']==export.KIND
    with tarfile.open(out/'native-features') as stream:
        assert set(stream.getnames())=={'anchors.npy','manifest.json','predictions.jsonl',*(r['payload'] for r in rows)}
        assert all(m.isfile() for m in stream)


@pytest.mark.parametrize('which',['archive','checkpoint','seed'])
def test_wrong_assets_rejected_before_model_or_output(tmp_path,which):
    bad=tmp_path/'bad';bad.write_text('not a source or model')
    # Deliberate invalid assets never reach model construction or data reads.
    archive=bad if which=='archive' else ARCHIVE
    with pytest.raises(ValueError):export.run(archive,tmp_path,bad,bad,999 if which=='seed' else 1337,tmp_path/'out',device='cpu')
    assert not (tmp_path/'out').exists()


def test_existing_namespace_cannot_substitute_ambient_NMS_core(core,source):
    with pytest.raises(ValueError,match='namespace'):export.frozen_core(source)



def test_manifest_schema_matches_official_test_consumer():
    from tools.event_track_v2x.build_v2v4real_raw_native_cache import RAW_MANIFEST_FIELDS
    tree=ast.parse(Path(export.__file__).read_bytes())
    run=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='run')
    manifest=next(n.value for n in run.body if isinstance(n,ast.Assign)
                  and any(isinstance(t,ast.Name) and t.id=='manifest' for t in n.targets))
    assert {k.arg for k in manifest.keywords}==RAW_MANIFEST_FIELDS|{'projection_manifest','split_admission'}


def test_reject_unpaired_source_frames_before_prediction(core,tmp_path):
    detector=types.SimpleNamespace(predict=lambda *a,**kw:pytest.fail('must not infer broken pair'))
    rows=[dict(sequence_id='scene',frame_ordinal=0,frame_key='000000',is_ego=True)]
    with pytest.raises(ValueError,match='ego and one partner'):
        export.export_rows(rows,tmp_path,detector,[],None,None,None,tmp_path/'raw')


@pytest.mark.parametrize('uuid',[None,'GPU-fixture-uuid'])
def test_device_uuid_is_observation_not_physical_resource_admission(monkeypatch,uuid):
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES','4,5')
    fake=types.SimpleNamespace(device=lambda _:types.SimpleNamespace(type='cuda',index=None),
        cuda=types.SimpleNamespace(current_device=lambda:1,
            get_device_properties=lambda index:types.SimpleNamespace(name='fixture GPU',uuid=uuid)))
    result=export.device_identity(fake,'cuda')
    assert result['logical_index']==1 and result['uuid']==uuid
    assert result['cuda_visible_devices']=='4,5'
    assert result['physical_identity_verified'] is False and result['resource_admission_verified'] is False
    assert result['physical_identity_status']==('uuid_observed_not_independently_verified' if uuid else 'uuid_unavailable')


def test_post_pack_source_gate_rejects_changed_runtime(tmp_path):
    source=tmp_path/'source';source.mkdir();file=source/'core.py';file.write_text('original bytes')
    inventory={'core.py':export.sha(file)}
    export.verify_runtime_sources(source,inventory)
    file.write_text('changed bytes')
    with pytest.raises(ValueError,match='asset bytes differ'):export.verify_runtime_sources(source,inventory)
    run=next(n for n in ast.parse(Path(export.__file__).read_bytes()).body if isinstance(n,ast.FunctionDef) and n.name=='run')
    # The second gate is a direct statement after archive packing, before receipt.
    pack_line=next(n.lineno for n in ast.walk(run) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='pack')
    checks=[n.lineno for n in ast.walk(run) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='verify_runtime_sources']
    assert len(checks)==2 and min(checks)<pack_line<max(checks)
