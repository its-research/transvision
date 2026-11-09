from copy import deepcopy
from types import SimpleNamespace
import pytest
from tools.event_track_v2x.readback_v2v4real_numeric_gpu import verify_registered_inputs,verify_original_rank
from tools.event_track_v2x import run_v2v4real_official_test_raw_features as o


def inputs():
    expected={('owner','source'):'a'*64}
    owners={'owner':SimpleNamespace(artifacts={'source':SimpleNamespace(hash='a'*64,size=123,url='http://10.100.35.118:8081/original/source')})}
    rows=[dict(task_id='owner',artifact='source',sha256='a'*64,bytes=123,registered_file_host='10.100.35.118:8081',transport_file_host='10.100.34.118:8081')]
    return rows,expected,owners


def test_registered_inputs_exact():
    verify_registered_inputs(*inputs())

@pytest.mark.parametrize('field,value',[('task_id','other'),('bytes',124),('sha256','b'*64),('transport_file_host','redirect')])
def test_registered_input_substitution(field,value):
    rows,expected,owners=inputs();rows[0][field]=value
    with pytest.raises(ValueError):verify_registered_inputs(rows,expected,owners)


def test_duplicate_registered_input():
    rows,expected,owners=inputs()
    with pytest.raises(ValueError):verify_registered_inputs(rows*2,expected,owners)


def rank():
    paths=('project/tools/event_track_v2x/prepare_v2v4real_inputs.py','project/transvision/models/event_track_v2x/v2v4real_inputs.py')
    files={p:'a'*64 for p in paths}
    value={'point_reader_binding':{n:dict(path=p,sha256=files[p]) for n,p in zip(('tools.event_track_v2x.prepare_v2v4real_inputs','transvision.models.event_track_v2x.v2v4real_inputs'),paths)}}
    e=dict(kind='v2v4real_frozen_official_extension_build_v1',status='completed',source_archive_sha256=o.SOURCE_ARCHIVE_SHA,
        builder_path=o.EXTENSION_BUILDER,builder_sha256=o.EXTENSION_BUILDER_SHA,builder_function_ast_sha256='b'*64,
        python_executable='/python',original_source_count=2,original_sources_unchanged=True,subprocess_import_verified=True,
        independent_runtime_acceptance=False,binaries=[dict(path='official/opencood/utils/box_overlaps.cpython.so',sha256='c'*64,bytes=42)])
    argv=[['/python','/rank/official/opencood/utils/setup.py','build_ext','--inplace'],['/python','-c','from opencood.utils.box_overlaps import bbox_overlaps; assert callable(bbox_overlaps)']]
    e['commands']=[dict(argv=a,cwd='/rank/official',check=True,status='completed',returncode=0,timeout_seconds=t) for a,t in zip(argv,(900,180))]
    return value,e,dict(files=files,builder_ast='b'*64,interpreter='/python')


def test_original_binding_and_extension():
    v,e,k=rank();verify_original_rank(v,e,**k)

@pytest.mark.parametrize('mutation',['reader','builder','interpreter','source_changed','command','binary'])
def test_original_rank_substitution(mutation):
    v,e,k=rank()
    if mutation=='reader':v['point_reader_binding']={}
    elif mutation=='builder':e['builder_sha256']='d'*64
    elif mutation=='interpreter':e['python_executable']='/ambient/python'
    elif mutation=='source_changed':e['original_sources_unchanged']=False
    elif mutation=='command':e['commands'][1]['argv']=['/python','-c','pass']
    elif mutation=='binary':e['binaries'][0]['path']='../outside.so'
    with pytest.raises(ValueError):verify_original_rank(v,e,**k)
