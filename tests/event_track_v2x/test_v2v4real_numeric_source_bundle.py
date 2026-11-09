import hashlib,json,subprocess,sys
from pathlib import Path
import pytest
from tools.event_track_v2x.prepare_v2v4real_numeric_gpu_source import build,RAW_TASKS,FILES

def proofs(tmp_path):
    result={}
    for seed,task in RAW_TASKS.items():
        path=tmp_path/f'seed{seed}.json';path.write_text(json.dumps(dict(task_id=task,checkpoint_seed=seed,
            independent_byte_schema_coverage_accepted=True,input_sha256={n:'a'*64 for n in ('native-features','native-feature-manifest','native-feature-index')})))
        result[seed]=(path,hashlib.sha256(path.read_bytes()).hexdigest())
    return result

SOURCE=Path(__file__).resolve().parents[2]/'tools/event_track_v2x'

def test_complete_cold_import_and_direct_rank_namespace(tmp_path):
    script,manifest=build(SOURCE,proofs(tmp_path));namespace={'__name__':'preparation'};exec(script,namespace)
    root=namespace['materialize'](tmp_path/'code')
    code="import sys,runpy;sys.path.insert(0,sys.argv[1]);runpy.run_path(sys.argv[1]+'/tools/event_track_v2x/run_v2v4real_native_numeric_capture.py',run_name='preflight');assert 'tools' not in sys.modules;from tools.event_track_v2x import run_v2v4real_numeric_capture_gpu_shards"
    result=subprocess.run([sys.executable,'-c',code,str(root)],cwd=tmp_path,capture_output=True,text=True)
    assert result.returncode==0,result.stderr
    assert len(manifest['sources'])==len(FILES)+3 and not manifest['actual_GPU_execution_verified']

def test_wrong_proof_pin_rejected(tmp_path):
    pins=proofs(tmp_path);pins[1337]=(pins[1337][0],'0'*64)
    with pytest.raises(ValueError):build(SOURCE,pins)

def test_embedded_damage_rejected_before_materializing(tmp_path):
    script,_=build(SOURCE,proofs(tmp_path));namespace={'__name__':'preparation'};exec(script,namespace)
    namespace['FILES'][next(iter(namespace['FILES']))]['sha256']='0'*64
    with pytest.raises(ValueError):namespace['materialize'](tmp_path/'code')
    assert not (tmp_path/'code').exists()
