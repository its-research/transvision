"""Fresh interpreter namespace admission; no CUDA or detector execution."""
from pathlib import Path
import subprocess,sys

SCRIPT=Path(__file__).resolve().parents[2]/'tools/event_track_v2x/run_v2v4real_native_numeric_capture.py'

def test_direct_rank_dependencies_leave_original_namespace_unclaimed():
    code="import runpy,sys;d=runpy.run_path(sys.argv[1],run_name='rank_preflight');assert not any(n=='tools' or n.startswith('tools.') or n=='transvision' or n.startswith('transvision.') for n in sys.modules);d['original'].bind_frozen_point_reader.__name__=='bind_frozen_point_reader'"
    result=subprocess.run([sys.executable,'-c',code,str(SCRIPT)],capture_output=True,text=True)
    assert result.returncode==0,result.stderr

def test_ambient_original_namespace_rejected():
    code="import runpy,sys,types;sys.modules['tools']=types.ModuleType('tools');runpy.run_path(sys.argv[1],run_name='rank_preflight')"
    result=subprocess.run([sys.executable,'-c',code,str(SCRIPT)],capture_output=True,text=True)
    assert result.returncode!=0 and 'fresh isolated numeric rank interpreter required' in result.stderr
