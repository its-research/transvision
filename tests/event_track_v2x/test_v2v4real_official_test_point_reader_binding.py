"""Execute the original lazy PCD import in a fresh interpreter, without repo paths."""
from pathlib import Path
import subprocess
import sys

import pytest

SCRIPT=Path(__file__).resolve().parents[2]/'tools/event_track_v2x/run_v2v4real_official_test_raw_features.py'
ARCHIVE=Path('/Volumes/Data/test/recover-before-fuse/historical/live-audit-20260926/fulltrain-native-export-source.tar.gz')


@pytest.mark.parametrize('case',['valid','hash-mismatch','symlink','ambient-namespace','tampered-source'])
def test_original_frozen_point_reader_lazy_import(tmp_path,case):
    # -I prevents the test runner and current checkout from supplying tools/transvision.
    code=r'''
import hashlib,importlib,importlib.util,json,struct,sys,types
from pathlib import Path
script,archive,root,case=map(str,sys.argv[1:]);root=Path(root)
spec=importlib.util.spec_from_file_location('isolated_export',script)
export=importlib.util.module_from_spec(spec);spec.loader.exec_module(export)
files=export.extract_source(archive,root/'runtime')
if case=='ambient-namespace':sys.modules['tools']=types.ModuleType('tools')
if case=='tampered-source':
 (root/'runtime/project/tools/event_track_v2x/prepare_v2v4real_inputs.py').write_text('raise AssertionError("must not execute")')
if case in ('ambient-namespace','tampered-source'):
 try:export.bind_frozen_point_reader(root/'runtime',files)
 except ValueError:print(json.dumps({'negative_rejected':case}));sys.exit(0)
 raise AssertionError('unsafe namespace/source was accepted')
bindings=export.bind_frozen_point_reader(root/'runtime',files)
package=types.ModuleType(export.PACKAGE);package.__path__=[str(root/'runtime/project/transvision/models/event_track_v2x')]
sys.modules[export.PACKAGE]=package
pcd=importlib.import_module(export.PACKAGE+'.v2v4real_pcd')
header='# .PCD v0.7\nVERSION 0.7\nFIELDS x y z rgb\nSIZE 4 4 4 4\nTYPE F F F F\nCOUNT 1 1 1 1\nWIDTH 1\nHEIGHT 1\nVIEWPOINT 0 0 0 1 0 0 0\nPOINTS 1\nDATA ascii\n'
word=128<<16|0x1234;rgb=struct.unpack('<f',struct.pack('<I',word))[0]
raw=(header+f'1.25 -2.5 0.75 {rgb:.10e}\n').encode();path=root/'point.pcd';path.write_bytes(raw)
digest=hashlib.sha256(raw).hexdigest()
if case=='hash-mismatch':digest='0'*64
if case=='symlink':
 link=root/'alias.pcd';link.symlink_to(path);path=link
try:cloud=pcd.read_native_pcd(path,expected_sha256=digest)
except pcd.V2V4RealInputError:
 assert case in ('hash-mismatch','symlink');print(json.dumps({'negative_rejected':case}));sys.exit(0)
assert case=='valid' and cloud.source_sha256==digest
assert cloud.xyzi.shape==(1,4) and abs(float(cloud.xyzi[0,3])-128/255)<1e-7
assert cloud.rgb_bits.tolist()==[word] and not cloud.xyzi.flags.writeable
export.verify_runtime_sources(root/'runtime',files)
print(json.dumps({'original_point_reader_executed':True,'bindings':bindings}))
'''
    result=subprocess.run([sys.executable,'-I','-c',code,str(SCRIPT),str(ARCHIVE),str(tmp_path.resolve()),case],
                          capture_output=True,text=True,timeout=60,cwd='/private/tmp')
    assert result.returncode==0,result.stdout+result.stderr
    assert ('original_point_reader_executed' if case=='valid' else 'negative_rejected') in result.stdout
