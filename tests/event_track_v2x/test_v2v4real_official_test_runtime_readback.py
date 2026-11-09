import json
import pytest
from tools.event_track_v2x import v2v4real_official_test_runtime as runtime
from tools.event_track_v2x import readback_v2v4real_official_test_raw_gpu as reader


def evidence():
    return dict(kind='v2v4real_native_runtime_dependency_attestation_v1',
        before=dict(interpreter='/usr/bin/python3.12'),after=dict(interpreter='/usr/bin/python3.12',
        dependencies=[dict(distribution=n,expected=v,actual=v,module=m,imported=True) for n,v,m in runtime.DEPENDENCIES]),
        same_interpreter_as_rank_children=True,model_or_inference_source_changed=False,
        raw_package_manager_output_persisted=False,experiment_accepted=False,
        system_packages=[dict(package='libx11-6',version='2:1.8.7-1',architecture='amd64')])


@pytest.mark.parametrize('mutation',['none','version','import','source','interpreter','scope','OS-package'])
def test_runtime_readback_binds_real_versions_imports_source_and_scope(tmp_path,mutation):
    value=evidence();source=dict(sources={'tools/event_track_v2x/v2v4real_official_test_runtime.py':dict(sha256=reader.export.sha(runtime.__file__))})
    if mutation=='version':value['after']['dependencies'][6]['actual']='0'
    elif mutation=='import':value['after']['dependencies'][6]['imported']=False
    elif mutation=='source':source['sources']['tools/event_track_v2x/v2v4real_official_test_runtime.py']['sha256']='0'*64
    elif mutation=='interpreter':value['after']['interpreter']='/different/python'
    elif mutation=='scope':value['experiment_accepted']=True
    elif mutation=='OS-package':value['system_packages'][0]['package']='unrecognized'
    path=tmp_path/'runtime.json';path.write_text(json.dumps(value))
    if mutation!='none':
        with pytest.raises(ValueError):reader.verify_runtime_dependency_receipt(path,source)
    else:
        accepted=reader.verify_runtime_dependency_receipt(path,source)
        assert accepted['fixed_Python_versions_and_import_results_verified']
        assert accepted['independent_detector_numeric_accepted'] is False


def test_offline_library_identity_requires_independently_pinned_manifest(tmp_path):
    from pathlib import Path
    original=Path('/Volumes/Data/test/recover-before-fuse/artifacts/spd-offline-libgl-runtime-20261001/manifest.json')
    v=evidence();v['offline_libraries']=[dict(task_id=runtime.OFFLINE_TASK,archive_sha256=runtime.OFFLINE_ARCHIVE_SHA,
        manifest_sha256=runtime.OFFLINE_MANIFEST_SHA,inventory=json.loads(original.read_bytes())['inventory'],
        libc_and_loader_excluded=True,prior_SPD_runtime_acceptance_inherited=False)]
    source=dict(sources={'tools/event_track_v2x/v2v4real_official_test_runtime.py':dict(sha256=reader.export.sha(runtime.__file__))})
    p=tmp_path/'receipt.json';p.write_text(json.dumps(v))
    with pytest.raises(ValueError,match='independent offline'):reader.verify_runtime_dependency_receipt(p,source)
    assert reader.verify_runtime_dependency_receipt(p,source,original)['offline_libraries']
    v['offline_libraries'][0]['inventory'][0]['sha256']='0'*64;p.write_text(json.dumps(v))
    with pytest.raises(ValueError,match='byte provenance'):reader.verify_runtime_dependency_receipt(p,source,original)
