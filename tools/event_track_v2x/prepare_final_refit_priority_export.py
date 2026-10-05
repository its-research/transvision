"""Prepare a distinct final-model priority consumer without exporting or fitting.

The v7 optimizer/model/checkpoint bytes and runtime kernel are unchanged.
Only receipt-format validation and training-only source roles change.
"""
import argparse
import ast
import datetime
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import tarfile

from rbf_nested_seen_val_v2_common import R,new,register,sha

PARENT=R/'source-freezes/rbf-exclusive-capacity-priority-original-paired-export-v7-portable-v2-20261002'
PARENT_MANIFEST_SHA='3f5c5229aef831762a447162b1bfee8921183c31fa3d1715c60c73dbfe945d0f'
BRIDGE_SHA='ebeae6fd7a264c307574f140c2de0699c1f50c1572586ecf03285b16bfc6ae4c'
ARCHIVE=R/'artifacts/rbf-original-joint-coupled-forest-GPU4-v3-source-20261001/independent-cloud-readback/source.bytes'
ARCHIVE_SHA='038fa8118c9540d91073fbb8bf594fb6abefe8347f13bb69dfb006d27fcfda03'
EXPORTER='tools/event_track_v2x/train_exclusive_paper_priority.py'
GATE='transvision/models/event_track_v2x/exclusive_priority_admission.py'
TRAINER='transvision/models/event_track_v2x/exclusive_allocation_training.py'
TRAINER_SHA='e61f2803d3e50e85c409659821e3395a38f2c7aadae8d831efb377b335610adf'


def replace_once(text,old,new):
    assert text.count(old)==1,'source contract differs; inspect rather than applying broad replacements'
    return text.replace(old,new)


def render_exporter(text):
    text=replace_once(text,
        "    if any(plan['source_sha256'].get(k) != v for k, v in sources.items()):\n        raise ValueError('teacher feature/solver sources differ')",
        "    from transvision.models.event_track_v2x.exclusive_priority_admission import verify_teacher_sources\n    verify_teacher_sources(plan['source_sha256'], sources)")
    text=replace_once(text,
        "'timings.json', 'resources.json'}",
        "'timings.json', 'resources.json', 'collection-binding.json'}")
    text=replace_once(text,
        "                         target=target_admission, target_sha256=target_admission_sha256)",
        "                         target=target_admission, target_sha256=target_admission_sha256,\n                         collection_binding=root / 'collection-binding.json',\n                         collection_binding_sha256=receipt['files']['collection-binding.json'])")
    ast.parse(text)
    return text


def prepare(output):
    assert output.resolve().is_relative_to(R/'source-freezes') and not output.exists()
    assert not any(p.is_symlink() for p in (output,*output.parents))
    parent_manifest=PARENT/'source-candidate-v7.json'
    assert sha(parent_manifest)==PARENT_MANIFEST_SHA and sha(ARCHIVE)==ARCHIVE_SHA
    parent=json.loads(parent_manifest.read_bytes())
    for name,digest in parent['candidate_files'].items():assert sha(PARENT/'candidate-v7'/name)==digest
    assert parent['candidate_files'][TRAINER]==TRAINER_SHA
    bridge=PARENT/'run_linux_after_capacity_targets_portable_v2.py';assert sha(bridge)==BRIDGE_SHA
    spec=importlib.util.spec_from_file_location('unchanged_exact_core_source_overlay',bridge)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    output.mkdir(parents=True,exist_ok=False)
    candidate=output/'candidate';shutil.copytree(PARENT/'candidate-v7',candidate)
    (candidate/EXPORTER).write_text(render_exporter((candidate/EXPORTER).read_text()))
    gate=Path(__file__).resolve().with_name('rbf_final_priority_admission.py')
    shutil.copyfile(gate,candidate/GATE)
    files={name:sha(candidate/name) for name in parent['candidate_files']}
    changed={name for name,digest in files.items() if digest!=parent['candidate_files'][name]}
    assert changed=={EXPORTER,GATE} and files[TRAINER]==TRAINER_SHA
    source=output/'source';source.mkdir()
    with tarfile.open(ARCHIVE,'r:gz') as archive:
        members=archive.getmembers()
        assert len({m.name for m in members})==len(members)
        assert all((m.isfile() or m.isdir()) and not m.issym() and not m.islnk()
            and not Path(m.name).is_absolute() and '..' not in Path(m.name).parts for m in members)
        archive.extractall(source,filter='data')
    manifest=dict(parent,candidate_files=files)
    module.apply_source_overlay(source,candidate,manifest)
    assert sha(source/TRAINER)==TRAINER_SHA
    for name,path in ((EXPORTER,source/EXPORTER),(GATE,source/GATE)):ast.parse(path.read_bytes(),filename=name)
    records={str(p.relative_to(source)):dict(sha256=sha(p),bytes=p.stat().st_size)
        for p in sorted(source.rglob('*')) if p.is_file()}
    proof=output/'preparation.json'
    new(proof,dict(kind='rbf_final_refit_priority_export_consumer_source_preparation_v1',
        created_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),sources=records,
        parent_source_candidate_sha256=PARENT_MANIFEST_SHA,parent_exact_overlay_bridge_sha256=BRIDGE_SHA,
        original_source_archive_sha256=ARCHIVE_SHA,overlay_candidate_files=files,
        changed_candidate_files=sorted(changed),teacher_runtime_source_changes=False,
        model_optimizer_fit_checkpoint_source_unchanged=True,trainer_sha256=TRAINER_SHA,
        builder_sha256=sha(__file__),admission_gate_sha256=sha(gate),
        target_driver_source_freeze_sha256='3a1b58959479a205b56235f82fd83af78ba3a0a419e87b0678d66de3e0a6c620',
        actual_new_Linux_runtime_verified=False,actual_teacher_export_executed=False,priority_fit_started=False,
        strict_pipeline_isolated_selection=False,labels_are_model_not_true_identity_risk=True,
        pending=['completed final-model teacher and independent target admission',
            'final-model live-provenance export/fit bridge and actual consumer runtime admission',
            'full exported data readback, priority fitting, independent checkpoint and learned replay acceptance'],
        full_Stage2_complete=False,paper_performance_complete=False))
    register(proof,'rbf-final-refit-priority-export-consumer-source')
    print(json.dumps(dict(preparation=str(proof),sha256=sha(proof),source_files=len(records),teacher_export=False,priority_fit=False)),flush=True)
    return proof


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
    prepare(parser.parse_args().output)
