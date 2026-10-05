"""Freeze a data-free Linux consumer probe from qualified offline runtime code."""
import argparse
import ast
import base64
import gzip
import hashlib
import io
import json
from pathlib import Path
import shutil
import tarfile

from rbf_nested_seen_val_v2_common import R,new,register,sha

OLD=R/'source-freezes/rbf-capacity-priority-v7-Linux-CPU-import-probe-v2-20261002'
OLD_BOOTSTRAP_SHA='c66308409f4f5f190b74d7e94f9922b5adfa8385e778cb77f1a701d3d6bb2e78'
CONSUMER=R/'source-freezes/rbf-final-refit-priority-export-consumer-v1-20261005'
CONSUMER_FREEZE_SHA='e358113abf6c23bab8cd2bb27f4a2df0d9586ff9ceda88e64679f6b87071b961'
PREPARATION_SHA='0579fffe9cd61217d238fe8e0dc87aa222ae533944be349a7383f895875c09ea'
PARENT=R/'source-freezes/rbf-exclusive-capacity-priority-original-paired-export-v7-portable-v2-20261002'
PARENT_SHA='3f5c5229aef831762a447162b1bfee8921183c31fa3d1715c60c73dbfe945d0f'
BRIDGE='run_final_refit_priority_after_targets.py'


def once(text,old,new):
    assert text.count(old)==1,'probe source contract changed'
    return text.replace(old,new)


def render(original,archive,manifest):
    text='\n'.join(line for line in original.splitlines() if not line.startswith(('EMBEDDED_ARCHIVE =','EMBEDDED_MANIFEST =')))+'\n'
    text=once(text,'def main():',"EMBEDDED_ARCHIVE = "+repr(base64.b85encode(archive).decode())+'\nEMBEDDED_MANIFEST = '+repr(manifest.decode())+'\n\ndef main():')
    text=once(text,'import sys\n','import sys\nsys.dont_write_bytecode = True\nos.environ[\'PYTHONDONTWRITEBYTECODE\'] = \'1\'\n')
    text=once(text,"task_name='capacity priority v7 Linux CPU consumer import and CLI'","task_name='final-refit priority Linux CPU consumer import and CLI'")
    text=once(text,"Path('capacity-priority-consumer-import-probe-v7-runtime-v1')","Path('final-priority-consumer-import-probe-runtime-v1')")
    text=once(text,"original, consumer = base / 'original', base / 'consumer'\n        extract(original_archive, original)\n        extract(source_archive, consumer)",
        "consumer = base / 'consumer'\n        extract(source_archive, consumer)\n        original = consumer / 'source'\n        extract(original_archive, original)")
    text=once(text,"consumer / 'source-candidate-v7.json'","consumer / 'preparation.json'")
    text=once(text,"plan['source_candidate_sha256']","plan['source_preparation_sha256']")
    text=once(text,"declaration['candidate_files'].items()","declaration['overlay_candidate_files'].items()")
    text=once(text,"consumer / 'candidate-v7' / name","consumer / 'candidate' / name")
    text=once(text,"        assert callable(gate.capacity_semantics)\n        assert gate.ACTUAL_CPU_WITNESS==plan['actual_CPU_witness']\n        assert gate.CORE_REPLACEMENTS==manifest['core_replacements']",
        "        roles = json.loads((consumer / 'source-role-audit.json').read_bytes())\n        assert sha(consumer / 'source-role-audit.json') == plan['source_role_audit_sha256']\n        assert callable(gate.validate) and callable(gate.verify_teacher_sources)\n        gate.verify_teacher_sources(roles['teacher_runtime_sources'], training.allocation_sources())")
    start="        bridge = load('capacity_priority_v7_portable_bridge', consumer / 'run_linux_after_capacity_targets_portable_v2.py')"
    end="        cli(bridge, ['--help'], 'help')"
    a=text.index(start);b=text.index(end,a)+len(end)
    text=text[:a]+"""        bridge = load('final_priority_portable_bridge', consumer / 'run_final_refit_priority_after_targets.py')
        assert sha(consumer / 'run_final_refit_priority_after_targets.py') == plan['bridge_sha256']
        checked_source, checked_preparation = bridge.source_gate(consumer)
        assert checked_source == original and len(checked_preparation['sources']) == plan['full_source_member_count']
        cli(bridge, [], 'parser-rejection')
        cli(bridge, ['--help'], 'help')"""+text[b:]
    text=once(text,"import torch\n        assert torch.__version__", "import numpy as np\n        assert np.__version__ == '1.26.4'\n        import torch\n        assert torch.__version__")
    text=once(text,"kind='rbf_capacity_priority_v7_Linux_CPU_consumer_actual_import_and_CLI_probe_v2'",
        "kind='rbf_final_refit_priority_Linux_CPU_import_CLI_probe_v1', numpy_version=np.__version__, full_source_member_count_verified=len(checked_preparation['sources']), training_only_source_roles_verified=True")
    text=once(text,"kind='rbf_capacity_priority_v7_CPU_consumer_probe_failure_v2'","kind='rbf_final_refit_priority_CPU_consumer_probe_failure_v1'")
    ast.parse(text)
    return text


def archive_bytes(files):
    target=io.BytesIO()
    with gzip.GzipFile(fileobj=target,mode='wb',mtime=0) as zipped:
        with tarfile.open(fileobj=zipped,mode='w') as archive:
            for name,data in sorted(files.items()):
                path=Path(name)
                assert not path.is_absolute() and '..' not in path.parts
                item=tarfile.TarInfo(name);item.size=len(data);item.mode=0o644
                archive.addfile(item,io.BytesIO(data))
    return target.getvalue()


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
    output=parser.parse_args().output.absolute()
    assert output.is_relative_to(R/'source-freezes') and not output.exists()
    assert not any(p.is_symlink() for p in (output,*output.parents))
    assert sha(CONSUMER/'source-freeze.json')==CONSUMER_FREEZE_SHA
    assert sha(CONSUMER/'preparation.json')==PREPARATION_SHA
    assert sha(OLD/'bootstrap.py')==OLD_BOOTSTRAP_SHA
    assert sha(PARENT/'source-candidate-v7.json')==PARENT_SHA
    prep=json.loads((CONSUMER/'preparation.json').read_bytes())
    for name,item in prep['sources'].items():assert sha(CONSUMER/'source'/name)==item['sha256']
    parent=json.loads((PARENT/'source-candidate-v7.json').read_bytes())
    files={f'candidate/{n}':(CONSUMER/'candidate'/n).read_bytes() for n in prep['overlay_candidate_files']}
    for name,digest in prep['overlay_candidate_files'].items():
        assert hashlib.sha256(files['candidate/'+name]).hexdigest()==digest
    for name in ('preparation.json','source-role-audit.json',BRIDGE):files[name]=(CONSUMER/name).read_bytes()
    frozen=json.loads((CONSUMER/'source-freeze.json').read_bytes())
    assert sha(CONSUMER/BRIDGE)==frozen['sources'][BRIDGE]['sha256']
    assert sha(CONSUMER/'source-role-audit.json')==frozen['source_role_audit_sha256']
    archive=archive_bytes(files)
    manifest=dict(members={n:dict(bytes=len(v),sha256=hashlib.sha256(v).hexdigest()) for n,v in files.items()},
        source_archive_sha256=hashlib.sha256(archive).hexdigest(),core_replacements=parent['capacity_core_replacements'],
        datasets_or_weights_included=False,training_started=False)
    raw=json.dumps(manifest,sort_keys=True,separators=(',',':')).encode()
    bootstrap=render((OLD/'bootstrap.py').read_text(),archive,raw)
    output.mkdir(parents=True)
    (output/'consumer-source.tar.gz').write_bytes(archive);(output/'consumer-manifest.json').write_bytes(raw)
    (output/'bootstrap.py').write_text(bootstrap)
    for name in ('control_final_priority_runtime_probe.py','prepare_final_priority_runtime_probe.py','rbf_nested_seen_val_v2_common.py'):
        shutil.copyfile(Path(__file__).with_name(name),output/name)
    new(output/'preparation.json',dict(kind='rbf_final_refit_priority_Linux_CPU_probe_preparation_v1',
        source_preparation_sha256=PREPARATION_SHA,source_consumer_freeze_sha256=CONSUMER_FREEZE_SHA,
        bridge_sha256=sha(CONSUMER/BRIDGE),source_role_audit_sha256=sha(CONSUMER/'source-role-audit.json'),
        full_source_member_count=len(prep['sources']),consumer_member_count=len(files),
        embedded_source_sha256=sha(output/'consumer-source.tar.gz'),embedded_manifest_sha256=sha(output/'consumer-manifest.json'),
        bootstrap_sha256=sha(output/'bootstrap.py'),CPU_only=True,dataset_read=False,priority_fit=False,
        original_runtime_bootstrap_sha256=OLD_BOOTSTRAP_SHA))
    sources={p.name:sha(p) for p in output.iterdir() if p.is_file()}
    new(output/'source-freeze.json',dict(kind='rbf_final_priority_runtime_probe_source_v1',sources=sources,
        old_bootstrap_sha256=OLD_BOOTSTRAP_SHA,new_consumer_freeze_sha256=CONSUMER_FREEZE_SHA,
        source_only=True,actual_Linux_probe_completed=False,full_Stage2_complete=False,paper_performance_complete=False))
    register(output/'source-freeze.json','rbf-final-priority-linux-runtime-probe-source')
    print(json.dumps(dict(output=str(output),source_freeze_sha256=sha(output/'source-freeze.json'),
        embedded_bytes=len(archive),consumer_members=len(files),actual_probe=False)))


if __name__=='__main__':main()
