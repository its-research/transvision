"""Read a completed CUDA state candidate once, with safe byte-bound extraction.

This downloads existing artifacts only. It never dispatches, uploads, restarts
an experiment, or grants numerical acceptance. Existing partials are retained.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import tarfile

from rbf_nested_seen_val_v2_common import R, new, register, sha

TRANSPORT = R/'source-freezes/rbf-branch-state-CUDA-transport-producer-v2-witness-20261004'
TRANSPORT_SHA = '4b87a785fe0f60d01c80fd27872295857eff3a45d40c9caaae223eb06288c3e9'
MANIFEST_SHA = 'ce8bf75c2b1c29aba37088aac6f286487097d8026a32ce3a5d718cc1d1313213'
CPU_ACCEPTANCE_SHA = 'a26c27c62f3eba4b76640d2ece2cbee5d5e1a151242da59e8271da79170069f1'


def extract(archive_path, receipt_path, output):
    assert not output.exists(), 'preserve previous extraction'
    receipt = json.loads(receipt_path.read_bytes())
    expected = {}
    for name,spec in receipt['output_files'].items():
        path = Path(name)
        assert name and path.as_posix() == name and not path.is_absolute() and '..' not in path.parts
        assert type(spec['bytes']) is int and spec['bytes'] >= 0
        assert re.fullmatch('[a-f0-9]{64}',spec['sha256'])
        expected['CUDA-candidate/'+name] = spec
    expected['receipt.json'] = dict(bytes=receipt_path.stat().st_size,sha256=sha(receipt_path))
    # The producer emits these two short metadata files outside candidate output.
    extras = {'device-before-execution.json','command.json'}
    parents = {str(parent) for name in expected for parent in Path(name).parents if str(parent) != '.'}
    seen = set(); outputs = {}
    output.mkdir(parents=True,exist_ok=False)
    with tarfile.open(archive_path,'r:gz') as archive:
        for member in archive:
            path = Path(member.name)
            assert path.as_posix() == member.name and not path.is_absolute() and '..' not in path.parts
            assert member.name not in seen, 'duplicate archive member'
            seen.add(member.name)
            if member.isdir():
                assert member.name in parents
                continue
            assert member.isfile(), 'non-regular archive member'
            assert member.name in expected or member.name in extras, 'unexpected archive file'
            if member.name in expected: assert member.size == expected[member.name]['bytes']
            else: assert member.size <= 65536
            target = output/member.name; target.parent.mkdir(parents=True,exist_ok=True)
            digest = hashlib.sha256(); total = 0
            with archive.extractfile(member) as stream, target.open('xb') as destination:
                for block in iter(lambda:stream.read(8*1024**2),b''):
                    total += len(block); assert total <= member.size
                    digest.update(block); destination.write(block)
            actual = dict(bytes=total,sha256=digest.hexdigest())
            assert total == member.size
            if member.name in expected: assert actual == expected[member.name], 'output bytes differ'
            outputs[member.name] = actual
    assert set(outputs) == set(expected)|extras, 'output file coverage differs'
    assert json.loads((output/'device-before-execution.json').read_bytes()) == receipt['device']
    command = json.loads((output/'command.json').read_bytes())
    assert command['shell'] is False and command['TF32_enabled'] is False and command['working_directory'] == 'bundle'
    new(output/'independent-archive-byte-binding.json',dict(archive_sha256=sha(archive_path),files=outputs))
    return outputs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dispatch',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    own = Path(__file__).resolve().parent
    preparation = json.loads((own/'source-freeze.json').read_bytes())
    for name,spec in preparation['sources'].items(): assert sha(own/name) == spec['sha256']
    for spec in preparation['references']: assert sha(spec['path']) == spec['sha256']
    assert sha(TRANSPORT/'source-freeze.json') == TRANSPORT_SHA
    frozen = json.loads((TRANSPORT/'source-freeze.json').read_bytes())
    for name,spec in frozen['sources'].items():
        assert sha(TRANSPORT/name) == spec['sha256']
    dispatch = json.loads(args.dispatch.read_bytes())
    assert dispatch['kind'] == 'rbf_branch_state_CUDA_collision_safe_dispatch_v1'
    assert len(dispatch['jobs']) == 1
    job = dispatch['jobs'][0]; plan = job['plan']
    assert plan['recipe'] == 'rbf_one_complete_sequence_independent_root_state_CUDA_candidate_v1'
    assert plan['required_compute_GPUs'] == 1
    assert plan['CPU_acceptance_sha256'] == CPU_ACCEPTANCE_SHA
    assert plan['manifest']['sha256'] == MANIFEST_SHA
    assert plan['bootstrap_sha256'] == sha(TRANSPORT/'run_branch_state_cuda_candidate.py')
    from clearml import Task
    task = Task.get_task(task_id=job['task_id'])
    if str(task.status) != 'completed':
        print(json.dumps(dict(task_id=task.id,status=str(task.status),accepted=False,ETA='unknown')),flush=True)
        return
    params = task.get_parameters()
    assert task.data.last_worker in job['eligible_workers'], 'actual worker is outside the admitted binding'
    assert json.loads(params['General/plan']) == plan
    assert params['General/recipe_sha256'] == hashlib.sha256(json.dumps(plan,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    assert hashlib.sha256(task.data.script.diff.encode()).hexdigest() == plan['bootstrap_sha256']
    assert set(task.artifacts) == {'candidate-output','receipt'}
    assert args.output.resolve().is_relative_to(R/'artifacts')
    assert not args.output.exists(), 'preserve completed or partial reader output; inspect before recovery'
    args.output.mkdir(parents=True,exist_ok=False)
    identity = args.output/'task-bound-before-read.json'
    new(identity,dict(task_id=task.id,plan=plan,dispatch_sha256=sha(args.dispatch),
        completed_task=True,artifacts={k:dict(bytes=v.size,sha256=v.hash) for k,v in task.artifacts.items()}))
    register(identity,'rbf-branch-state-CUDA-output-read-started')
    try:
        helper = TRANSPORT/'read_rbf_final_refit_top1_outputs.py'
        spec = importlib.util.spec_from_file_location('CUDA_frozen_cloud_reader',helper)
        reader = importlib.util.module_from_spec(spec); spec.loader.exec_module(reader)
        artifacts = {}
        for key,name in [('receipt','receipt.json'),('candidate-output','candidate-output.tar.gz')]:
            artifacts[key] = reader.read_artifact(task,key,args.output/name)
        receipt = json.loads((args.output/'receipt.json').read_bytes())
        assert receipt['kind'] == 'rbf_measured_branch_state_CUDA_complete_sequence_candidate_v1'
        assert receipt['task_id'] == task.id and receipt['plan'] == plan
        assert receipt['candidate_checks_completed'] is True and receipt['failure'] is None
        files = extract(args.output/'candidate-output.tar.gz',args.output/'receipt.json',args.output/'unpack')
        path = args.output/'independent-byte-admission.json'
        new(path,dict(kind='rbf_branch_state_CUDA_output_independent_cloud_bytes_v1',task_id=task.id,
            plan=plan,dispatch_sha256=sha(args.dispatch),artifacts=artifacts,extracted_files=files,
            reader_sha256=sha(__file__),transport_freeze_sha256=TRANSPORT_SHA,
            completed_task=True,last_worker=task.data.last_worker,full_cloud_bytes_independently_read=True,
            independent_numeric_acceptance=False,full_cohort_accepted=False,paper_performance_complete=False))
        register(path,'rbf-branch-state-CUDA-output-independent-bytes')
        print(json.dumps(dict(receipt=str(path))),flush=True)
    except BaseException as error:
        path = args.output/'failure.json'
        new(path,dict(task_id=task.id,exception_type=type(error).__name__,accepted=False,
            partials_preserved=True,automatic_retry=False))
        register(path,'rbf-branch-state-CUDA-output-read-failure'); raise


if __name__ == '__main__': main()
