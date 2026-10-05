"""Opt-in full real-sequence serial/batched runtime qualification; no GPU claim."""
import argparse
import ast
import base64
import datetime
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import traceback
import types
import zlib

from rbf_nested_seen_val_v2_common import R, new, register, sha


def setup(root):
    root.mkdir(exist_ok=False)
    dispatch = R/'receipts/rbf-final-refit-full-train-exclusive-forest-GPU-dispatch-20261004.json'
    job = next(j for j in json.loads(dispatch.read_bytes())['jobs'] if j['seed'] == 2027)
    plan = job['plan']
    reference = R/'source-freezes/rbf-original-real-prefix-GPU-function-profile-executor-v1-20261004'
    preparation = json.loads((reference/'preparation.json').read_bytes())
    for key in ('source', 'events', 'checkpoint', 'configuration', 'exclusive_patches', 'source_replacements'):
        assert plan[key] == preparation['plan'][key]
    bootstrap = reference/'bootstrap.py'
    assert sha(bootstrap) == preparation['plan']['bootstrap_sha256']
    constants = {}
    for node in ast.parse(bootstrap.read_text()).body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            if node.targets[0].id in ('PATCHES', 'REPLACEMENTS'):
                constants[node.targets[0].id] = ast.literal_eval(node.value)
    original = R/'artifacts/rbf-original-joint-coupled-forest-GPU4-v3-source-20261001/independent-cloud-readback/source.bytes'
    assert sha(original) == plan['source']['sha256']
    source = root/'serial-source'
    source.mkdir()
    with tarfile.open(original) as archive:
        members = archive.getmembers()
        assert len({m.name for m in members}) == len(members)
        assert all((m.isfile() or m.isdir()) and not m.issym() and not m.islnk()
            and not Path(m.name).is_absolute() and '..' not in Path(m.name).parts for m in members)
        archive.extractall(source, filter='data')
    for category in ('PATCHES', 'REPLACEMENTS'):
        for name, spec in constants[category].items():
            path = source/name
            if category == 'PATCHES':
                assert not path.exists() and plan['exclusive_patches'][name] == spec['sha256']
            else:
                assert sha(path) == spec['before_sha256']
                assert plan['source_replacements'][name] == {k: spec[k] for k in ('before_sha256', 'sha256')}
            raw = zlib.decompress(base64.b64decode(spec['data']))
            import hashlib
            assert hashlib.sha256(raw).hexdigest() == spec['sha256']
            path.write_bytes(raw)
    candidate = root/'batched-source'
    shutil.copytree(source, candidate)
    frozen = R/'source-freezes/rbf-batched-row-online-adapter-candidate-v1-20261004'
    frozen_spec = json.loads((frozen/'source-freeze.json').read_bytes())
    assert sha(frozen/'source-freeze.json') == '553b316e776b5f9bea205e9409e81c6e3af48288d0f315ed27ef1f943fa15b89'
    for name in ('batched_row_context_scoring.py', 'batched_persistent_cache_stream.py'):
        path = frozen/name
        assert sha(path) == frozen_spec['sources'][name]['sha256']
        with (candidate/'transvision/models/event_track_v2x'/name).open('xb') as stream:
            stream.write(path.read_bytes())
    runtime = candidate/'transvision/models/event_track_v2x/exclusive_paper_runtime.py'
    old = 'from .persistent_cache_stream import PersistentForestCacheStream'
    replacement = 'from .batched_persistent_cache_stream import create_batched_cache_stream as PersistentForestCacheStream'
    text = runtime.read_text()
    assert text.count(old) == 1
    runtime.write_text(text.replace(old, replacement))
    sources = {role: {str(p.relative_to(folder)): sha(p) for p in folder.rglob('*.py')}
        for role, folder in [('serial', source), ('batched', candidate)]}
    checkpoint = R/'artifacts/rbf-all-class-final-refit-independent-byte-freeze-v1-20261004/seed2027'
    assert sha(checkpoint/'checkpoint') == plan['checkpoint']['sha256']
    ck = json.loads((checkpoint/'checkpoint').read_bytes())
    (root/'checkpoint').mkdir()
    shutil.copyfile(checkpoint/'checkpoint', root/'checkpoint/checkpoint.json')
    weights = checkpoint/'archive-unpack/training/seed-2027/weights.pt'
    assert sha(weights) == ck['weights']['sha256']
    shutil.copyfile(weights, root/'checkpoint/weights.pt')
    events = R/'artifacts/rbf-original-cache-CPU-metadata-export-v1-20261001/seed2027/events.json'
    assert sha(events) == plan['events']['sha256']
    cache = R/'artifacts/rbf-original-nested-cache-replay-input-readback-v1-20261001/seed2027/cache-unpack/cache-v2'
    assert sha(cache/'manifest.json') == plan['cache_manifest']['sha256']
    selected = [e for e in json.loads(events.read_bytes())['events'] if e['sequence_id'] == '0000']
    assert selected
    new(root/'input-binding.json', dict(seed=2027, sequence_id='0000', expected_events=len(selected),
        original_dispatch_sha256=sha(dispatch), original_GPU_task=job['task_id'], configuration=plan['configuration'],
        cache_root=str(cache), cache_manifest_sha256=plan['cache_manifest']['sha256'],
        events_path=str(events), events_sha256=sha(events), checkpoint_sha256=sha(checkpoint/'checkpoint'),
        source_hashes=sources, runner_sha256=sha(__file__), CPU_threads_per_process=1,
        GPU_or_full_dataset_or_paper_acceptance=False, selection='all original events in sequence 0000, no query or GT filtering'))


def work(root, role):
    binding = json.loads((root/'input-binding.json').read_bytes())
    source = root/(role+'-source')
    for name, checksum in binding['source_hashes'][role].items():
        assert sha(source/name) == checksum
    for name in ('transvision', 'transvision.models', 'transvision.models.event_track_v2x'):
        module = types.ModuleType(name); module.__path__ = [str(source.joinpath(*name.split('.')))]; sys.modules[name] = module
    import torch
    torch.set_num_threads(1)
    from transvision.models.event_track_v2x.forest_cache_stream import VerifiedForestCache
    from transvision.models.event_track_v2x.forest_training_checkpoint import load_identity_checkpoint
    from transvision.models.event_track_v2x.forest_training_data import frozen_cache_identity
    from transvision.models.event_track_v2x.forest_tracking import PaperForestTrackingConfig
    from transvision.models.event_track_v2x.paper_protocol import PaperProtocol
    from transvision.models.event_track_v2x.exclusive_paper_runtime import replay, default_configuration
    config = binding['configuration']
    assert config == default_configuration('rbf', allocation='bound', width=4)
    cache = VerifiedForestCache(Path(binding['cache_root']), binding['cache_manifest_sha256'])
    scorer, ck = load_identity_checkpoint(root/'checkpoint', binding['checkpoint_sha256'], config=PaperForestTrackingConfig(**config['state']), device='cpu')
    assert frozen_cache_identity(cache) == ck['frozen_cache_identity']
    path = Path(binding['events_path']); assert sha(path) == binding['events_sha256']
    events = [e for e in json.loads(path.read_bytes())['events'] if e['sequence_id'] == binding['sequence_id']]
    assert len(events) == binding['expected_events']
    model_binding = dict(candidate_protocol='rbf-all-class-top64-v1', dataset='spd', fit_split='train',
        checkpoint_sha256=binding['checkpoint_sha256'], model_sha256=ck['model_sha256'], seed=2027, frozen_cache_identity=ck['frozen_cache_identity'])
    with torch.inference_mode():
        replay(cache, events, root/(role+'-replay'), protocol=PaperProtocol('spd', 'train'),
            configuration=config, scorer=scorer, model_binding=model_binding)


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--role', choices=('serial', 'batched'))
    args = parser.parse_args(); root = args.output.resolve()
    assert root.is_relative_to(R/'artifacts')
    if args.role:
        try:
            work(root, args.role)
        except BaseException as error:
            new(root/(args.role+'-runner-failure.json'), dict(error_type=type(error).__name__, message=str(error), traceback=traceback.format_exc()))
            raise
        return
    setup(root)
    children = []
    for role in ('serial', 'batched'):
        command = [sys.executable, str(Path(__file__).resolve()), '--output', str(root), '--role', role]
        log = (root/(role+'.log')).open('xb')
        child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT); log.close()
        children.append((role, child, command))
    launch = root/'launch.json'
    new(launch, dict(kind='rbf_real_full_sequence_serial_batched_CPU_pair_launch_v1',
        created_at=datetime.datetime.now(datetime.timezone.utc).isoformat(), parent_pid=os.getpid(),
        processes=[dict(role=r, pid=p.pid, command=c) for r,p,c in children],
        binding_sha256=sha(root/'input-binding.json'), ETA='unknown until real event progress', experiment_accepted=False))
    register(launch, 'rbf-real-full-sequence-serial-batched-CPU-pair-launch')
    print(json.dumps(dict(launch=str(launch), processes=[dict(role=r,pid=p.pid) for r,p,c in children])), flush=True)
    results = {role: process.wait() for role, process, command in children}
    completion = root/'process-completion.json'
    new(completion, dict(exit_codes=results, both_processes_exited_zero=all(code == 0 for code in results.values()),
        independent_numeric_and_decision_acceptance_pending=True, GPU_memory_target_admitted=False))
    register(completion, 'rbf-real-full-sequence-serial-batched-process-terminal')
    print(json.dumps(dict(completion=str(completion), exit_codes=results)), flush=True)


if __name__ == '__main__':
    main()
