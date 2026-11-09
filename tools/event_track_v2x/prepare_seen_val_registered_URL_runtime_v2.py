#!/usr/bin/env python3
"""Freeze the original seen-val forest with direct registered-URL transport."""
import argparse
import ast
import hashlib
import json
from pathlib import Path


PARENT = Path('/Volumes/Data/test/recover-before-fuse/source-freezes/rbf-seen-val-bound-forest-GPU-producer-v1-20261005')
PARENT_SHA = 'd9cacf90832ca7e8ba6f25a3ce15d89bac970db857acccea8c4eb0345a60d98f'
PARENT_FREEZE_SHA = 'e3f42ad26a96620220def0c10593ba122295f04c8e3e5a57fbd04387dc1866ae'

ROUTE = '''def routed_url(url):
 original=urlsplit(url);target=urlsplit(FILES_SERVER)
 allowed=('10.100.34.118:8081','10.100.35.118:8081')
 assert original.scheme=='http' and original.netloc in allowed
 assert target.scheme=='http' and target.netloc in allowed and not target.path
 assert original.username is None and original.password is None and not original.fragment
 return url
'''

FETCH = '''def fetch(spec,p):
 t=Task.get_task(task_id=spec['task']);assert str(t.status)=='completed'
 a=t.artifacts[spec['key']];assert a.hash==spec['sha256'] and a.size==spec['bytes']
 start=time.monotonic();done=last=0;h=hashlib.sha256()
 with requests.get(routed_url(a.url),headers={'Authorization':'Bearer '+Session().token},timeout=(10,120),stream=True) as response:
  response.raise_for_status()
  if response.headers.get('Content-Length') is not None:assert int(response.headers['Content-Length'])==spec['bytes']
  with p.open('xb') as f:
   for b in response.iter_content(1024**2):
    if not b:continue
    done+=len(b);assert done<=spec['bytes'];f.write(b);h.update(b)
    if time.monotonic()-last>30:
     print(json.dumps({'phase':'actual_registered_URL_input_transfer','role':spec['key'],'completed_bytes':done,'total_bytes':spec['bytes'],'ETA_seconds':(time.monotonic()-start)*(spec['bytes']-done)/done,'ETA_scope':'this_one_asset_only'}),flush=True);last=time.monotonic()
 assert done==spec['bytes'] and h.hexdigest()==spec['sha256']
 assert sha(p)==spec['sha256']
'''


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def functions(source):
    return {n.name: n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef)}


def build(original):
    assert hashlib.sha256(original.encode()).hexdigest() == PARENT_SHA
    nodes = functions(original)
    candidate = original
    for name, replacement in [('routed_url', ROUTE), ('fetch', FETCH)]:
        segment = ast.get_source_segment(original, nodes[name])
        assert candidate.count(segment) == 1
        candidate = candidate.replace(segment, replacement.rstrip(), 1)
    compile(candidate, 'bootstrap.py', 'exec')
    changed = functions(candidate)
    preserved = {k: ast.dump(n, include_attributes=False) == ast.dump(changed[k], include_attributes=False)
                 for k,n in nodes.items() if k not in ('routed_url','fetch')}
    assert all(preserved.values())
    def literals(text):
        return {n.targets[0].id: ast.literal_eval(n.value) for n in ast.parse(text).body
                if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name)
                and n.targets[0].id in ('PATCHES','REPLACEMENTS')}
    assert literals(candidate) == literals(original)
    # No statement outside the two transport functions may change.
    def remaining(text):
        return ast.dump(ast.Module(body=[n for n in ast.parse(text).body
            if not (isinstance(n, ast.FunctionDef) and n.name in ('routed_url','fetch'))], type_ignores=[]), include_attributes=False)
    assert remaining(candidate) == remaining(original)
    return candidate, dict(kind='seen_val_direct_registered_URL_transport_only_source_control_v2',
        original_bootstrap_sha256=PARENT_SHA, original_source_freeze_sha256=PARENT_FREEZE_SHA,
        unchanged_function_AST=preserved, all_other_module_AST_unchanged=True,
        forest_worker_function_AST_unchanged=True, all_encoded_original_kernel_patches_and_replacements_unchanged=True,
        original_main_train_admission_and_seven_asset_publication_gate_unchanged=True,
        forest_configuration_budgets_tolerances_checkpoint_and_inputs_changed=False,
        changes=['Use the actual registered artifact URL without changing its host',
                 'Require byte limit and optional HTTP Content-Length match before stream completion',
                 'Independently reread the saved file SHA after complete streamed SHA matches'],
        automatic_HTTP_retry=False, denied_CUDA_branch_state_or_Linux_priority_packages_added=False,
        actual_GPU_execution=False, full_seen_val_forest_independently_accepted=False, paper_complete=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists() and not args.output.is_symlink()
    assert sha(PARENT/'source-freeze.json') == PARENT_FREEZE_SHA
    freeze = json.loads((PARENT/'source-freeze.json').read_bytes())
    for name, checksum in freeze['sources'].items():
        assert sha(PARENT/name) == checksum
    candidate, control = build((PARENT/'bootstrap.py').read_text())
    args.output.mkdir()
    (args.output/'bootstrap.py').open('x').write(candidate)
    (args.output/Path(__file__).name).open('xb').write(Path(__file__).read_bytes())
    (args.output/'source-control.json').open('x').write(json.dumps(control, sort_keys=True, indent=2)+'\n')
    result = dict(kind='rbf_seen_val_bound_forest_actual_registered_URL_source_freeze_v2',
        sources={p.name:sha(p) for p in args.output.iterdir()},
        original_source_freeze=dict(path=str(PARENT/'source-freeze.json'),sha256=PARENT_FREEZE_SHA),
        bootstrap_sha256=sha(args.output/'bootstrap.py'),
        actual_native_registered_URL_transport_readback_qualified=False,
        actual_GPU_tasks_created=0, dispatch_ready=False, full_seen_val_forest_independently_accepted=False,
        paper_performance_complete=False)
    (args.output/'source-freeze.json').open('x').write(json.dumps(result, sort_keys=True, indent=2)+'\n')
    print(json.dumps(dict(source_freeze=str(args.output/'source-freeze.json'),sha256=sha(args.output/'source-freeze.json'),
                          bootstrap_sha256=result['bootstrap_sha256'],actual_GPU_tasks_created=0)), flush=True)


if __name__ == '__main__':
    main()
