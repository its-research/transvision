"""Preserve frozen oracles; correct only cohort/sequence row receipt naming."""
import ast
import json
from pathlib import Path
from types import SimpleNamespace

from rbf_nested_seen_val_v2_common import R, new, register, sha


def main():
    old = R/'source-freezes/rbf-final-refit-fixed-topK-full-independent-CPU-v1-20261004'
    spec = json.loads((old/'source-freeze.json').read_bytes())
    for name, item in spec['sources'].items():
        assert sha(old/name) == item['sha256']
    for item in spec['unchanged_references']:
        assert sha(item['path']) == item['sha256']
    original = R/'source-freezes/rbf-final-refit-full-forest-independent-CPU-v3-20261004/final_cache203.py'
    assert sha(original) == spec['reference_final_cache_sha256']
    text = original.read_text()
    needle = '**admission.binding,events=len(events)'
    replacement = '**{("model_total_rows" if k == "rows" else k): v for k, v in admission.binding.items()},events=len(events)'
    assert text.count(needle) == 1
    fixed = text.replace(needle, replacement)
    # The only source change is the return-receipt field name; numerical work is identical.
    assert fixed.replace(replacement, needle) == text
    tree = ast.parse(fixed)
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'verify_database')
    expression = next(n.value for n in reversed(function.body) if isinstance(n, ast.Return))
    env = dict(admission=SimpleNamespace(binding={'seed':1337, 'rows':229212}), sequence='0000',
               expected_sha='a'*64, events=[1,2], original=[1,2], previous=[1,2,3], seen={1,2},
               duplicate_count=0, contexts=3, classes=SimpleNamespace(tolist=lambda:[3,0,0]),
               maximum={}, ATOL=1e-8, RTOL=1e-8, allow_prefix=False)
    receipt = eval(compile(ast.Expression(expression), '<receipt-only-control>', 'eval'), env)
    assert receipt['rows'] == 3 and receipt['model_total_rows'] == 229212
    assert receipt['atol'] == receipt['rtol'] == 1e-8 and receipt['seed'] == 1337
    directory = R/'source-freezes/rbf-final-refit-fixed-topK-full-independent-CPU-v2-receipt-rows-20261004'
    assert not directory.exists(); directory.mkdir()
    payloads = {name:(old/name).read_bytes() for name in spec['sources']}
    payloads['final_cache203_receipt_v2.py'] = fixed.encode()
    name = 'accept_final_refit_topk_cohort.py'
    body = payloads[name].decode()
    needle = 'from final_cache203 import CacheAdmission, verify_database as feature_database'
    assert body.count(needle) == 1
    payloads[name] = body.replace(needle, 'from final_cache203_receipt_v2 import CacheAdmission, verify_database as feature_database').encode()
    name = 'rbf_final_refit_topk_binding.py'
    body = payloads[name].decode()
    needle = 'binding_sha256=sha(__file__), final_model_cache_constructor_sha256=sha(MAIN_CPU/\'final_cache203.py\'))'
    assert body.count(needle) == 1
    payloads[name] = body.replace(needle, 'binding_sha256=sha(__file__), final_model_cache_constructor_sha256=sha(MAIN_CPU/\'final_cache203.py\'), effective_cache_receipt_module_sha256=sha(directory/\'final_cache203_receipt_v2.py\'))').encode()
    for name, raw in payloads.items():
        if name.endswith('.py'): compile(raw, name, 'exec')
        with (directory/name).open('xb') as stream: stream.write(raw)
    spec['sources'] = {name:dict(sha256=sha(directory/name), bytes=len(raw)) for name,raw in payloads.items()}
    spec['unchanged_references'].append(dict(path=str(old/'source-freeze.json'),sha256=sha(old/'source-freeze.json')))
    spec['receipt_serialization_fix'] = dict(cohort_rows_key='model_total_rows',sequence_rows_key='rows',
        mathematical_operations_and_tolerances_unchanged=True,control_passed=True,
        original_failure_preserved=str(R/'artifacts/rbf-final-topk-after-readback-continuation-v2-20261004/seed1337/independent-acceptance/failure.json'))
    new(directory/'source-freeze.json',spec);register(directory/'source-freeze.json','rbf_topk_CPU_receipt_row_namespace_fix')
    print(directory)


if __name__ == '__main__': main()
