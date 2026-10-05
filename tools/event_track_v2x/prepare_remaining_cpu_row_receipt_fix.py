"""Freeze the same receipt-only correction for pending main and Top1 readers."""
import json
from rbf_nested_seen_val_v2_common import R, new, register, sha


def main():
    correction = R/'source-freezes/rbf-final-refit-fixed-topK-full-independent-CPU-v2-receipt-rows-20261004'
    contract = json.loads((correction/'source-freeze.json').read_bytes())
    fixed = correction/'final_cache203_receipt_v2.py'
    assert sha(fixed) == contract['sources'][fixed.name]['sha256']
    for variant, oldname, newname in (
        ('main', 'rbf-final-refit-full-forest-independent-CPU-v3-20261004', 'rbf-final-refit-full-forest-independent-CPU-v4-receipt-rows-20261004'),
        ('top1', 'rbf-final-refit-fixed-Top1-full-independent-CPU-v1-20261004', 'rbf-final-refit-fixed-Top1-full-independent-CPU-v2-receipt-rows-20261004')):
        old = R/'source-freezes'/oldname
        spec = json.loads((old/'source-freeze.json').read_bytes())
        payload = {}
        for name, entry in spec['sources'].items():
            digest = entry if isinstance(entry,str) else entry['sha256']
            assert sha(old/name) == digest
            payload[name] = (old/name).read_bytes()
        reference_key = 'unchanged_independent_sources' if variant == 'main' else 'unchanged_references'
        for item in spec[reference_key]: assert sha(item['path']) == item['sha256']
        if variant == 'main':
            original = payload['final_cache203.py'].decode()
            needle = '**admission.binding,events=len(events)'
            replacement = '**{("model_total_rows" if k == "rows" else k): v for k, v in admission.binding.items()},events=len(events)'
            assert original.count(needle) == 1 and original.replace(needle,replacement).encode() == fixed.read_bytes()
            payload['final_cache203.py'] = fixed.read_bytes()
        else:
            payload[fixed.name] = fixed.read_bytes()
            name = 'accept_final_refit_top1_cohort.py'
            text = payload[name].decode(); needle = 'from final_cache203 import CacheAdmission, verify_database as feature_database'
            assert text.count(needle) == 1
            payload[name] = text.replace(needle,'from final_cache203_receipt_v2 import CacheAdmission, verify_database as feature_database').encode()
            name = 'rbf_final_refit_top1_binding.py'; text = payload[name].decode()
            needle = "binding_sha256=sha(__file__), final_model_cache_constructor_sha256=sha(MAIN_CPU/'final_cache203.py'))"
            assert text.count(needle) == 1
            payload[name] = text.replace(needle,"binding_sha256=sha(__file__), final_model_cache_constructor_sha256=sha(MAIN_CPU/'final_cache203.py'), effective_cache_receipt_module_sha256=sha(directory/'final_cache203_receipt_v2.py'))").encode()
        directory = R/'source-freezes'/newname; assert not directory.exists(); directory.mkdir()
        for name, raw in payload.items():
            if name.endswith('.py'): compile(raw,name,'exec')
            with (directory/name).open('xb') as stream:stream.write(raw)
        spec['sources'] = {name:sha(directory/name) if variant == 'main' else dict(sha256=sha(directory/name),bytes=len(raw)) for name,raw in payload.items()}
        spec[reference_key].extend(dict(path=str(p),sha256=sha(p)) for p in (old/'source-freeze.json',correction/'source-freeze.json'))
        spec['receipt_serialization_fix'] = contract['receipt_serialization_fix']
        spec['new_full_cohort_executed'] = False
        new(directory/'source-freeze.json',spec);register(directory/'source-freeze.json','rbf_'+variant+'_CPU_receipt_row_namespace_fix')
        print(directory)


if __name__ == '__main__':main()
