"""Update pending memory experiment readback; preserve GPU computation and caps."""
import json
from rbf_nested_seen_val_v2_common import R,new,register,sha


def main():
    base=R/'source-freezes'
    old=base/'rbf-batched-complete-sequence-GPU-independent-reader-v1-20261004'
    newdir=base/'rbf-batched-complete-sequence-GPU-independent-reader-v2-receipt-rows-20261004'
    freeze=json.loads((old/'source-freeze.json').read_bytes())
    payload={}
    for name,record in freeze['sources'].items():
        assert sha(old/name)==record['sha256'];payload[name]=(old/name).read_bytes()
    for record in freeze['references']:assert sha(record['path'])==record['sha256']
    fixed=base/'rbf-final-refit-fixed-topK-full-independent-CPU-v2-receipt-rows-20261004/final_cache203_receipt_v2.py'
    assert sha(fixed)=='f1c2c1d5eb2b575b0029740deefb2afb37573da5302c9f1fe5ac67c792375801'
    name='read_rbf_batched_gpu_measurement.py';body=payload[name].decode()
    needle="cache_oracle=module('batch_cache203',legacy.CPU/'final_cache203.py')"
    assert body.count(needle)==1
    payload[name]=body.replace(needle,"cache_oracle=module('batch_cache203',own/'final_cache203_receipt_v2.py')").encode()
    payload[fixed.name]=fixed.read_bytes()
    newdir.mkdir(exist_ok=False)
    for name,raw in payload.items():
        if name.endswith('.py'):compile(raw,name,'exec')
        with (newdir/name).open('xb') as stream:stream.write(raw)
    freeze['sources']={name:dict(sha256=sha(newdir/name),bytes=len(raw)) for name,raw in payload.items()}
    freeze['references'].extend(dict(path=str(p),sha256=sha(p)) for p in (old/'source-freeze.json',fixed.parent/'source-freeze.json'))
    freeze['receipt_row_namespace_corrected']=True
    new(newdir/'source-freeze.json',freeze);register(newdir/'source-freeze.json','rbf_batched_GPU_reader_receipt_row_fix')
    prior=base/'rbf-batched-complete-sequence-GPU-safe-dispatch-v1-20261004'
    target=base/'rbf-batched-complete-sequence-GPU-safe-dispatch-v2-receipt-rows-20261004'
    prep=json.loads((prior/'preparation.json').read_bytes());payload={}
    for name,record in prep['sources'].items():
        assert sha(prior/name)==record['sha256'];payload[name]=(prior/name).read_bytes()
    for record in prep['references']:assert sha(record['path'])==record['sha256']
    name='submit_rbf_batched_sequence_gpu_measurement.py';body=payload[name].decode()
    assert body.count(prior.name)==1;payload[name]=body.replace(prior.name,target.name).encode()
    target.mkdir(exist_ok=False)
    for name,raw in payload.items():
        if name.endswith('.py'):compile(raw,name,'exec')
        with (target/name).open('xb') as stream:stream.write(raw)
    prep['sources']={name:dict(sha256=sha(target/name),bytes=len(raw)) for name,raw in payload.items()}
    prep['reader_path']=str(newdir/'read_rbf_batched_gpu_measurement.py')
    prep['reader_source_freeze_sha256']=sha(newdir/'source-freeze.json')
    prep['references'].extend(dict(path=str(p),sha256=sha(p)) for p in (newdir/'source-freeze.json',newdir/'read_rbf_batched_gpu_measurement.py',prior/'preparation.json'))
    prep['commands_after_core_priority']=[str(target/'submit_rbf_batched_sequence_gpu_measurement.py'),'--execute']
    prep['receipt_row_namespace_corrected']=True
    new(target/'preparation.json',prep);register(target/'preparation.json','rbf_batched_GPU_dispatch_reader_receipt_row_fix')
    print(newdir);print(target)


if __name__=='__main__':main()
