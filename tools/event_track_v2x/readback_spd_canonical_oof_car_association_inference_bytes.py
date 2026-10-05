#!/usr/bin/env python3
"""Full inference artifacts and registered-source byte readback, no acceptance shortcut."""
import argparse,hashlib,json,sys
from pathlib import Path
from clearml import Task


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main(fold):
    R=Path('/Volumes/Data/test/recover-before-fuse');journal=json.loads((R/'receipts/spd-canonical-oof-full-heldout-car-association-GPU-inference-v1-dispatch-20261001.json').read_text());job=next(r for r in journal['jobs'] if r['fold_id']==fold);t=Task.get_task(task_id=job['task_id']);assert t.status=='completed' and hashlib.sha256(t.data.script.diff.encode()).hexdigest()==journal['source_sha256'];assert set(t.artifacts)=={'actual-GPU-runtime','all-logits','all-retained-hypotheses','inference-manifest','source-event-inventory'};B=R/f'artifacts/spd-canonical-oof-fold{fold}-heldout-car-association-inference-v1-independent-readback-20261001';B.mkdir(exist_ok=True);sys.path.insert(0,str(R/'source-freezes/spd-canonical-association-compatible-GPU-source-publication-v2-20261001'));from spd_registered_artifact_chunked_candidate import download_registered_artifact
    files={'actual-GPU-runtime':'runtime.json','all-logits':'all-logits.tar.gz','all-retained-hypotheses':'all-hypotheses.jsonl','inference-manifest':'inference-manifest.json','source-event-inventory':'source-events.json'};rows={}
    for name,filename in files.items():
        a=t.artifacts[name];p=B/filename
        if not p.exists():download_registered_artifact(a,a.hash,a.size,p)
        assert sha(p)==a.hash and p.stat().st_size==a.size;rows[name]=dict(path=filename,sha256=a.hash,bytes=a.size)
    manifest=json.loads((B/'inference-manifest.json').read_text());assert manifest['fold_id']==fold and manifest['training_task_id']==job['training_task_id'] and manifest['checkpoint_sha256']==job['checkpoint_sha256'] and manifest['input_manifest_sha256']==job['input_manifest_sha256'];proof=dict(task_id=t.id,fold_id=fold,source_sha256=journal['source_sha256'],artifacts=rows,all_registered_artifact_bytes_verified=True,full_numeric_and_hypothesis_acceptance=False,paper_eligible=False);p=B/'byte-readback-receipt.json'
    if p.exists():assert json.loads(p.read_text())==proof
    else:p.write_text(json.dumps(proof,indent=2)+'\n')
    print(json.dumps(dict(task_id=t.id,fold_id=fold,all_registered_bytes_verified=True)),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--fold',type=int,choices=range(5),required=True);a=p.parse_args();main(a.fold)
