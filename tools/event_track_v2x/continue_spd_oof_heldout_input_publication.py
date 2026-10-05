"""Wait for an existing fit upload, then publish admitted inference inputs once."""
import hashlib,json,subprocess,sys,time
from pathlib import Path
from clearml import Task
ROOT=Path('/Volumes/Data/test/recover-before-fuse')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 source=Path(__file__)
 if source.read_bytes()!=(ROOT/'source-freezes/spd-oof-heldout-inference-package-upload-20261001'/source.name).read_bytes():raise ValueError('continuation source differs from freeze')
 while True:
  task=Task.get_task(task_id='99f20b72842240b795ab94bc72b87a07')
  if task.status=='completed':break
  if task.status not in ('created','queued','in_progress'):raise RuntimeError('fit upload is terminal without completion')
  print('INPUT_PUBLICATION_WAITING_EXISTING_UPLOAD',task.id,str(task.status),'ETA_unknown',flush=True);time.sleep(30)
 package=ROOT/'artifacts/spd-official-oof-fivefold-20260930/fold-4-package'
 if not (package/'clearml-upload-acceptance.json').is_file():raise ValueError('completed upload lacks local acceptance')
 folder=Path(__file__).parent
 if not (package/'clearml-independent-readback.json').exists():
  subprocess.run([sys.executable,'-u',str(folder/'accept_spd_oof_package_remote112_v3.py'),'--fold','4','--package',str(package)],check=True)
 admission=ROOT/'receipts/spd-canonical-oof-fivefold-heldout-inference-package-admission-20261001.json'
 if sha(admission)!='a38ed021adb55158c9546d33f94280efe83cbd44e88dd68cefad79f7c374b98c':raise ValueError('input archive admission changed')
 for row in json.loads(admission.read_bytes())['folds']:
  path=ROOT/('artifacts/spd-canonical-oof-heldout-inference-packages-20261001/fold-%d'%row['fold_id'])
  subprocess.run([sys.executable,'-u',str(folder/'upload_spd_oof_heldout_inference_package.py'),'--package',str(path),'--independent-receipt-sha256',row['independent_receipt_sha256']],check=True)
 print('FIVEFOLD_INPUT_PUBLICATION_DONE_REMOTE_READBACK_PENDING',flush=True)
if __name__=='__main__':main()
