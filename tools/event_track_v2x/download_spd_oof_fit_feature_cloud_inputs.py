"""Download only the exact four admitted GT-free fit-complement archives.

Used by a separate fit-only feature runtime. Does not load labels, fit a head,
or admit held-out selection scores. Actual execution must be independently read.
"""
import hashlib
import json
from pathlib import Path
import subprocess
from clearml import Task
from run_cooptrack_official_oof_gpu4_offline_gl_v6 import download_registered_artifact, validate_archive
from materialize_spd_oof_single_fit_feature_overlay import materialize_fit_fold

CLOUD_SHA = 'eca6277da199e198fbcf07b8098421787c8eca08136c67e05761497106cdada1'
OVERLAY_ADMISSION_SHA = '0a2d53e84f346383852efadb9dec05a7064d8e0543c3bc4c2156f4bd1d04e889'
OVERLAY_TASK = '135fae3c1eb74afbb70a8574e759edfb'


def require(ok, message):
    if not ok:
        raise ValueError(message)


def admit(cloud_text, overlay_admission_text, package_manifest, fold_id):
    require(type(fold_id) is int and 0 <= fold_id < 5, 'unexpected fold')
    require(hashlib.sha256(cloud_text.encode()).hexdigest() == CLOUD_SHA,
            'cloud input evidence differs')
    require(hashlib.sha256(overlay_admission_text.encode()).hexdigest() == OVERLAY_ADMISSION_SHA,
            'overlay independent admission differs')
    cloud, overlay = json.loads(cloud_text), json.loads(overlay_admission_text)
    require(cloud['fit_overlay_task_id'] == OVERLAY_TASK
            and overlay['task_id'] == OVERLAY_TASK
            and overlay['status'] == 'independent_bytes_verified', 'overlay owner/admission differs')
    package = json.loads(Path(package_manifest).read_bytes())
    require(package['fold_id'] == fold_id and package['cohort'] == 'official-oof-fold-fit',
            'wrong training fold/package role')
    rows = sorted(cloud['heldout_folds'], key=lambda r: r['fold_id'])
    require([r['fold_id'] for r in rows] == list(range(5)), 'cloud fold index incomplete')
    return [r for r in rows if r['fold_id'] != fold_id], overlay


def download_fit_inputs(cloud_text, overlay_admission_text, package_manifest, fold_id, output):
    rows, admission = admit(cloud_text, overlay_admission_text, package_manifest, fold_id)
    output = Path(output)
    require(not output.exists(), 'fit runtime inputs are create-once')
    output.mkdir(parents=True)
    def fetch(owner, name, record, path):
        require(owner.status == 'completed', 'input owner not completed')
        item = owner.artifacts[name]
        require(item.hash == record['sha256'] and item.size == record['bytes'],
                'registered input artifact differs')
        return download_registered_artifact(item, record['sha256'], record['bytes'], path)
    overlay = output / 'overlay'
    overlay.mkdir()
    owner = Task.get_task(task_id=OVERLAY_TASK)
    files = {'fit-overlay': 'fit-feature-overlay.tar.gz', 'overlay-manifest': 'overlay-manifest.json',
             'independent-overlay-readback': 'independent-overlay-readback.json',
             'materialization-acceptance': 'prior-materialization-acceptance.json'}
    for name, file in files.items():
        fetch(owner, name, admission['artifacts'][name], overlay / file)
    manifest = json.loads((overlay / 'overlay-manifest.json').read_bytes())
    require(manifest['archive']['path'] == 'fit-feature-overlay.tar.gz', 'unexpected overlay archive filename')
    shared = output / 'shared'
    shared.mkdir()
    for index, row in enumerate(rows):
        task = Task.get_task(task_id=row['task_id'])
        folder = output / ('download-fold-%d' % row['fold_id'])
        folder.mkdir()
        item = task.artifacts['package-manifest']
        p = fetch(task, 'package-manifest', {'sha256': row['package_manifest_sha256'],
                  'bytes': item.size}, folder / 'package-manifest.json')
        m = json.loads(p.read_bytes())
        require(m['input_manifest_sha256'] == row['input_manifest_sha256']
                and m['archive'] == row['archive'] and m['gt_or_val_test_included'] is False,
                'shared input package identity/role differs')
        archive = fetch(task, 'heldout-inputs', row['archive'], folder / 'heldout-inputs.tar.gz')
        validate_archive(archive, ['inputs'])
        subprocess.run(['tar', '--no-same-owner', '-xzf', str(archive), '-C', str(folder)], check=True)
        (folder / 'inputs').rename(shared / ('fold-%d' % row['fold_id']))
        print('EVENTTRACK_PHASE_ETA ' + json.dumps({'phase': 'fit-complement-input-download',
              'fold_id': fold_id, 'shared_fold_id': row['fold_id'], 'completed_packages': index + 1,
              'total_packages': 4, 'eta_status': 'unknown', 'eta_seconds': None,
              'overall_eta': 'unknown'}), flush=True)
    return materialize_fit_fold(overlay, shared, package_manifest, fold_id, output / 'materialized')
