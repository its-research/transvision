"""Create a separate val export source without altering frozen train producers.

The only inference changes are explicit TF32 disabling and generic CUDA
admission. Val scope, four deterministic sequence shards and provenance are
recorded separately. This is a cache candidate, not an online RBF evaluation.
"""
import argparse
import ast
import hashlib
import json
from pathlib import Path

SOURCE = Path('/Volumes/Data/test/recover-before-fuse/artifacts/'
              'rbf-final-refit-SPD-seen-val-matching-producer-input-recovery-v1-20261004/'
              'c1f4f328d78e4cb89cf588003d57e5f2')
ORIGINALS = {
    'run_spd_raw_cache.py': 'ccd8fe6595ded2224b954dc07469e4b53b33540ed03b49a7df5ce2cd4718bd37',
    'spd_cache_primitives.py': '311021ce12ce7ce1e1e8db8d031776c81a05215fcc91573d5639ad9672914926',
    'cooptrack_raw_query_decode.py': 'cd066ca98e1a89fbe725cd8b54218eb1ebdb1e3404aa1d124925f78688955123',
    'spd_export_training_binding.py': '931ef68ce2e3e850b7329c455486444a4543b321314087b17b04cd75145aeb39',
    'run_cooptrack_a100.py': 'be8651ce7192ef3aa1a224d8330edeb692fdfeeec8ced1ab73231e3c298b3c2c',
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def make_runner(text):
    changes = []
    def replace(old, new, count=1):
        nonlocal text
        if text.count(old) != count:
            raise ValueError('unexpected original source match: ' + old[:100])
        text = text.replace(old, new)
        changes.append(dict(before=old, after=new, occurrences=count))
    replace('Export uncropped train predictions', 'Export uncropped seen-val predictions')
    replace('manifest["kind"] != "eventtrack_train_image_pose_inputs_v1"',
            'manifest["kind"] != "eventtrack_validation_image_pose_inputs_v1"')
    replace('or manifest["val_payloads_read"]:', 'or manifest["val_payloads_read"] is not True:')
    replace('not label-free train inputs', 'not label-free official val inputs')
    replace('shard_count != 2', 'shard_count != 4')
    replace('one of two deterministic sequence shards', 'one of four deterministic sequence shards')
    replace('manifest["train_sequences"]', 'manifest["validation_sequences"]')
    replace('root / "train-split.json"', 'root / "validation-split.json"')
    replace('p.add_argument("--shard-count", type=int, default=2)',
            'p.add_argument("--shard-count", type=int, default=4)\n'
            '    p.add_argument("--training-sequences-json", type=Path, required=True)')
    replace("official_train_sequences=inputs_manifest['train_sequences']",
            'official_train_sequences=json.loads(args.training_sequences_json.read_bytes())')
    replace('if torch.cuda.device_count() != 1 or "A100" not in torch.cuda.get_device_name(0):\n'
            '        raise RuntimeError("one assigned A100 required")',
            'if torch.cuda.device_count() != 1:\n'
            '        raise RuntimeError("one assigned CUDA device required")\n'
            '    capability = torch.cuda.get_device_capability(0)\n'
            '    supported = torch.cuda.get_arch_list()\n'
            '    if "sm_%d%d" % capability not in supported:\n'
            '        raise RuntimeError("frozen runtime lacks native device architecture: " + str(capability))\n'
            '    torch.backends.cuda.matmul.allow_tf32 = False\n'
            '    torch.backends.cudnn.allow_tf32 = False\n'
            '    if torch.backends.cuda.matmul.allow_tf32 or torch.backends.cudnn.allow_tf32:\n'
            '        raise RuntimeError("TF32 must be disabled")')
    replace('"val_payloads_read": False, "score_filter_changed": True,',
            '"val_payloads_read": True, "score_filter_changed": True,\n'
            '              "evaluation_scope": "SPD-seen-val-exploratory-cache-only",\n'
            '              "actual_cuda_device": torch.cuda.get_device_name(0),\n'
            '              "compute_capability": list(capability), "torch_version": torch.__version__,\n'
            '              "compiled_cuda_architectures": supported, "TF32_enabled": False,')
    replace('"detections": counts["detections"], "appearance_valid": counts["appearance_valid"]}),',
            '"detections": counts["detections"], "appearance_valid": counts["appearance_valid"],\n'
            '                  "ETA_seconds": (time.monotonic()-started)*(len(dataset)-index-1)/(index+1),\n'
            '                  "ETA_scope": "current shard raw detector forward only"}),')
    replace('cache_role="label-free train prediction export; training provenance admission is separate"',
            'cache_role="label-free seen-val prediction export; no full online or paper acceptance"')
    ast.parse(text)
    return text, changes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(exist_ok=False)
    for name, digest in ORIGINALS.items():
        if sha(SOURCE / name) != digest:
            raise ValueError('original source byte identity differs: ' + name)
    runner, changes = make_runner((SOURCE / 'run_spd_raw_cache.py').read_text())
    (args.output / 'run_seen_val_cache.py').write_text(runner)
    for name in ORIGINALS:
        if name != 'run_spd_raw_cache.py':
            (args.output / name).write_bytes((SOURCE / name).read_bytes())
    for name in ('run_rbf_nested_seen_val_export.py', 'bootstrap_rbf_nested_seen_val_export.py',
                 'submit_rbf_nested_seen_val_export.py', 'submit_rbf_final_identity.py'):
        path = Path(__file__).with_name(name)
        ast.parse(path.read_text())
        (args.output / name).write_bytes(path.read_bytes())
    value = dict(kind='rbf_nested_seen_val_raw_export_source_preparation_v1',
                 original_source_task_id='c1f4f328d78e4cb89cf588003d57e5f2',
                 original_source_hashes=ORIGINALS, exact_runner_changes=changes,
                 inventory={p.name: dict(bytes=p.stat().st_size, sha256=sha(p))
                            for p in args.output.glob('*.py')},
                 runtime_and_input_admission=False, GPU_inference_started=False,
                 full_online_RBF_accepted=False, paper_performance_complete=False)
    (args.output / 'source-preparation.json').write_text(json.dumps(value, indent=2) + '\n')
    print(json.dumps(dict(source_prepared=True, producer_sha256=sha(args.output / 'run_seen_val_cache.py'))))


if __name__ == '__main__':
    main()
