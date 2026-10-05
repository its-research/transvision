"""Stage immutable inputs for one measured CUDA branch-state candidate.

This only packages independently admitted inputs; it never starts a replay,
uploads credentials, modifies frozen sources, or grants GPU acceptance.
"""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import tarfile

from rbf_nested_seen_val_v2_common import R, new, register, sha

SOURCE = R/'source-freezes/rbf-independent-root-state-measured-CUDA-entry-v1-20261004'
SOURCE_SHA = '5704205925c7f8ec6e339d9e15e58091194922ed2ffa83b3c7fb7994020b4c3c'
REFERENCE = R/'artifacts/rbf-batched-full-real-sequence-CPU-pair-v1-20261004/serial-replay'
REFERENCE_RECEIPT_SHA = 'd30d1879c65ee84cfbbefd04eabbf1acd22d7954e30f5811779530955cd307de'
PROOF = R/'artifacts/rbf-batched-full-sequence-independent-acceptance-v2-20261004'
PROOF_SHA = 'd104416c40b65e9e6082afc41fa74ad6f7f0d21f8a7e411ebd1420ac5bc0adbc'
CANDIDATE = R/'artifacts/rbf-independent-root-state-full-sequence-CPU-v1-20261004'
WITNESS = R/'source-freezes/rbf-branch-state-CUDA-commitment-witness-v1-20261004'
WITNESS_SHA = 'e4174398fa8386af4ecbf478dd79a7de8fd24e346be046d71c193cbe989d5222'


def inputs(acceptance):
    assert sha(SOURCE/'preparation.json') == SOURCE_SHA
    assert sha(REFERENCE/'receipt.json') == REFERENCE_RECEIPT_SHA
    assert sha(PROOF/'independent-sequence-receipt.json') == PROOF_SHA
    accepted = json.loads(acceptance.read_bytes())
    assert accepted['kind'] == 'rbf_optimized_branch_state_single_sequence_independent_correspondence_and_fresh_state_v3'
    assert accepted['events'] == 195 and accepted['states'] == 56389
    assert accepted['atol'] == accepted['rtol'] == 1e-8
    assert accepted['request_commitments_independently_bound'] == 390
    assert accepted['branch_commitments_independently_bound'] > 0
    assert accepted['complete_audit_and_raw_structural_table_correspondence'] is True
    assert accepted['reference_acceptance_sha256'] == PROOF_SHA
    assert sha(WITNESS/'source-freeze.json') == WITNESS_SHA
    witness = json.loads((WITNESS/'source-freeze.json').read_bytes())
    for name, record in witness['sources'].items():
        assert sha(WITNESS/name) == record['sha256'] and (WITNESS/name).stat().st_size == record['bytes']
    assert accepted['source_sha256'] == sha(WITNESS/'accept_optimized_branch_state_sequence.py')
    for key, name in [('correspondence_sha256','structural-and-event-correspondence.json'),
                      ('fresh_state_sha256','fresh-state-independent.json')]:
        assert sha(acceptance.parent/name) == accepted[key]
    candidate = json.loads((CANDIDATE/'candidate-check.json').read_bytes())
    assert accepted['candidate_database_sha256'] == candidate['database_sha256']['candidate']
    assert sha(CANDIDATE/'input-binding.json') == candidate['input_binding_sha256']
    source = json.loads((SOURCE/'preparation.json').read_bytes())
    files = {}
    for name, record in source['sources'].items():
        path = SOURCE/name
        assert path.resolve().is_relative_to(SOURCE.resolve()) and not path.is_symlink()
        assert sha(path) == record['sha256'] and path.stat().st_size == record['bytes']
        files['execution/'+name] = path
    # The admitted CPU implementation must be exactly the model code staged.
    binding = json.loads((CANDIDATE/'input-binding.json').read_bytes())
    for name, checksum in binding['source_files'].items():
        assert sha(files['execution/'+name]) == checksum
    files['execution/preparation.json'] = SOURCE/'preparation.json'
    for name in ('export_branch_state_commitment_witness.py','accept_optimized_branch_state_sequence.py','rbf_nested_seen_val_v2_common.py'):
        files['execution/tools/event_track_v2x/'+name] = WITNESS/name
    files['execution/witness-source-freeze.json'] = WITNESS/'source-freeze.json'
    reference = json.loads((REFERENCE/'receipt.json').read_bytes())
    database = reference['databases']['0000']
    db = REFERENCE/database['path']
    assert db.parent == REFERENCE and sha(db) == database['sha256']
    assert candidate['recorded_reference']['database_sha256'] == database['sha256']
    files['reference/receipt.json'] = REFERENCE/'receipt.json'
    files['reference/'+database['path']] = db
    proof = json.loads((PROOF/'independent-sequence-receipt.json').read_bytes())
    files['reference-admission/independent-sequence-receipt.json'] = PROOF/'independent-sequence-receipt.json'
    for role, checksum in proof['reports'].items():
        assert role in ('serial','batched')
        path = PROOF/(role+'-independent.json'); assert sha(path) == checksum
        files['reference-admission/'+path.name] = path
    for name in ('acceptance.json','structural-and-event-correspondence.json','fresh-state-independent.json'):
        files['CPU-candidate-admission/'+name] = acceptance.parent/name
    for name in ('candidate-check.json','input-binding.json','process-completion.json'):
        files['CPU-candidate/'+name] = CANDIDATE/name
    return files


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--CPU-acceptance', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists(), 'preserve any previous package or failure'
    files = inputs(args.CPU_acceptance)
    args.output.mkdir(parents=True, exist_ok=False)
    inventory = {name:dict(bytes=path.stat().st_size,sha256=sha(path)) for name,path in sorted(files.items())}
    command = ['python','execution/tools/event_track_v2x/qualify_batched_branch_states_gpu.py',
        '--replay','reference','--receipt-sha256',REFERENCE_RECEIPT_SHA,'--sequence','0000','--events','195',
        '--recorded-reference-acceptance','reference-admission/independent-sequence-receipt.json',
        '--reference-acceptance-sha256',PROOF_SHA,'--device','cuda:0','--max-batch','64','--output','CUDA-candidate']
    manifest = args.output/'manifest.json'
    new(manifest,dict(kind='rbf_branch_state_CUDA_exact_source_and_admitted_reference_bundle_v1',
        files=inventory,command_relative_to_bundle_root=command,
        CPU_acceptance_sha256=sha(args.CPU_acceptance),preparer_sha256=sha(__file__),
        portable_branch_commitment_witness_source_freeze_sha256=WITNESS_SHA,
        GPU_devices_required=1,actual_GPU_execution=False,production_promotion_allowed=False,
        memory_target_percent=[75,80],memory_target_achieved=False))
    files['manifest.json'] = manifest
    archive_path = args.output/'inputs.tar.gz'
    with archive_path.open('xb') as raw, gzip.GzipFile(filename='',mode='wb',fileobj=raw,mtime=0) as compressed:
        with tarfile.open(fileobj=compressed,mode='w|') as archive:
            for name,path in sorted(files.items()):
                assert not Path(name).is_absolute() and '..' not in Path(name).parts
                info = archive.gettarinfo(str(path),arcname=name)
                info.mtime=0;info.uid=info.gid=0;info.uname=info.gname='';info.mode=0o644
                with path.open('rb') as stream:archive.addfile(info,stream)
    expected = dict(inventory, **{'manifest.json':dict(bytes=manifest.stat().st_size,sha256=sha(manifest))})
    observed = {}
    with tarfile.open(archive_path,'r:gz') as archive:
        for member in archive:
            assert member.isfile() and member.name in expected and member.name not in observed
            digest=hashlib.sha256();total=0
            with archive.extractfile(member) as stream:
                for block in iter(lambda:stream.read(8*1024**2),b''):
                    digest.update(block);total+=len(block)
            observed[member.name]=dict(bytes=total,sha256=digest.hexdigest())
    assert observed == expected
    receipt = args.output/'package-readback.json'
    new(receipt,dict(kind='rbf_branch_state_CUDA_input_package_independent_stream_readback_v1',
        archive_sha256=sha(archive_path),archive_bytes=archive_path.stat().st_size,
        manifest_sha256=sha(manifest),members=len(observed),full_archive_stream_hashes_match=True,
        CPU_acceptance_sha256=sha(args.CPU_acceptance),preparer_sha256=sha(__file__),
        remote_upload_complete=False,GPU_task_created=False,actual_GPU_execution=False))
    register(receipt,'rbf-branch-state-CUDA-input-package');print(json.dumps(dict(receipt=str(receipt))),flush=True)


if __name__ == '__main__':main()
