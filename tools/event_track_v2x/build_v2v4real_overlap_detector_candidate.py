#!/usr/bin/env python3
"""Build a non-formal detector trainer from the pinned historical implementation.

This preparation step does not submit a ClearML task.  It fails if the pinned
trainer changes, so a future diagnostic cannot silently inherit formal labels.
"""
import hashlib
from pathlib import Path


PINNED_SHA256 = '4efb73663adfc898c86bde3de0fccda26f99bd4b17bb1d4c55d00705fa04f2fc'


def build(source: Path, output: Path) -> str:
    raw = source.read_bytes()
    if hashlib.sha256(raw).hexdigest() != PINNED_SHA256:
        raise ValueError('historical trainer differs from reviewed source')
    text = raw.decode()
    replacements = (
        ("\"\"\"Deterministic DDP training controller for the pinned V2V4Real detector.\"\"\"",
         "\"\"\"Diagnostic DDP training on the isolated overlap-controlled v1 view.\"\"\""),
        ("from transvision.models.event_track_v2x.detection_cache_v2 import canonical, sha_file",
         "from transvision.models.event_track_v2x.detection_cache_v2 import canonical, sha_file\nfrom transvision.models.event_track_v2x.experiment_progress import ExperimentProgress"),
        ("'v2v4real_detector_partition_materialization_v1'",
         "'v2v4real_overlap_controlled_materialization_v1'"),
        ("or receipt.get('protocol_id') != PROTOCOL_ID",
         "or receipt.get('variant_id') != 'v2v4real-train-overlap-controlled-v1'\n            or receipt.get('formal_independent_test_eligible') is not False\n            or receipt.get('physical_session_provenance_verified') is not False\n            or receipt.get('membership_sha256') != '0f429069e56c1525484d32cb7c9ec4affd49bc5eee5a724d16f3cf44a6c9b4d9'\n            or len(receipt.get('detector_fit_sequences', [])) != 19\n            or len(receipt.get('calibration_fit_sequences', [])) != 6"),
        ("or receipt.get('identity_selection_included') is not False",
         "or len(receipt.get('identity_selection_sequences', [])) != 4"),
        ("allowed = {'train', 'validate', receipt_path.name}",
         "allowed = {'train', 'validate', 'identity_selection', receipt_path.name}"),
        ("for name in ('train', 'validate')):",
         "for name in ('train', 'validate', 'identity_selection')):"),
        ("'formal training requires 60 epochs and batch size 8 per rank'",
         "'diagnostic training retains 60 epochs and batch size 8 per rank'"),
        ("'formal detector training requires CUDA'",
         "'diagnostic detector training requires CUDA'"),
        ("'v2v4real_formal_detector_training_receipt_v1'",
         "'v2v4real_overlap_controlled_detector_diagnostic_training_receipt_v1'"),
        ("protocol_id=PROTOCOL_ID, official_commit=OFFICIAL_COMMIT)",
         "protocol_id=PROTOCOL_ID, official_commit=OFFICIAL_COMMIT,\n                   variant_id='v2v4real-train-overlap-controlled-v1',\n                   paper_eligible=False, formal_independent_test_eligible=False)"),
        ("paper_eligible=True)",
         "paper_eligible=False, formal_independent_test_eligible=False,\n            physical_session_provenance_verified=False,\n            variant_id='v2v4real-train-overlap-controlled-v1')"),
        ("    best = None\n    for epoch in range(args.epochs):",
         "    best = None\n    progress = ExperimentProgress('v2v4real_overlap_controlled_detector_epochs',\n                                  args.epochs) if rank == 0 else None\n    for epoch in range(args.epochs):"),
        ("            with log_path.open('ab') as stream:\n                stream.write(canonical(record))",
         "            with log_path.open('ab') as stream:\n                stream.write(canonical(record))\n            progress.update(epoch + 1)"),
    )
    for before, after in replacements:
        if text.count(before) != 1:
            raise ValueError(f'expected exactly one source occurrence: {before!r}')
        text = text.replace(before, after)
    if output.exists():
        raise ValueError('fresh candidate output required')
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(text)
    return hashlib.sha256(output.read_bytes()).hexdigest()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    print(build(args.source, args.output))
