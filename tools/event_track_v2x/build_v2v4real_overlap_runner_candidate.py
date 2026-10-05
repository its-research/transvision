#!/usr/bin/env python3
"""Build a hash-pinned non-formal v1 detector runner candidate."""
import hashlib
from pathlib import Path

PINNED_RUNNER_SHA256 = {
    '141c3293eb5e8f897f4e75392dca48622ba72358fec246f6aaa8524ea8b54146',
    '094dc725ff6a6dac9be863c204e4f734fdd49ff5e2d8430e34c15aac97017304',
}
MEMBERSHIP_TASK = '14e1ad745c544ede80a6ad7b460d8157'
MEMBERSHIP_SHA = '0f429069e56c1525484d32cb7c9ec4affd49bc5eee5a724d16f3cf44a6c9b4d9'


def build(source: Path, output: Path) -> str:
    raw = source.read_bytes()
    if hashlib.sha256(raw).hexdigest() not in PINNED_RUNNER_SHA256:
        raise ValueError('historical runner differs from reviewed source')
    content = raw.decode()
    edits = (
        ('Prepare frozen V2V4Real train roles and execute one formal detector seed.',
         'Prepare overlap-controlled v1 train roles and execute one diagnostic detector seed.'),
        ('from tools.event_track_v2x.materialize_v2v4real_detector_partition import materialize',
         'from tools.event_track_v2x.materialize_v2v4real_overlap_controlled import materialize'),
        ("PROTOCOL_ID = 'v2v4real-nominal-10hz-formal-v1'",
         "PROTOCOL_ID = 'v2v4real-nominal-10hz-formal-v1'\nMEMBERSHIP_TASK = '" + MEMBERSHIP_TASK + "'\nMEMBERSHIP_SHA = '" + MEMBERSHIP_SHA + "'"),
        ("task_name='V2V4Real formal detector training'",
         "task_name='V2V4Real overlap-controlled v1 diagnostic detector training'"),
        ("raise ValueError('formal detector seed/topology/worker differs')",
         "raise ValueError('diagnostic detector seed/topology/worker differs')"),
        ("partition_path = fetch(PARTITION_TASK, 'nested-partition', PARTITION_SHA)",
         "partition_path = fetch(PARTITION_TASK, 'nested-partition', PARTITION_SHA)\n    membership_path = fetch(MEMBERSHIP_TASK, 'membership', MEMBERSHIP_SHA)"),
        ('materialize(native_roots, partition_path, PARTITION_SHA, materialized)',
         'materialize(native_roots, partition_path, PARTITION_SHA,\n                    membership_path, MEMBERSHIP_SHA, materialized)'),
        ("or receipt.get('paper_eligible') is not True):",
         "or receipt.get('paper_eligible') is not False\n                or receipt.get('formal_independent_test_eligible') is not False\n                or receipt.get('variant_id') != 'v2v4real-train-overlap-controlled-v1'):"),
        ("kind='v2v4real_detector_clearml_publication_v1'",
         "kind='v2v4real_overlap_controlled_detector_diagnostic_publication_v1'"),
        ('partition_sha256=PARTITION_SHA, seed=seed, world_size=world_size,',
         'partition_sha256=PARTITION_SHA, membership_task_id=MEMBERSHIP_TASK,\n            membership_sha256=MEMBERSHIP_SHA,\n            variant_id=\'v2v4real-train-overlap-controlled-v1\',\n            paper_eligible=False, formal_independent_test_eligible=False,\n            seed=seed, world_size=world_size,'),
    )
    for before, after in edits:
        if content.count(before) != 1:
            raise ValueError(f'expected one reviewed occurrence: {before!r}')
        content = content.replace(before, after)
    if output.exists():
        raise ValueError('fresh candidate output required')
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(content)
    return hashlib.sha256(output.read_bytes()).hexdigest()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    print(build(args.source, args.output))
