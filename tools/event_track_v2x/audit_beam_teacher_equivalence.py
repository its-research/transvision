#!/usr/bin/env python3
"""Compare a complete beam teacher with its frozen ordinary train inference.

No GT, tuning, training or metrics. Negative output-equivalence findings are
preserved; mismatched input/configuration/factors are invalid comparisons.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tools.event_track_v2x.assemble_allocation_teachers import _pairs, _leaf, _unchanged
from tools.event_track_v2x.audit_train_inference_comparison import inspect
from tools.event_track_v2x.run_train_inference_diagnostic import diagnostic_sources
from transvision.models.event_track_v2x.detection_cache_v2 import sha_file
from transvision.models.event_track_v2x.forest_training_data import _new_json

COMMON = ('configuration', 'cache_sha256', 'cooperative_metadata_sha256',
          'identity_checkpoint_sha256', 'ego_pose_table_sha256',
          'allocation_policy_signature', 'class_scope', 'cache_split')


def compare(teacher, inference, cooperative_metadata, output, *, allow_fixture=False):
    output = Path(output).absolute()
    roots = [Path(reference[0]).resolve() for reference in (teacher, inference)]
    if (output.exists() or output.is_symlink() or any(p.is_symlink() for p in output.parents)
            or any(root == output or root in output.parents for root in roots)):
        raise ValueError('fresh output outside both immutable runs required')
    expected, pair_sha = _pairs(cooperative_metadata, allow_fixture)
    leaf = _leaf(*teacher, expected, pair_sha, allow_fixture)
    ordinary = inspect(*inference, allow_fixture=allow_fixture)
    left, right = leaf['plan'], ordinary['plan']
    if (ordinary['backend'] != 'beam_recovery' or left['configuration'].get('enable_recovery') is not True
            or any(left.get(k) != right.get(k) for k in COMMON)
            or right['selected_sequence'] != leaf['scene']
            or right['scorer_signature'] != left['allocation_training_binding']['factor_scorer_signature']):
        raise ValueError('same enabled beam configuration, sequence, checkpoint and inputs required')
    current = diagnostic_sources()
    # A newly added, unconsumed model module does not alter a sealed past run.
    # Every file actually bound by either run must still match, and all teacher
    # dependencies must have been present in the ordinary run's own binding.
    if (any(current.get(p) != h for p, h in right['source_sha256'].items())
            or any(right['source_sha256'].get(p) != h for p, h in left['source_sha256'].items())):
        raise ValueError('current inference and shared teacher sources required')
    before, after = leaf['report']['events'], ordinary['events']
    def schedule(events):
        return [(e['sequence_id'], e['frame_id'], e['reference_us'], e['decision_us']) for e in events]
    if schedule(before) != schedule(after):
        raise ValueError('same ordered events and decision times required')
    if leaf['report']['factor_stream_sha256'] != ordinary['factor_stream_sha256']:
        raise ValueError('actual factor stream differs; cannot isolate teacher probes')
    changed = [a['frame_id'] for a, b in zip(before, after)
               if a['output_payload_sha256'] != b['output_payload_sha256']]
    changed_ids = [a['frame_id'] for a, b in zip(before, after)
                   if a['output_ids_sha256'] != b['output_ids_sha256']]
    _unchanged([leaf])
    if diagnostic_sources() != current or sha_file(cooperative_metadata) != pair_sha:
        raise ValueError('bound sources or cohort changed during comparison')
    # Revalidate the ordinary artifacts after examining the complete teacher.
    if inspect(*inference, allow_fixture=allow_fixture) != ordinary:
        raise ValueError('inference artifacts changed during comparison')
    result = dict(kind='beam_teacher_output_equivalence_v1', status='complete',
        sequence_id=leaf['scene'], frames=len(before), fixture_only=allow_fixture,
        complete_selected_sequence_verified=not allow_fixture,
        teacher_directory=str(roots[0]), inference_directory=str(roots[1]),
        teacher_receipt_sha256=teacher[1], inference_receipt_sha256=inference[1],
        teacher_predictions_sha256=leaf['report']['predictions_sha256'],
        inference_predictions_sha256=ordinary['predictions_sha256'],
        byte_identical_predictions=leaf['report']['predictions_sha256'] == ordinary['predictions_sha256'],
        output_payload_changed_frames=changed, output_id_changed_frames=changed_ids,
        actual_factors_identical=True, factor_stream_sha256=ordinary['factor_stream_sha256'],
        ordered_events_and_decision_times_identical=True, shared_source_files=len(left['source_sha256']),
        comparison_sources={p: sha_file(ROOT / p) for p in (
            'tools/event_track_v2x/audit_beam_teacher_equivalence.py',
            'tools/event_track_v2x/assemble_allocation_teachers.py',
            'tools/event_track_v2x/audit_train_inference_comparison.py')},
        GT_read=False, parameter_training_performed=False, real_tracking_evaluation=False,
        all_unchosen_states_verified=False, universal_probe_purity_proved=False,
        equal_resource_benchmark=False, full_official_train_trace_completed=False, paper_eligible=False)
    _new_json(output, result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--teacher', nargs=2, metavar=('DIRECTORY', 'RECEIPT_SHA256'), required=True)
    parser.add_argument('--inference', nargs=2, metavar=('DIRECTORY', 'FINAL_RECEIPT_SHA256'), required=True)
    parser.add_argument('--cooperative-metadata', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    import json
    print(json.dumps(compare(args.teacher, args.inference, args.cooperative_metadata, args.output), sort_keys=True))
