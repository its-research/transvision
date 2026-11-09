"""Check full diagnostic cohort and agreement with prior positive-case audit."""
import argparse
import hashlib
import json
from pathlib import Path

DOC = Path(__file__).resolve().parents[2]/'docs/recover-before-fuse'


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed',type=int,choices=(1337,3407),default=1337)
    seed=parser.parse_args().seed
    path = DOC/f'SPD-seed{seed}-full-train-GT-assignment-diagnostic-current.json'
    raw = path.read_bytes(); data = json.loads(raw)
    index = json.loads((DOC/f'SPD-seed{seed}-real-recovery-candidate-index-current.json').read_text())
    assert len(data['results'])==46
    assert {r['sequence'] for r in data['results']}=={r['sequence_id'] for r in index['sequence_reports']}
    lookup={}; matched=0; unmatched_pred=0; unmatched_gt=0
    for sequence in data['results']:
        events=sequence['events']
        assert [r['event'] for r in events]==list(range(len(events)))
        assert len({r['frame'] for r in events})==len(events)
        for frame in events:
            key=(sequence['sequence'],frame['event']);assert key not in lookup;lookup[key]=frame
            pairs=frame['matches'];pred={p['prediction_id'] for p in pairs};gt={p['GT_id'] for p in pairs}
            assert len(pred)==len(gt)==len(pairs)
            assert not pred.intersection(frame['unmatched_predictions'])
            assert not gt.intersection(frame['unmatched_GT'])
            assert len(pairs)+len(frame['unmatched_predictions'])==frame['ROI_prediction_count']
            assert len(pairs)+len(frame['unmatched_GT'])==frame['ROI_GT_count']
            assert all(0<=p['distance_m']<2 for p in pairs)
            matched+=len(pairs);unmatched_pred+=len(frame['unmatched_predictions']);unmatched_gt+=len(frame['unmatched_GT'])
    assert len(lookup)==7445
    prior_comparison_count=0
    if seed == 1337:
        positive=json.loads((DOC/'SPD-seed1337-all-positive-recovery-GT-assignment-current.json').read_text())
        for case in positive['results']:
            frame=lookup[(case['sequence'],case['event'])]
            assert frame['prediction_commit']==case['prediction_commit']
            assert frame['matches']==case['matched_pairs']
            assert frame['unmatched_predictions']==case['unmatched_predictions']
            assert frame['unmatched_GT']==case['unmatched_GT']
        prior_comparison_count=len(positive['results'])
    else:
        prefix=json.loads((DOC/'recovery-off-3407-failed-output-readback-20261011/sealed-prefix-GT-geometric-comparison.json').read_text())
        for sequence in prefix['results']:
            for case in sequence['events']:
                frame=lookup[(sequence['sequence'],case['event'])]
                assert frame['frame']==case['frame'] and frame['members_sha256']==case['members_sha256']
                assert frame['matches']==case['main']['matches']
                assert len(frame['unmatched_predictions'])==case['main']['unmatched_predictions']
                assert len(frame['unmatched_GT'])==case['main']['unmatched_GT']
                prior_comparison_count+=1
        assert prior_comparison_count==392
    out={'diagnostic_sha256':hashlib.sha256(raw).hexdigest(),'complete_sequences':46,'complete_events':7445,
         'prior_comparison_count':prior_comparison_count,'all_prior_case_assignments_equal':True,'all_246_positive_case_assignments_equal':True if seed==1337 else None,'unique_full_frame_matching_verified':True,
         'matched_prediction_rows':matched,'unmatched_prediction_rows':unmatched_pred,'unmatched_GT_rows':unmatched_gt,
         'scope':'train geometric diagnostic counts, not official FP/FN/AMOTA/HOTA metrics',
         'formal_metrics_accepted':False,'recovery_advantage_proven':False}
    (DOC/f'SPD-seed{seed}-full-train-GT-assignment-reconciliation-current.json').write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps(out))


if __name__=='__main__':
    main()
