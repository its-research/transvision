"""Read-only full-cohort raw-calibration XY diagnostic; no paper metrics."""
import hashlib
import json
import math
from pathlib import Path
import sqlite3
import zipfile
from audit_rbf_recovery_xy_flow import solve

ROOT = Path(__file__).resolve().parents[2]
DOC = ROOT / 'docs/recover-before-fuse'
BASE = Path('/Volumes/Data/test/recover-before-fuse/artifacts/rbf-final-refit-full-train-forest-independent-byte-factor-v1-20261004/rbf/seed1337')
ARCHIVE = Path('/Volumes/Data/test/recover-before-fuse/artifacts/spd-mht-k4-gt-source-recovery-20260929/V2X-Seq-SPD.zip')


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1048576), b''):
            h.update(block)
    return h.hexdigest()


def transform(payload, xyz):
    return [sum(float(payload['rotation'][i][j]) * xyz[j] for j in range(3))
            + float(payload['translation'][i][0]) for i in range(3)]


def main():
    binding = DOC / 'SPD-seed1337-all-positive-recovery-GT-member-binding-current.json'
    data = json.loads(binding.read_text())
    index = json.loads((DOC / 'SPD-seed1337-real-recovery-candidate-index-current.json').read_text())
    assert sha(ARCHIVE) == '10bbf7130114e693cda8c99c6f3e0f88949b2224be9725a268cd5ee3825c97c1'
    connections = {}
    results = []
    with zipfile.ZipFile(ARCHIVE) as archive:
        for case in data['records']:
            sid = case['sequence']
            if sid not in connections:
                paths = list(BASE.glob(f'rank*-unpack/rank-*/{sid}/sequence-*.sqlite'))
                assert len(paths) == 1
                expected = next(r for r in index['sequence_reports'] if r['sequence_id'] == sid)
                assert sha(paths[0]) == expected['database_sha256']
                connections[sid] = sqlite3.connect(paths[0].as_uri() + '?mode=ro&immutable=1', uri=True)
            p = json.loads(connections[sid].execute('select prediction from events where ordinal=?', (case['event'],)).fetchone()[0])
            canonical = json.dumps({k:v for k,v in p.items() if k != 'commit_sha256'}, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
            assert hashlib.sha256(canonical).hexdigest() == p['commit_sha256'] == case['prediction_commit']
            members = {}
            for key, meta in case['members'].items():
                raw = archive.read('V2X-Seq-SPD/' + meta['path'])
                assert hashlib.sha256(raw).hexdigest() == meta['sha256']
                members[key] = json.loads(raw)
            world = members['world']
            lidar = members['lidar'].get('transform', members['lidar'])
            ego = [float(v[0]) for v in world['translation']]
            gt = []
            for row in members['GT']:
                assert int(row['veh_pointcloud_timestamp']) == int(p['box_reference_timestamp_us'])
                if row['type'].lower() not in ('car', 'truck', 'van', 'bus'):
                    continue
                xyz = transform(world, transform(lidar, [float(row['3d_location'][k]) for k in ('x','y','z')]))
                if math.hypot(xyz[0]-ego[0], xyz[1]-ego[1]) < 50:
                    gt.append((row['track_id'], xyz))
            pred = [v for v in p['predictions'] if v['class_label']=='car' and math.hypot(v['mean'][0]-ego[0],v['mean'][1]-ego[1]) < 50]
            screen = json.loads((DOC / f'SPD-seed1337-sequence{sid}-recovery-ROI-screen-current.json').read_text())
            event = next(v for v in screen['all_results'] if v['event_ordinal']==case['event'] and v['component']==case['component'])
            related = {v['track_id'] for v in event['visible_outputs'] if v['car_inside_ROI']}
            matrix = [[math.hypot(v['mean'][0]-xyz[0],v['mean'][1]-xyz[1]) for _,xyz in gt] for v in pred]
            cost, pairs = solve(matrix)
            matched = {pred[i]['track_id']: {'GT_id':gt[j][0], 'distance_m':d} for i,j,d in pairs}
            results.append({**case, 'ROI_predictions':len(pred), 'ROI_GT':len(gt), 'matched_count':len(pairs), 'distance_sum_m':cost,
                            'related_outputs':[{'track_id':tid,'assignment':matched.get(tid)} for tid in sorted(related)],
                            'matched_pairs':[{'prediction_id':pred[i]['track_id'],'GT_id':gt[j][0],'distance_m':d} for i,j,d in pairs],
                            'unmatched_predictions':[v['track_id'] for v in pred if v['track_id'] not in matched],
                            'unmatched_GT':[tid for j,(tid,_) in enumerate(gt) if j not in {b for _,b,_ in pairs}]})
    for conn in connections.values():
        conn.close()
    assert len(results)==data['required_positive_cases']==246
    out = {'binding_sha256':sha(binding),'reader_sha256':sha(Path(__file__)), 'cases_checked':len(results),
           'related_outputs':sum(len(v['related_outputs']) for v in results),
           'related_outputs_matched':sum(r['assignment'] is not None for v in results for r in v['related_outputs']),
           'method':'maximum cardinality then minimum XY distance; strict 2m; car strict 50m XY ROI; raw calibration',
           'native_metric_engine_executed':False,'quaternion_pose_parity_verified':False,
           'recovery_off_comparison_performed':False,'recovery_advantage_proven':False,'results':results}
    (DOC/'SPD-seed1337-all-positive-recovery-GT-assignment-current.json').write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps({k:v for k,v in out.items() if k!='results'}))


if __name__ == '__main__':
    main()
