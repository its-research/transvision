"""Complete sequence streams for seven diagnostic conflict tracks; no IDS metrics."""
import hashlib
import json
import math
from pathlib import Path
import sqlite3
import zipfile
from audit_rbf_positive_recovery_gt import DOC, BASE, ARCHIVE, sha, transform
from audit_rbf_recovery_xy_flow import solve


def main():
    source = DOC / 'SPD-recovery-sparse-temporal-GT-association-current.json'
    candidates = [r for r in json.loads(source.read_text())['records']
                  if r['multiple_GT_ids_in_sparse_diagnostic']]
    index = json.loads((DOC / 'SPD-seed1337-real-recovery-candidate-index-current.json').read_text())
    assert sha(ARCHIVE) == '10bbf7130114e693cda8c99c6f3e0f88949b2224be9725a268cd5ee3825c97c1'
    results = []
    with zipfile.ZipFile(ARCHIVE) as archive:
        for sid in sorted({r['sequence'] for r in candidates}):
            paths = list(BASE.glob(f'rank*-unpack/rank-*/{sid}/sequence-*.sqlite'))
            assert len(paths) == 1
            report = next(r for r in index['sequence_reports'] if r['sequence_id'] == sid)
            assert sha(paths[0]) == report['database_sha256']
            conn = sqlite3.connect(paths[0].as_uri() + '?mode=ro&immutable=1', uri=True)
            for case in [r for r in candidates if r['sequence'] == sid]:
                start, end = conn.execute('select min(ordinal),max(ordinal) from events').fetchone()
                timeline = []
                for ordinal, raw_prediction in conn.execute('select ordinal,prediction from events where ordinal between ? and ? order by ordinal', (start,end)):
                    p = json.loads(raw_prediction)
                    canonical = json.dumps({k:v for k,v in p.items() if k!='commit_sha256'},sort_keys=True,separators=(',', ':'),allow_nan=False).encode()
                    assert hashlib.sha256(canonical).hexdigest() == p['commit_sha256']
                    frame = p['frame_id']; members = {}; pins = {}
                    for key, path in [('GT',f'cooperative/label/{frame}.json'),('lidar',f'vehicle-side/calib/lidar_to_novatel/{frame}.json'),('world',f'vehicle-side/calib/novatel_to_world/{frame}.json')]:
                        raw = archive.read('V2X-Seq-SPD/'+path)
                        members[key] = json.loads(raw); pins[key] = hashlib.sha256(raw).hexdigest()
                    world = members['world']; lidar = members['lidar'].get('transform',members['lidar']); ego = [float(v[0]) for v in world['translation']]
                    gt = []
                    for row in members['GT']:
                        assert int(row['veh_pointcloud_timestamp']) == int(p['box_reference_timestamp_us'])
                        if row['type'] not in ('Car','Truck','Van','Bus'):
                            continue
                        xyz = transform(world,transform(lidar,[float(row['3d_location'][k]) for k in ('x','y','z')]))
                        if math.hypot(xyz[0]-ego[0],xyz[1]-ego[1]) < 50:
                            gt.append((row['track_id'],xyz))
                    pred = [r for r in p['predictions'] if r['class_label']=='car' and math.hypot(r['mean'][0]-ego[0],r['mean'][1]-ego[1])<50]
                    matrix = [[math.hypot(r['mean'][0]-xyz[0],r['mean'][1]-xyz[1]) for _,xyz in gt] for r in pred]
                    _, pairs = solve(matrix)
                    target = next((i for i,r in enumerate(pred) if r['track_id']==case['track_id']),None)
                    match = next(({'GT_id':gt[j][0],'distance_m':d} for i,j,d in pairs if i==target),None)
                    timeline.append({'event':ordinal,'frame':frame,'prediction_commit':p['commit_sha256'],'members_sha256':pins,'target_in_car_ROI':target is not None,'assignment':match})
                assert len(timeline)==end-start+1
                results.append({'sequence':sid,'track_id':case['track_id'],'first_event':start,'last_event':end,'timeline':timeline})
            conn.close()
            print('sequence complete',sid,flush=True)
    assert len(results)==7
    out={'source_sha256':sha(source),'reader_sha256':sha(Path(__file__)),'tracks_checked':7,'continuous_interval_events':sum(len(r['timeline']) for r in results),'interval_coverage_complete':True,'full_sequence_prediction_stream_checked':True, 'official_metric_evaluation_complete':False,'IDS_metric_computed':False,'physical_identity_correctness_proven':False,'recovery_off_comparison_performed':False,'results':results}
    (DOC/'SPD-recovery-seven-conflict-full-sequences-current.json').write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps({k:v for k,v in out.items() if k!='results'}))


if __name__=='__main__':
    main()
