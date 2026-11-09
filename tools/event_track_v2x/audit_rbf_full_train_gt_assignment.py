"""Complete accepted train-output geometric diagnostic, not official metrics."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import sqlite3
import zipfile
from audit_rbf_positive_recovery_gt import DOC, BASE, ARCHIVE, sha, transform
from audit_rbf_recovery_xy_flow import solve


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed', type=int, choices=(1337,3407), default=1337)
    args = parser.parse_args()
    seed = args.seed
    base = BASE if seed == 1337 else Path('/Volumes/Data/test/recover-before-fuse/artifacts/rbf-final-refit-full-train-forest-independent-byte-factor-v1-20261004/rbf/seed3407')
    index_path = DOC/f'SPD-seed{seed}-real-recovery-candidate-index-current.json'
    index = json.loads(index_path.read_text())
    assert len(index['sequence_reports']) == 46
    assert sha(ARCHIVE)=='10bbf7130114e693cda8c99c6f3e0f88949b2224be9725a268cd5ee3825c97c1'
    results = []
    with zipfile.ZipFile(ARCHIVE) as archive:
        for report in index['sequence_reports']:
            sid = report['sequence_id']
            assert sha(Path(report['path'])) == report['sha256']
            paths = list(base.glob(f'rank*-unpack/rank-*/{sid}/sequence-*.sqlite'))
            assert len(paths)==1 and sha(paths[0])==report['database_sha256']
            conn = sqlite3.connect(paths[0].as_uri()+'?mode=ro&immutable=1',uri=True)
            frames = []; previous = None
            for ordinal, raw in conn.execute('select ordinal,prediction from events order by ordinal'):
                p = json.loads(raw)
                canonical = json.dumps({k:v for k,v in p.items() if k!='commit_sha256'},sort_keys=True,separators=(',', ':'),allow_nan=False).encode()
                assert hashlib.sha256(canonical).hexdigest()==p['commit_sha256']
                if previous is not None:
                    assert p['previous_commit_sha256']==previous
                previous = p['commit_sha256']; fid = p['frame_id']; members = {}; pins = {}
                for key,path in [('GT',f'cooperative/label/{fid}.json'),('lidar',f'vehicle-side/calib/lidar_to_novatel/{fid}.json'),('world',f'vehicle-side/calib/novatel_to_world/{fid}.json')]:
                    raw = archive.read('V2X-Seq-SPD/'+path); members[key]=json.loads(raw); pins[key]=hashlib.sha256(raw).hexdigest()
                world=members['world'];lidar=members['lidar'].get('transform',members['lidar']);ego=[float(v[0]) for v in world['translation']]
                assert len({r['track_id'] for r in members['GT']})==len(members['GT'])
                gt=[]
                for r in members['GT']:
                    assert int(r['veh_pointcloud_timestamp'])==int(p['box_reference_timestamp_us'])
                    if r['type'] not in ('Car','Truck','Van','Bus'):
                        continue
                    xyz=transform(world,transform(lidar,[float(r['3d_location'][k]) for k in ('x','y','z')]))
                    if math.hypot(xyz[0]-ego[0],xyz[1]-ego[1])<50:
                        gt.append((r['track_id'],xyz))
                pred=[r for r in p['predictions'] if r['class_label']=='car' and math.hypot(r['mean'][0]-ego[0],r['mean'][1]-ego[1])<50]
                assert len({r['track_id'] for r in pred})==len(pred)
                matrix=[[math.hypot(r['mean'][0]-xyz[0],r['mean'][1]-xyz[1]) for _,xyz in gt] for r in pred]
                cost,pairs=solve(matrix);pi={i for i,_,_ in pairs};gi={j for _,j,_ in pairs}
                frames.append({'event':ordinal,'frame':fid,'prediction_commit':p['commit_sha256'],'members_sha256':pins,'ROI_prediction_count':len(pred),'ROI_GT_count':len(gt),'distance_sum_m':cost,
                               'matches':[{'prediction_id':pred[i]['track_id'],'GT_id':gt[j][0],'distance_m':d} for i,j,d in pairs],
                               'unmatched_predictions':[r['track_id'] for i,r in enumerate(pred) if i not in pi], 'unmatched_GT':[tid for j,(tid,_) in enumerate(gt) if j not in gi]})
            conn.close()
            assert len(frames)>0 and [r['event'] for r in frames]==list(range(len(frames)))
            results.append({'sequence':sid,'database_sha256':report['database_sha256'],'events':frames})
            print('completed',sid,len(frames),flush=True)
    assert sum(len(r['events']) for r in results)==7445
    out={'index_sha256':sha(index_path),'reader_sha256':sha(Path(__file__)),'sequences':46,'events':7445,'scope':f'complete accepted seed{seed} train stream; strict car 50m ROI and XY<2m diagnostic assignment','official_native_metrics_computed':False,'recovery_off_comparison_performed':False,'formal_recovery_advantage_proven':False,'results':results}
    (DOC/f'SPD-seed{seed}-full-train-GT-assignment-diagnostic-current.json').write_text(json.dumps(out,indent=2)+'\n')
    print('complete',46,7445,flush=True)


if __name__=='__main__':
    main()
