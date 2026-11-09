#!/usr/bin/env python3
"""Independently stream every registered SPD val tar member against manifest."""
import argparse,datetime,hashlib,json,tarfile
from pathlib import Path

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda:f.read(8*1024**2),b''):h.update(block)
    return h.hexdigest()
def main():
    p=argparse.ArgumentParser();p.add_argument('--publication',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();v=json.loads(a.publication.read_bytes());assert v['kind']=='rbf_seen_val_complete_forest_inputs_cloud_bytes_v1' and v['independent_cloud_bytes_verified'];folder=a.publication.parent/'independent-cloud-bytes'
    for role,spec in v['artifacts'].items():
        path=folder/spec['key'];assert path.is_file() and not path.is_symlink() and sha(path)==spec['sha256'] and path.stat().st_size==spec['bytes']
    manifest_path=folder/v['artifacts']['cache_manifest']['key'];manifest=json.loads(manifest_path.read_bytes());assert manifest['split']=='val' and manifest['frame_count']==len(manifest['frames'])==7189 and not manifest['gt_in_cache'];expected={'cache/manifest.json':v['artifacts']['cache_manifest']}
    for frame in manifest['frames']:
        for field in ['arrays','metadata']:
            r=frame[field];name='cache/'+r['path'];assert name not in expected and not Path(r['path']).is_absolute() and '..' not in Path(r['path']).parts;expected[name]=r
    assert len(expected)==14379;seen=set();payload_bytes=0
    with tarfile.open(folder/v['artifacts']['cache_archive']['key'],'r|') as tf:
        for member in tf:
            assert member.isfile() and not member.issym() and not member.islnk() and member.name in expected and member.name not in seen
            r=expected[member.name];assert member.size==r['bytes'];h=hashlib.sha256();count=0
            with tf.extractfile(member) as stream:
                for block in iter(lambda:stream.read(8*1024**2),b''):h.update(block);count+=len(block)
            assert count==r['bytes'] and h.hexdigest()==r['sha256'];seen.add(member.name);payload_bytes+=count
    assert seen==set(expected)
    events=json.loads((folder/v['artifacts']['events']['key']).read_bytes());assert len(events['events'])==3316 and len(events['origin_us_by_sequence'])==21
    transport=json.loads((folder/v['artifacts']['transport']['key']).read_bytes());assert transport['members']==len(seen) and transport['payload_bytes']==payload_bytes
    from clearml import Task
    task=Task.get_task(task_id=v['task_id']);assert str(task.status)=='completed' and set(task.artifacts)=={r['key'] for r in v['artifacts'].values()}
    for spec in v['artifacts'].values():assert task.artifacts[spec['key']].hash==spec['sha256'] and task.artifacts[spec['key']].size==spec['bytes']
    result=dict(kind='rbf_seen_val_all_registered_cache_members_independent_readback',seed=v['seed'],task_id=task.id,task_status=str(task.status),publication_sha256=sha(a.publication),source_sha256=sha(__file__),registered_artifacts=7,member_count=len(seen),payload_bytes=payload_bytes,source_frames=7189,sequences=21,events=3316,all_registered_members_independently_streamed_and_hashed=True,cache_or_neural_forward_rebuilt=False,real_measured_arrival_history=False,seen_SPD_val_exploratory=True,full_val_replay_complete=False,same_resource_complete=False,paper_complete=False,checked_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
    a.output.open('x').write(json.dumps(result,sort_keys=True,indent=2)+'\n');print(json.dumps({'seed':v['seed'],'member_count':len(seen),'all_registered_member_readback_passed':True}))
if __name__=='__main__':main()
