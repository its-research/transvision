import hashlib
import json
import pytest
from tools.event_track_v2x.prepare_v2v4real_identity_labels import inputs


def spec(path,value):
    path.write_text(json.dumps(value))
    return dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def test_official_test_partition_rejected_before_projection_read(tmp_path):
    part=spec(tmp_path/'partition',dict(official_split='official_test',official_test_used_for_partition=True,sequence_to_name_group={}))
    with pytest.raises(ValueError,match='before labels'):
        inputs(dict(partition=part,projection=dict(path=str(tmp_path/'missing'),sha256='0'*64)))


def test_partial_train_projection_rejected_before_volume_read(tmp_path):
    part=spec(tmp_path/'partition',dict(official_split='train',official_test_used_for_partition=False,
        sequence_to_name_group={str(i):str(i) for i in range(32)}))
    proj=spec(tmp_path/'frames',[])
    # JSONL row with valid identity, but incomplete cohort.
    p=tmp_path/'frames';p.write_text(json.dumps(dict(sequence_id='0',frame_key='0',cav_id='0',frame_ordinal=0))+'\n')
    proj['sha256']=hashlib.sha256(p.read_bytes()).hexdigest()
    with pytest.raises(ValueError,match='full paired'):
        inputs(dict(partition=part,projection=proj,volumes=[dict(path=str(tmp_path/'missing'),sha256='0'*64)]))
