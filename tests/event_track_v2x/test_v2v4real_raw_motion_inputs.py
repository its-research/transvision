import hashlib
import io
import json
import tarfile
import numpy as np
import pytest
from tools.event_track_v2x.v2v4real_raw_motion_inputs import load_raw


def digest(b):return hashlib.sha256(b).hexdigest()


def fixture(root,*,extra=False,gt=False):
    members={};index=[]
    for ordinal in (0,1):
        for side in ('vehicle-side','infrastructure-side'):
            n=0 if ordinal==1 and side=='vehicle-side' else 1
            s=np.tile([1.,2.,3.,2.,4.,1.,0.],(n,1));b=io.BytesIO()
            np.savez_compressed(b,states_source_legacy7=s,raw_scores=np.ones(n),class_indices=np.zeros(n,dtype=np.int64),
                                appearance=np.ones((n,128)),appearance_valid=np.ones(n,dtype=bool))
            name=f'payloads/{len(index):05d}.npz';members[name]=b.getvalue()
            index.append(dict(sequence_id='s',frame_id=f'{ordinal:06d}',frame_ordinal=ordinal,side=side,
                source_to_world=np.eye(4).tolist(),pcd_sha256='a'*64,payload=name,payload_sha256=digest(b.getvalue()),candidates=n))
    index_bytes=''.join(json.dumps(x)+'\n' for x in reversed(index)).encode()
    b=io.BytesIO();np.save(b,np.zeros((1,7)));members['anchors.npy']=b.getvalue()
    m=dict(split='train',gt_read=gt,kind='v2v4real_train_raw_native_features_v1',official_test_read=False,
        checkpoint_seed=1337,checkpoint_sha256='b'*64,
        protocol_id='v2v4real-nominal-10hz-formal-v1',candidate_protocol='rbf-all-class-top64-v1',
        matching_candidate_rule='raw_score_ge_0.05_stable_top64_before_nms',rows_sha256=digest(index_bytes),
        anchor_file_sha256=digest(b.getvalue()),rows=4,total_candidates=3,sequence_ids=['s'])
    mb=json.dumps(m).encode();members.update({'manifest.json':mb,'predictions.jsonl':index_bytes})
    if extra:members['unknown.txt']=b'foreign'
    (root/'native-feature-manifest').write_bytes(mb);(root/'native-feature-index').write_bytes(index_bytes)
    with tarfile.open(root/'native-features','w:gz') as t:
        for name,b in members.items():
            info=tarfile.TarInfo(name);info.size=len(b);t.addfile(info,io.BytesIO(b))
    return digest(mb),digest((root/'native-features').read_bytes())


def test_complete_gt_free_raw_sorted_with_empty_frame(tmp_path):
    ms,ars=fixture(tmp_path);m,rows,frames=load_raw(tmp_path,ms,ars,{})
    assert len(rows)==4 and sum(len(f['states_source_legacy7']) for f in frames)==3
    assert [(r['side'],r['frame_ordinal']) for r in rows]==[
        ('infrastructure-side',0),('infrastructure-side',1),('vehicle-side',0),('vehicle-side',1)]
    assert frames[-1]['states_source_legacy7'].shape==(0,7)
    assert all(len(f['row_sha256'])==64 for f in frames)


def test_unexpected_archive_member_rejected(tmp_path):
    ms,ars=fixture(tmp_path,extra=True)
    with pytest.raises(ValueError,match='archive members'):load_raw(tmp_path,ms,ars,{})


def test_GT_raw_envelope_rejected_before_archive(tmp_path):
    ms,ars=fixture(tmp_path,gt=True);(tmp_path/'native-features').unlink()
    with pytest.raises(ValueError,match='GT-free'):load_raw(tmp_path,ms,ars,{})


def test_original_archive_hash_mismatch_rejected(tmp_path):
    ms,_=fixture(tmp_path)
    with pytest.raises(ValueError,match='archive SHA'):load_raw(tmp_path,ms,'0'*64,{})
