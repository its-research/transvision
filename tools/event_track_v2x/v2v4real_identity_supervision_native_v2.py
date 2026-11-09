"""Offline native identity adapter; car network slot denotes merged vehicle.

Temporal links use explicit per-source object IDs. Cross-source links require
equal author-native mapped IDs in one paired frame, with at least one
explicit association. Coincident two-fallback aliases do not create links. Conflicting associations, native aliases
and merged components with duplicates are unknown supervision.
"""
from collections import Counter,defaultdict
import math
import numpy as np
from transvision.models.event_track_v2x.forest_supervision import AnnotationIdentity,AnnotationIdentityIndex,DetectionIdentityTarget
from transvision.models.event_track_v2x.forest_training_data import FrameSupervision


def source_supervision(rows):
    frames={};annotations=[];pairs=defaultdict(dict);association_history=defaultdict(set);bad_tracks=set()
    for r in rows:
        side=r['side'];source=0 if side=='vehicle-side' else 1
        if side not in ('vehicle-side','infrastructure-side') or r['cav_id']!=str(source):
            raise ValueError('explicit original native CAV-to-source mapping required')
        k=r['sequence_id'],side,r['frame_key']
        if k in frames:raise ValueError('duplicate native source supervision frame')
        ids=r['object_ids'];aids=r['association_ids'];mapped=r['native_mapped_ids']
        boxes=np.asarray(r['states_source_lwh'],dtype=np.float64).reshape(-1,7)
        if len(set(ids))!=len(ids) or len(aids)!=len(ids) or len(mapped)!=len(ids) or len(boxes)!=len(ids):
            raise ValueError('aligned unique native object IDs required')
        # The cache retains legacy gravity-centre width/length/yaw convention.
        converted=boxes.copy();converted[:,3]=boxes[:,4];converted[:,4]=boxes[:,3]
        converted[:,6]=-boxes[:,6]-math.pi/2
        ann=tuple(AnnotationIdentity(r['sequence_id'],source,r['frame_key'],oid,
            r['frame_key']+':'+oid,'car') for oid in ids)
        annotations.extend(ann)
        frames[k]=FrameSupervision(r['sequence_id'],source,r['frame_key'],r['frame_ordinal']*100000,converted,ann)
        pairs[r['sequence_id'],r['frame_key']][source]=(ann,aids,mapped)
        for oid,aid,mid in zip(ids,aids,mapped):
            expected=int(aid) if int(aid)>=0 else int(oid)+100*source
            if int(mid)!=expected:raise ValueError('author native ID mapping differs')
        counts=Counter(mapped)
        for a,aid,mid in zip(ann,aids,mapped):
            if int(aid)<-1:raise ValueError('native association ID outside declared domain')
            if int(aid)>=0:association_history[a.track].add(aid)
            if counts[mid]>1:bad_tracks.add(a.track)
    bad_tracks.update(k for k,v in association_history.items() if len(v)>1)
    links=[]
    for pair in pairs.values():
        if set(pair)!={0,1}:raise ValueError('both original source label frames required')
        lookups=[]
        for source in (0,1):
            ann,aids,mapped=pair[source];count=Counter(mapped)
            lookups.append({int(mid):(a,int(aid)) for a,aid,mid in zip(ann,aids,mapped) if count[mid]==1})
        for mid in sorted(lookups[0].keys()&lookups[1].keys()):
            (a,aid),(b,bid)=lookups[0][mid],lookups[1][mid]
            if aid>=0 or bid>=0:links.append((a.reference,b.reference))
    index=AnnotationIdentityIndex(annotations,links,class_scope=('car','bicycle','pedestrian'))
    bad_components={index.targets[a.reference].identity for a in annotations if a.track in bad_tracks}
    # Propagate ambiguity across an already joined component; do not allow an
    # apparently good partner to turn a conflict into a positive target.
    for a in annotations:
        old=index.targets[a.reference]
        if a.track in bad_tracks or old.identity is not None and old.identity in bad_components:
            index.targets[a.reference]=DetectionIdentityTarget(a.sequence_id,a.source_id,None,old.cross_linked,'ambiguous')
    index.audit.update(native_vehicle_class_scope=['vehicle'],network_class_slot_mapping={'car':'vehicle'},
        temporal_label_basis='native_source_object_id',cross_source_label_basis='author_native_mapped_id_equality_at_paired_frame_at_least_one_explicit_association_v2',
        native_alias_or_association_conflict_tracks=len(bad_tracks),exact_native_cross_source_links=len(links),
        physical_identity_continuity_verified=False,status_counts=dict(Counter(t.status for t in index.targets.values())))
    schedule=[]
    for (scene,key),pair in sorted(pairs.items()):
        left,right=frames[scene,'vehicle-side',key],frames[scene,'infrastructure-side',key]
        if left.timestamp_us!=right.timestamp_us:raise ValueError('paired native nominal clock differs')
        schedule.append(dict(sequence_id=scene,vehicle_frame=key,infrastructure_frame=key,box_reference_timestamp_us=left.timestamp_us))
    return frames,index,schedule
