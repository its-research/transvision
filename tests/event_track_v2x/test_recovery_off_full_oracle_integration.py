"""Actual frozen cache ingestion checked by five independent small-data oracles.

This fixture substitutes only the dataset admission envelope. It does not
certify a real checkpoint, cohort, runtime, or experiment performance.
"""
import collections
import json
from pathlib import Path
import sqlite3
import sys

import numpy as np
import pytest
import torch

FROZEN = Path('/Volumes/Data/test/recover-before-fuse/source-freezes/rbf-recovery-off-original-source-bound-candidate-v3-20261005')
sys.path.insert(0, str(FROZEN))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]/'tools/event_track_v2x'))
from transvision.models.event_track_v2x import recovery_off_paper_runtime as producer
from transvision.models.event_track_v2x.forest_potentials import LearnedForestScorer
from transvision.models.event_track_v2x.learned_identity import RecoverableIdentityModel
from transvision.models.event_track_v2x.paper_native_cache import NativePaperCache, write_native_cache
from transvision.models.event_track_v2x.paper_protocol import PaperProtocol
from test_paper_pipeline import native_frame
from recovery_off_final_binding import load_unchanged_oracles
from recovery_off_structure_oracle import verify_database as structure_verify
from recovery_off_action_oracle import verify_database as action_verify


def frozen_fixture(tmp_path):
    assert Path(producer.__file__).resolve().is_relative_to(FROZEN)
    frames = [native_frame(sequence_id='0003', side=side, agent_mask=i+1, dataset_split='train')
              for i, side in enumerate(('vehicle-side', 'infrastructure-side'))]
    root = tmp_path/'cache'
    digest = write_native_cache(root, frames, split='train', producer={'fit_split':'train'}, fixture=True)
    cache = NativePaperCache(root, digest)
    records = {key: dict(manifest_entry=json.loads(entry), metadata=json.loads(meta))
               for key,(entry,meta) in cache.index.items()}
    deliveries=[]
    for (sequence,side,frame), record in records.items():
        meta=record['metadata']; entry=record['manifest_entry']
        deliveries.append(dict(sequence_id=sequence, side=side, frame_id=frame,
            arrival_us=max(meta['box_reference_timestamp_us'],meta['source_image_timestamp_us'])+100000,
            frame_sha256=entry['frame_sha256']))
    decision=max(d['arrival_us'] for d in deliveries)
    events=[dict(sequence_id='0003',frame_id='first',event_id='first',reference_us=decision,
                 decision_us=decision,deliveries=deliveries),
            dict(sequence_id='0003',frame_id='duplicate',event_id='duplicate',reference_us=decision+100000,
                 decision_us=decision+100000,deliveries=deliveries),
            dict(sequence_id='0003',frame_id='expired',event_id='expired',reference_us=decision+10000000,
                 decision_us=decision+10000000,deliveries=[])]
    torch.manual_seed(1337)
    model=RecoverableIdentityModel(hidden=8,heads=2,dropout=0.).eval().requires_grad_(False)
    scorer=LearnedForestScorer(model,max_nodes=9,max_pairs=81)
    out=tmp_path/'replay'
    receipt=producer.replay(cache,events,out,protocol=PaperProtocol('v2v4real','train'),
        configuration=producer.default_configuration(),scorer=scorer,
        model_binding=dict(candidate_protocol='rbf-all-class-top64-v1',dataset='v2v4real',fit_split='train'),fixture=True)
    _, oracle, causal, fresh=load_unchanged_oracles()
    # The unchanged frame loader reads and hashes actual raw cache members;
    # only real-cohort provenance checks are replaced by a declared tiny fixture.
    admission=object.__new__(oracle.CacheAdmission)
    admission.root=root; admission.frames=records; admission.memo=collections.OrderedDict()
    admission.origins={'0003':min(x['metadata']['box_reference_timestamp_us'] for x in records.values())}
    admission.events={'0003':events}
    admission.checkpoint=dict(model_sha256=scorer.model_sha256,geometry_weight=scorer.options['geometry_weight'])
    count=sum(len(admission.frame(d)) for d in deliveries)
    admission.proof=dict(manifest_sha256=digest,per_sequence_rows={'0003':count})
    admission.binding=dict(rows=count,fixture=True)
    entry=receipt['databases']['0003']
    return out/entry['path'],entry['sha256'],admission,oracle,causal,fresh


def test_original_raw_cache_causal_fresh_and_restricted_oracles_integrate(tmp_path):
    path,digest,admission,cache,causal,fresh=frozen_fixture(tmp_path)
    checks=[cache.verify_database(path,digest,admission),causal.verify_database(path,digest),
            fresh.verify_database(path,digest),structure_verify(path,digest),action_verify(path,digest)]
    assert all(v['events']==3 for v in checks)
    assert checks[0]['rows']>0 and checks[0]['duplicate_deliveries']==2
    assert checks[0]['complete_original_sequence'] is True  # tiny fixture schedule only
    assert checks[0]['all_203_features_and_world_state_values_bound_to_raw_cache'] is True
    assert checks[2]['stored_states_and_chosen_outputs_checked'] is True
    assert checks[3]['independent_restricted_support_coverage_verified'] is True
    assert checks[4]['restricted_action_search_or_capacity_fallback_verified'] is True
    assert all(v['paper_performance_complete'] is False for v in (checks[0],checks[4]))


@pytest.mark.parametrize('field', ['features','mean','score'])
def test_rehashed_numeric_mutation_rejected_by_original_cache_checker(tmp_path,field):
    path,_,admission,cache,_,_=frozen_fixture(tmp_path)
    with sqlite3.connect(path) as db:
        i,raw=db.execute('SELECT i,raw FROM observations ORDER BY i LIMIT 1').fetchone()
        value=json.loads(raw)
        if field=='score':
            value[field] *= .5
            db.execute('UPDATE observations SET score=? WHERE i=?',(value[field],i))
        else: value[field][0] += .02
        db.execute('UPDATE observations SET raw=?,sha=? WHERE i=?',(cache.canonical(value),cache.digest(value),i))
    with pytest.raises(AssertionError): cache.verify_database(path,cache.sha(path),admission)


def test_raw_cache_member_tampering_rejected_before_schema_acceptance(tmp_path):
    path,digest,admission,cache,_,_=frozen_fixture(tmp_path)
    admission.memo.clear()
    entry=next(iter(admission.frames.values()))['manifest_entry']['arrays']
    with (admission.root/entry['path']).open('ab') as stream: stream.write(b'changed')
    with pytest.raises(AssertionError,match='consumed cache member byte binding'):
        cache.verify_database(path,digest,admission)
