#!/usr/bin/env python3
"""Compare one-step vs full-class teacher labels on one sealed state copy.

All probes and the artificial no-new-input event are rolled back. This is not
tracking validation, a complete training trace, or a claimed method gain.
"""
from __future__ import annotations

import argparse
from contextlib import closing
import json
from pathlib import Path
import shutil
import sqlite3
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))

from transvision.models.event_track_v2x.allocation_training import allocation_sources, _directory
from transvision.models.event_track_v2x.detection_cache_v2 import sha_file
from transvision.models.event_track_v2x.forest_training_data import _new_json
from transvision.models.event_track_v2x.frontier_completion import frontier_completion_operation
from transvision.models.event_track_v2x.covered_proposal_capacity import (
    covered_proposal_operation, proposal_frontier_reserve,
)
from transvision.models.event_track_v2x.learned_component_allocation import AllocationTeacherTracker


class _ProbeFinished(Exception):
    pass


class _ComparisonTeacher(AllocationTeacherTracker):
    def _execute_allocation_work(self,kernel,operation,decision_us,prefix_count,frontier_count):
        before=len(kernel.frontier)
        result=super()._execute_allocation_work(kernel,operation,decision_us,prefix_count,frontier_count)
        after=len(kernel.frontier)
        if (after>kernel.config.state.max_frontier
                or frontier_count+after-before>self.config.max_total_frontier):
            raise ValueError('probed operation exceeded actual frontier capacity')
        self._executed_capacity=dict(frontier_before=before,frontier_after=after,
            global_frontier_after=frontier_count+after-before,
            local_frontier_cap=kernel.config.state.max_frontier,global_frontier_cap=self.config.max_total_frontier)
        return result

    def _probe_uncached(self,record,state):
        result=super()._probe_uncached(record,state)
        return dict(result,**self._executed_capacity)

    def _component_inference(self,*args,**kwargs):
        super()._component_inference(*args,**kwargs)
        # A state with no eligible allocation point must also roll back rather
        # than commit the artificial diagnostic event before reporting failure.
        raise ValueError('no eligible allocation point; comparison not performed')

    def _select_allocation(self,candidates,**state):
        self.comparison=[]
        for record in self._priority_candidates(candidates,state):
            c=record['component']; k=self.kernels[c]
            before=tuple(self.db.iterdump()) if self.check_exact_dump else None
            ordinary=self._probe(record,state)
            operation=frontier_completion_operation(k,remaining=state['remaining'],
                prefix_count=state['prefix_count'],frontier_count=state['frontier_count'],config=self.config)
            alternative=None
            if operation is not None:
                alternative=self._probe(dict(record,operation=operation),state)
            coverage_operation=coverage_result=None
            if self.check_coverage_admission:
                coverage_operation=covered_proposal_operation(k,remaining=state['remaining'],
                    prefix_count=state['prefix_count'],frontier_count=state['frontier_count'],config=self.config)
                if coverage_operation is not None:
                    coverage_result=self._probe(dict(record,operation=coverage_operation),state)
            if before is not None and tuple(self.db.iterdump()) != before:
                raise ValueError('counterfactual changed the persistent state')
            self.comparison.append(dict(component=c,nodes=k.n,frontier=len(k.frontier),
                loss_weight=state['weights'][c],model_bound_before=record['model_bound_before'],
                ordinary_operation=record['operation'],ordinary=ordinary,
                completion_operation=operation,completion=alternative,
                coverage_frontier_reserve=proposal_frontier_reserve(k) if self.check_coverage_admission else None,
                coverage_operation=coverage_operation,coverage_result=coverage_result))
        raise _ProbeFinished()


def compare(database,database_sha256,prediction_sha256,output,*,check_exact_dump=False,check_coverage_admission=False):
    database=Path(database).absolute()
    if not database.is_file() or database.is_symlink() or sha_file(database)!=database_sha256:
        raise ValueError('sealed source database identity required')
    if any(Path(str(database)+s).exists() for s in ('-wal','-shm','-journal')):
        raise ValueError('closed source database required')
    with closing(sqlite3.connect(database.as_uri()+'?mode=ro',uri=True)) as connection:
        meta={k:json.loads(v) for k,v in connection.execute('SELECT k,v FROM meta')}
    if (meta['schema']!=AllocationTeacherTracker.SCHEMA or not meta['state']['events']
            or meta['state']['prediction_sha256']!=prediction_sha256):
        raise ValueError('committed teacher state and expected output head required')
    sources=allocation_sources()
    for name in ('tools/event_track_v2x/probe_frontier_completion.py',
                 'transvision/models/event_track_v2x/frontier_completion.py',
                 'transvision/models/event_track_v2x/covered_proposal_capacity.py'):
        sources[name]=sha_file(ROOT/name)
    output=_directory(output); copied=output/'state.sqlite'
    shutil.copyfile(database,copied)
    tracker=_ComparisonTeacher.open(copied,expected_prediction_sha256=prediction_sha256,
                                     expected_database_sha256=database_sha256)
    tracker.check_exact_dump=check_exact_dump
    tracker.check_coverage_admission=check_coverage_admission
    started=time.monotonic()
    try:
        binding=meta.get('cache_binding')
        cache_ingestion=None if binding is None else dict(kind='persistent_cache_ingestion_v1',
            configuration_sha256=binding,new_deliveries=[])
        try:
            tracker.step((),(),frame_id='completion-probe',event_id='completion-probe',
                reference_us=meta['state']['reference_us']+1,decision_us=meta['state']['decision_us']+1,
                cache_ingestion=cache_ingestion,scorer_binding=meta.get('scorer_binding'))
        except _ProbeFinished:
            pass
        else:
            raise ValueError('no eligible allocation point; not a successful comparison')
        if tracker.meta!=meta['state'] or tracker.db.execute('SELECT count(*) FROM events').fetchone()[0]!=meta['state']['events']:
            raise ValueError('probe event failed to roll back')
        if sha_file(database)!=database_sha256 or any(sha_file(ROOT/p)!=h for p,h in sources.items()):
            raise ValueError('original state or source changed during probing')
        result=dict(kind='frontier_completion_teacher_comparison_v1',status='complete',
            source_database_sha256=database_sha256,source_prediction_sha256=prediction_sha256,
            source_sha256=sources,observations=tracker.n,committed_events=tracker.meta['events'],
            coverage_admission_compared=check_coverage_admission,
            new_observations=0,elapsed_seconds=time.monotonic()-started,candidates=tracker.comparison,
            future_or_gt_inputs=False,new_event_committed=False,tracking_validation=False,paper_eligible=False)
        _new_json(output/'comparison.json',result)
        print(json.dumps(result,sort_keys=True),flush=True)
        return result
    except BaseException as error:
        _new_json(output/'failure.json',dict(error_type=type(error).__name__,error=str(error)))
        raise
    finally:
        tracker.close()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--database',type=Path,required=True)
    p.add_argument('--database-sha256',required=True)
    p.add_argument('--prediction-sha256',required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--coverage-aware-admission',action='store_true')
    a=p.parse_args()
    compare(a.database,a.database_sha256,a.prediction_sha256,a.output,
            check_coverage_admission=a.coverage_aware_admission)
