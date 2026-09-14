"""The verified node-beam backbone with separately bounded recoverable search.

Disabling recovery calls the original beam unchanged. Enabling it keeps an
exhaustive covering frontier of raw identity classes, seeds the SAME current
beam candidates, then spends an extra declared budget. No averaged-state
reconstruction, learned confidence bound, cost equivalence or gain is claimed.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
import math

from .covered_proposal_capacity import covered_proposal_operation
from .detection_cache_v2 import canonical
from .identity_forest import digest
from .persistent_beam_tracking import PersistentBeamConfig, PersistentBeamTracker
from .persistent_component_store import ComponentChange
from .recovery_task_scope import SCOPE_MODE, validate_scope, allocation_weights


@dataclass(frozen=True)
class BeamRecoveryConfig(PersistentBeamConfig):
    enable_recovery: bool = True
    recovery_budget: int = 256
    recovery_completions_per_component: int = 1
    recovery_allocation_scope: str = 'all_recent'

    def __post_init__(self):
        # Base validation deliberately sees only its own integer-valued fields.
        PersistentBeamConfig(**{k:getattr(self,k) for k in PersistentBeamConfig.__dataclass_fields__})
        if (type(self.enable_recovery) is not bool or type(self.recovery_budget) is not int
                or not 0<=self.recovery_budget<=100000
                or type(self.recovery_completions_per_component) is not int
                or not 0<=self.recovery_completions_per_component<=8
                or self.recovery_allocation_scope not in ('all_recent', SCOPE_MODE)):
            raise ValueError('explicit recovery switch and bounded integer search budgets required')


class BeamRecoveryTracker(PersistentBeamTracker):
    CONFIG_TYPE = BeamRecoveryConfig
    SCHEMA = 'persistent_node_beam_with_recoverable_search_v1'

    def step(self, observations, rows, *, frame_id, reference_us, decision_us, event_id,
             rescored_rows=(), decision_indices=None, cache_ingestion=None, scorer_binding=None):
        scope = None if cache_ingestion is None else cache_ingestion.get('recovery_task_scope')
        if self.config.recovery_allocation_scope == 'all_recent':
            if scope is not None:raise ValueError('all-recent recovery cannot consume a task pose')
        else:
            validate_scope(scope,self,cache_ingestion,reference_us,decision_us)
        # The complete scope is part of cache_ingestion in the parent's atomic
        # request digest. This temporary context never survives a failed event.
        self._event_recovery_scope = scope
        try:
            return super().step(observations,rows,frame_id=frame_id,reference_us=reference_us,
                decision_us=decision_us,event_id=event_id,rescored_rows=rescored_rows,
                decision_indices=decision_indices,cache_ingestion=cache_ingestion,scorer_binding=scorer_binding)
        finally:
            self._event_recovery_scope = None

    def _audit_properties(self):
        return dict(super()._audit_properties(),kind=self.SCHEMA,
            recovery_enabled=self.config.enable_recovery,
            archived_prefixes_used_for_recovery=self.config.enable_recovery,
            component_allocation_integrated=self.config.enable_recovery,
            recovery_backbone='unchanged_arrival_ordered_node_beam',
            recovery_disabled_is_original_node_beam=True,
            full_raw_cover_retained_for_recovery=self.config.enable_recovery,
            additional_recovery_compute_is_not_free=True)

    def _old_search_state(self):
        states={}
        for component in self.store.live():
            kernel=self._kernel(component)
            row=kernel.db.execute("SELECT v FROM meta WHERE k='recovery_frontier'").fetchone()
            if row is None:
                raise ValueError('existing recoverable beam is missing its full-support frontier')
            states[component]=dict(meta=dict(kernel.meta),active=set(kernel.active),
                                   frontier=set(json.loads(row[0])),n=kernel.n)
        return states

    def _restore_cover(self,kernel,previous,beam,decision_us):
        restarted=False
        if previous is None:
            kernel.active=set();kernel.frontier={0}
        else:
            kernel.active=set(previous['active']);kernel.frontier=set(previous['frontier'])
            if kernel.n>previous['n']:
                for h in kernel.active:
                    if not kernel._covered(h):kernel.frontier.add(h)
                kernel.active.clear()
        if len(kernel.frontier)>kernel.config.state.max_frontier:
            kernel.frontier={0};restarted=True
        kernel._promote(decision_us)
        for h in sorted(beam):
            # Coarsen a covering partition, never delete an identity region.
            if len(kernel.frontier)+1>kernel.config.state.max_frontier:
                kernel.frontier={0};restarted=True
            kernel.seed_complete_action(h,decision_us)
        return restarted

    def _recovery_operation(self,kernel,remaining,prefix_count,frontier_count,excluded,completion_count):
        if completion_count<self.config.recovery_completions_per_component:
            operation=covered_proposal_operation(kernel,remaining=remaining,prefix_count=prefix_count,
                frontier_count=frontier_count,config=self.config,excluded=excluded)
            if operation is not None:return operation
        return dict(base=None,kind='recoverable_prefix_refinement',requested_steps=1)

    def _execute_recovery(self,kernel,operation,remaining,prefix_count,frontier_count,decision_us):
        base=operation['base']
        if base is None:
            return kernel._refine(decision_us,budget=min(1,remaining),
                prefix_cap=kernel.prefix_count+self.config.max_total_prefix_nodes-prefix_count,
                frontier_cap=len(kernel.frontier)+self.config.max_total_frontier-frontier_count)
        done,handle=0,base
        while kernel._prefix(handle).depth<kernel.n:
            choice=min(kernel._choices(handle),key=lambda p:(-kernel.transition_log_weight(handle,p),p))
            handle=kernel._child(handle,choice);done+=1
        kernel.seed_complete_action(handle,decision_us)
        return done,False

    def _select_recovery(self,candidates,*,summaries,priority_weights,masses,remaining,
                         prefix_count,frontier_count,excluded,completed,decision_us):
        """Scheduling hook only; legal operations and their executor stay shared."""
        component=min(candidates,key=lambda c:(-priority_weights[c]*masses[c][-1],c))
        operation=self._recovery_operation(self.kernels[component],remaining,prefix_count,
            frontier_count,excluded[component],completed[component])
        return component,operation,{}

    def _component_inference(self,old_n,rescored_indices,*,reference_us,decision_us,indices):
        previous=self._old_search_state() if self.config.enable_recovery else {}
        predictions,audit=super()._component_inference(old_n,rescored_indices,
            reference_us=reference_us,decision_us=decision_us,indices=indices)
        audit.update(recovery_budget=self.config.recovery_budget,
                     recovery_search_steps=0,recovery_allocation_trace=[],recovery_backbone_unchanged=True,
                     recovery_allocation_scope=self.config.recovery_allocation_scope,
                     recovery_task_scope=self._event_recovery_scope,
                     recovery_scope_changes_decision_loss=False)
        if not self.config.enable_recovery:
            return predictions,audit
        summaries={s['component']:s for s in audit['components']}
        priority_weights,scope_counts=allocation_weights(self,summaries,self._event_recovery_scope,reference_us)
        restarted={};masses={}
        for component,summary in summaries.items():
            kernel=self.kernels[component]
            restarted[component]=self._restore_cover(kernel,previous.get(component),set(kernel.active),decision_us)
            masses[component]=kernel._mass()
        prefix_count,frontier_count=self._storage_counts()
        if (prefix_count>self.config.max_total_prefix_nodes or frontier_count>self.config.max_total_frontier):
            raise ValueError('restored recovery cover exceeds shared capacity; no support dropped')
        remaining=self.config.recovery_budget;blocked=set();trace=[]
        completed={c:0 for c in summaries};excluded={c:set() for c in summaries}
        spent={c:0 for c in summaries};limited={c:False for c in summaries}
        while remaining:
            eligible=[c for c,s in summaries.items() if c not in blocked and priority_weights[c]>0
                and masses[c][-1]>0 and any(self.kernels[c]._prefix(h).depth<self.kernels[c].n
                                             for h in self.kernels[c].frontier)]
            if not eligible:break
            component,operation,selection=self._select_recovery(eligible,summaries=summaries,
                priority_weights=priority_weights,masses=masses,remaining=remaining,
                prefix_count=prefix_count,frontier_count=frontier_count,excluded=excluded,
                completed=completed,decision_us=decision_us)
            kernel=self.kernels[component]
            before_prefix,before_frontier=kernel.prefix_count,len(kernel.frontier)
            done,stopped=self._execute_recovery(kernel,operation,remaining,prefix_count,frontier_count,decision_us)
            if done<0 or done>remaining:raise ValueError('recovery operation exceeded declared step budget')
            if operation['base'] is not None:
                excluded[component].add(operation['base']);completed[component]+=1
            prefix_count+=kernel.prefix_count-before_prefix;frontier_count+=len(kernel.frontier)-before_frontier
            if (prefix_count>self.config.max_total_prefix_nodes or frontier_count>self.config.max_total_frontier
                    or len(kernel.frontier)>kernel.config.state.max_frontier):
                raise ValueError('recovery search exceeded shared storage capacity')
            remaining-=done;spent[component]+=done;limited[component]|=stopped
            masses[component]=kernel._mass()
            trace.append(dict(component=component,kind=operation['kind'],charged_search_steps=done,
                              eta_after=masses[component][-1],priority_weight=priority_weights[component],
                              **selection))
            if not done:blocked.add(component)
        predictions=[];final=[];state_work=0
        for component,backbone in summaries.items():
            kernel=self.kernels[component];active,absolute,retained,upper,eta=masses[component]
            # The original beam candidate is a legal same-model fallback action,
            # never a moment-matched state. It is not forced by a risk threshold.
            chosen,decision=kernel._decode(active,absolute,retained,eta,backbone['decision_indices'],
                                           backbone['output_handle'],fallback_on_risk=False)
            branches=[]
            for h in sorted(set(active)|{chosen}):
                state=[] if backbone['expired_output_only'] else kernel._predict(h,reference_us)
                branches.append(dict(handle=h,state_sha256=digest(state)))
                if h==chosen:predictions.extend(state)
            state_work+=kernel.state_updates
            if state_work>self.config.state.max_replay_operations:
                raise ValueError('shared beam plus recovery state-replay work capacity exhausted')
            old=previous.get(component)
            context=dict(old_n=old['n'] if old else 0,old_active=old['active'] if old else set())
            change=(ComponentChange(component,tuple(backbone['predecessors']),(),True)
                    if backbone['merge_retained_beams_only'] else None)
            current_meta=kernel.meta
            try:
                if old is not None:kernel.meta=old['meta']
                recoveries=self.store.output_recoveries(component,kernel,chosen,context,change)
            finally:kernel.meta=current_meta
            # The backbone already advanced this kernel's event clock once in
            # the still-open outer transaction; replace only search/output data.
            kernel.meta=dict(kernel.meta,frontier=sorted(kernel.frontier),active=sorted(kernel.active),output=chosen)
            kernel._set('state',kernel.meta)
            kernel._set('recovery_frontier',sorted(kernel.frontier))
            kernel._set('beam_support_complete',not kernel.frontier)
            summary=dict(backbone,
                active=[dict(handle=h,sha256=kernel._prefix(h).sha256,log_weight=w) for h,w in zip(active,absolute)],
                frontier=[dict(handle=h,sha256=kernel._prefix(h).sha256,depth=kernel._prefix(h).depth,
                               log_upper=kernel._upper(h)) for h in sorted(kernel.frontier)],
                log_retained=retained,log_partition_upper=upper,eta_upper=eta,decision=decision,
                output_handle=chosen,output_sha256=kernel._prefix(chosen).sha256,branches=branches,
                full_model_support_still_in_beam=not kernel.frontier,complete_raw_support_retained=True,
                prefix_nodes=kernel.prefix_count,state_updates=kernel.state_updates,recovery_events=recoveries,
                backbone_active=backbone['active'],backbone_output_handle=backbone['output_handle'],
                backbone_retained_log_mass=backbone['log_retained'],backbone_pruning_only=True,
                recovery_steps=spent[component],recovery_resource_limited=limited[component],
                recovery_priority_weight=priority_weights[component],
                recovery_scope_raw_nodes=None if scope_counts is None else scope_counts[component],
                recovery_cover_restarted_from_root=restarted[component] or old is None)
            self.db.execute('INSERT OR REPLACE INTO component_summaries VALUES(?,?)',(component,canonical(summary)))
            final.append(summary)
        if len({p['track_id'] for p in predictions})!=len(predictions):
            raise ValueError('recovery outputs contain duplicate track IDs')
        log_ratio=math.fsum(s['log_retained']-s['log_partition_upper'] for s in final)
        audit.update(components=final,recovery_search_steps=self.config.recovery_budget-remaining,
            recovery_allocation_trace=trace,allocation_policy='node_beam_plus_weighted_model_omission_recovery',
            total_prefix_nodes=prefix_count,total_frontier=frontier_count,state_updates=state_work,
            shared_cache_entries=len(self.shared_cache),recovery_preserves_unexpanded_support=True,
            log_partition_upper=math.fsum(s['log_partition_upper'] for s in final),
            product_omitted_mass_upper=max(0.,min(1.,-math.expm1(min(0.,log_ratio)))),
            weighted_truncation_risk_upper=math.fsum(s['weight']*s['eta_upper'] for s in final),
            model_regret_upper=math.fsum(s['weight']*s['decision']['risk_bound'] for s in final),
            recovery_extra_state_updates=state_work-audit['state_updates'],
            backbone_state_computation_included=True,work_counters_are_latency_equivalence=False)
        return sorted(predictions,key=lambda p:p['track_id']),audit
