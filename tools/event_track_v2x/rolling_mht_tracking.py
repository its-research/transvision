"""Declared age-limited local MHT support; archival rows are not recovery support.

Expire arbitrary arrived observation positions, preserving every surviving
history compatible with the last current output on those positions. This does
not reclaim archived SQL storage or implement a recoverable separator prior.
"""
from dataclasses import dataclass
import json

from tools.event_track_v2x.persistent_mht_tracking import PersistentMHTConfig, PersistentMHTTracker
from transvision.models.event_track_v2x.identity_forest import digest


@dataclass(frozen=True)
class RollingMHTConfig(PersistentMHTConfig):
    history_retention_us: int = 150000


class RollingMHTTracker(PersistentMHTTracker):
    CONFIG_TYPE = RollingMHTConfig
    SCHEMA = 'persistent_rolling_age_mask_mht_v1'

    def _policy(self):
        return dict(kind='irreversible_arrived_observation_age_mask_v1',
                    retention_us=self.config.history_retention_us,
                    predicate='state_us < decision_us - retention_us',
                    incoming_outside_window='reject_entire_new_event',
                    exact_old_retransmission='return_original_bytes_before_age_check',
                    expired_identity='condition_on_previous_current_output',
                    within_window='preserve_all_compatible_retained_histories',
                    archived_SQL='audit_and_conditional_state_replay_only',
                    full_history_guarantee=False, bounded_SQL_storage=False)

    def _record(self):
        row = self.db.execute("SELECT v FROM meta WHERE k='rolling_age_mask'").fetchone()
        return None if row is None else json.loads(row[0])

    def _roots(self, handle):
        roots = []
        for i, parent in enumerate(self.parents(handle)):
            roots.append(i if parent < 0 else roots[parent])
        return roots

    def _load(self):
        super()._load()
        self.expiry_transaction = None
        record = self._record()
        if record is not None:
            if record['policy_sha256'] != digest(self._policy()):
                raise ValueError('rolling support policy binding changed')
            for handle in self.active | self.frontier:
                roots = self._roots(handle)
                if any(roots[i] != root for i, root in record['constraints']):
                    raise ValueError('retained history violates expired identity constraints')
            if self.meta['events']:
                blob = self.db.execute('SELECT audit FROM events ORDER BY ordinal DESC LIMIT 1').fetchone()[0]
                if json.loads(blob)['rolling_age_mask_record'] != record:
                    raise ValueError('rolling support record and committed audit differ')

    def step(self, observations, rows, **kwargs):
        new = tuple(observations)
        # The base request digest remains authoritative for exact/conflicting IDs.
        duplicate = self.db.execute('SELECT 1 FROM events WHERE event_id=?',
                                    (kwargs.get('event_id'),)).fetchone()
        if not duplicate:
            decision, reference = kwargs.get('decision_us'), kwargs.get('reference_us')
            if type(decision) is not int or type(reference) is not int:
                raise ValueError('integer rolling event times required')
            cutoff = decision - self.config.history_retention_us
            for observation in new:
                if observation.state_us < cutoff:
                    raise ValueError('new observation outside rolling support window; event rejected')
                if observation.state_us > reference or observation.node.information_us > reference:
                    raise ValueError('new observation after current reference; event rejected')
        self.expiry_transaction = None
        return super().step(new, rows, **kwargs)

    def _promote(self, decision):
        cutoff = decision - self.config.history_retention_us
        expired = [i for i, in self.db.execute(
            'SELECT i FROM observations WHERE i<? AND state_us<? ORDER BY i',
            (self.event_start, cutoff))]
        old = self._record()
        previous = [] if old is None else [i for i, root in old['constraints']]
        if not set(previous) <= set(expired):
            raise ValueError('rolling expired support cannot be reinstated')
        if expired == previous:
            return
        before = self.active | self.frontier
        chosen = self.meta['output']
        if chosen not in before:
            raise ValueError('rolling current output is outside predecessor support')
        roots = self._roots(chosen)
        constraints = [[i, roots[i]] for i in expired]
        kept = []
        for handle in sorted(before):
            candidate = self._roots(handle)
            if all(candidate[i] == root for i, root in constraints):
                kept.append(handle)
        if chosen not in kept:
            raise ValueError('rolling constraint removed its own current output')
        record = dict(policy_sha256=digest(self._policy()), decision_us=decision,
                      cutoff_us=cutoff, previous_expired_indices=previous,
                      constraints=constraints, previous_observations=self.event_start,
                      previous_output_handle=chosen,
                      previous_retained=[dict(handle=h, parents=list(self.parents(h)),
                                              sha256=self._prefix(h).sha256) for h in sorted(before)],
                      kept_handles=kept, discarded_handles=sorted(before - set(kept)),
                      full_history_guarantee_revoked=True)
        had_frontier = bool(self.frontier)
        self.frontier = set(kept) if had_frontier else set()
        self.active = set() if had_frontier else set(kept)
        self._set('rolling_age_mask', record)
        self.expiry_transaction = record

    def _finalize_audit(self, audit):
        audit = super()._finalize_audit(audit)
        record = self._record()
        audit.update(rolling_age_mask_policy=self._policy(),
                     rolling_age_mask_transaction=self.expiry_transaction,
                     rolling_age_mask_record=record,
                     history_coverage_scope='conditional_on_expired_identity_mask' if record else 'unexpired_MHT_support',
                     full_history_risk_or_mass_certificate=False,
                     archived_support_reconstruction_allowed=False,
                     bounded_SQL_storage=False)
        return audit
