"""Recoverable prefix-minus-leaves representation; legacy backends preserved.

The live kernel's mass, component priorities and decoder bounds use disjoint
residual regions. Immutable prefix handles still support the same budgeted
expansion operations. Exclusions change whenever retained classes change, so
an evicted class immediately returns to its covering residual region.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

from .covered_completion_tracking import (
    CoveredCompletionTracker, CoveredCompletionTeacher, CoveredCompletionLearned,
    PersistentCoveredCompletionConfig,
)
from .detection_cache_v2 import canonical
from .hypothesis_bank import logsumexp
from .identity_forest import digest
from .persistent_component_store import PersistentComponentStore, _RootPartitionKernel
from .residual_frontier import RECIPE, residual_log_upper


@dataclass(frozen=True)
class PersistentExclusiveCompletionConfig(PersistentCoveredCompletionConfig):
    residual_partition_version: int = 1

    def __post_init__(self):
        super().__post_init__()
        if type(self.residual_partition_version) is not int or self.residual_partition_version != 1:
            raise ValueError('unsupported explicit residual partition version')


class _ExclusivePartitionKernel(_RootPartitionKernel):
    def residual_regions(self, active, weights):
        excluded_count = {h: 0 for h in active}
        regions = []
        for h in sorted(self.frontier):
            prefix = self._prefix(h)
            excluded = []
            for leaf, weight in zip(active, weights):
                if self.ancestor(leaf, prefix.depth) == h:
                    if self._prefix(leaf).depth != self.n:
                        raise ValueError('residual exclusion must reference a complete current class')
                    excluded_count[leaf] += 1
                    excluded.append(dict(handle=leaf, sha256=self._prefix(leaf).sha256, log_weight=weight))
            gross = self._upper(h)
            value = residual_log_upper(gross, [r['log_weight'] for r in excluded], nodes=self.n)
            body = dict(handle=h, sha256=prefix.sha256, depth=prefix.depth,
                        nodes=self.n, factor_revision=self.revision, region_recipe=RECIPE,
                        excluded_leaves=excluded, log_gross_upper=gross, **value)
            body['region_sha256'] = digest(body)
            regions.append(body)
        if any(v > 1 for v in excluded_count.values()):
            raise ValueError('retained class is inside overlapping gross prefix regions')
        if sum(excluded_count.values()) > self.config.state.active_limit:
            raise ValueError('residual exclusion storage exceeds retained-class capacity')
        return regions

    def _mass(self):
        active = sorted(self.active, key=lambda h: (-self._weight(h), self._prefix(h).sha256))
        weights = [self._weight(h) for h in active]
        retained = logsumexp(weights)
        regions = self.residual_regions(active, weights)
        upper = logsumexp([r['log_upper'] for r in regions]+weights, upper=True)
        eta = 0. if not regions else 1. if not active else max(0., min(1., -math.expm1(min(0., retained-upper))))
        return active, weights, retained, upper, eta


class _ExclusiveStore(PersistentComponentStore):
    def open_kernel(self, component, config, *, caches=None):
        kernel = super().open_kernel(component, config, caches=caches)
        version = kernel.db.execute("SELECT v FROM meta WHERE k='residual_partition_version'").fetchone()
        if version is None:
            if kernel.n or not self.db.in_transaction:
                raise ValueError('existing history lacks explicit residual partition binding')
            kernel._set('residual_partition_version', 1)
        elif version[0] != canonical(1):
            raise ValueError('residual partition binding changed')
        kernel.__class__ = _ExclusivePartitionKernel
        return kernel


class _ExclusiveRegions:
    CONFIG_TYPE = PersistentExclusiveCompletionConfig

    def _setup_store(self):
        super()._setup_store()
        self.store = _ExclusiveStore(self, max_components=self.config.max_components,
                                     max_member_rows=self.config.max_member_rows)

    def _component_inference(self, *args, **kwargs):
        predictions, inference = super()._component_inference(*args, **kwargs)
        references, bytes_used = 0, 0
        for summary in inference['components']:
            kernel = self.kernels[summary['component']]
            active = [r['handle'] for r in summary['active']]
            weights = [r['log_weight'] for r in summary['active']]
            regions = kernel.residual_regions(active, weights)
            assert [r['handle'] for r in regions] == [r['handle'] for r in summary['frontier']]
            summary['representation'] = 'exclusive_root_partition_regions_v1'
            summary['frontier'] = regions
            summary['residual_partition_recipe'] = RECIPE
            summary['residual_exclusion_references'] = sum(len(r['excluded_leaves']) for r in regions)
            payload = canonical(regions)
            summary['residual_region_metadata_bytes'] = len(payload)
            # Persist exactly the live residual sets, not just a derived report.
            kernel._set('residual_regions', regions)
            self.db.execute('INSERT OR REPLACE INTO component_summaries VALUES(?,?)',
                            (summary['component'], canonical(summary)))
            references += summary['residual_exclusion_references']
            bytes_used += len(payload)
        if references > len(inference['components'])*self.config.state.active_limit:
            raise ValueError('live residual exclusions exceed explicit branch capacity')
        return predictions, dict(inference, residual_exclusion_references=references,
                                 residual_region_metadata_bytes=bytes_used,
                                 residual_regions_counted_in_database_storage=True)

    def _audit_properties(self):
        return dict(super()._audit_properties(), explicit_residual_partition=True,
                    residual_partition_recipe=RECIPE, residual_partition_version=1,
                    float64_subtraction_padding_declared=True,
                    formal_numeric_certificate=False, true_posterior_or_metric_bound=False)


class ExclusiveCompletionTracker(_ExclusiveRegions, CoveredCompletionTracker):
    SCHEMA = 'persistent_exclusive_completion_component_identity_v1'


class ExclusiveCompletionTeacher(_ExclusiveRegions, CoveredCompletionTeacher):
    SCHEMA = 'persistent_exclusive_completion_allocation_teacher_v1'


class ExclusiveCompletionLearned(_ExclusiveRegions, CoveredCompletionLearned):
    SCHEMA = 'persistent_exclusive_completion_learned_allocation_v1'
