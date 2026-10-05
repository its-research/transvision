"""Explicit candidate execution path; existing replay entrypoints are unchanged."""
from .batched_row_context_scoring import BatchedLearnedRowScorer
from .forest_row_context import build_row_contexts
from .forest_tracking import RawIdentityDetection
from .identity_forest import ForestFactors, digest
from .persistent_cache_stream import PersistentForestCacheStream, RowContextScorer


class BatchedRowContextScorer(RowContextScorer):
    def __init__(self, tracker, scorer):
        if type(scorer) is not BatchedLearnedRowScorer:
            raise TypeError('explicit admitted batching candidate required')
        super().__init__(tracker, scorer)
        self.execution_batch = scorer.max_batch
        self.binding = digest([self.binding, scorer.execution_recipe, self.execution_batch])

    def rows(self, observations, decision_us):
        new = tuple(observations)
        if (type(decision_us) is not int or any(type(o) is not RawIdentityDetection
                or o.node.arrival_us > decision_us or o.sequence_id != self.tracker.sequence_id for o in new)):
            raise ValueError('unavailable or invalid raw row-context input')
        if self.scorer.signature != self.scorer_signature or self.scorer.max_batch != self.execution_batch:
            raise ValueError('frozen persistent scorer or execution batch changed')
        tracker = self.tracker

        def older(lo, hi):
            for index, in tracker.db.execute(
                    'SELECT i FROM observations WHERE state_us BETWEEN ? AND ? ORDER BY i', (lo, hi)):
                yield index, tracker._observation(index)

        contexts = build_row_contexts(new, old_count=tracker.n, older_candidates=older,
            config=tracker.config.state, decision_us=decision_us, sequence_id=tracker.sequence_id)
        factors = self.scorer.score_contexts(contexts)
        if len(factors) != len(contexts):
            raise ValueError('batch changed query coverage')
        rows = []
        for context, factor in zip(contexts, factors, strict=True):
            if (type(factor) is not ForestFactors
                    or factor.nodes != tuple(o.node for o in context.observations)
                    or tuple(tuple(p for p, _ in row) for row in factor.rows) != context.support):
                raise ValueError('batch changed nodes or candidate support')
            rows.append(tuple((-1 if p < 0 else context.indices[p], weight)
                for p, weight in factor.rows[-1]))
        return tuple(rows), tuple(context.indices for context in contexts)


def create_batched_cache_stream(cache, tracker, original_scorer, *, max_batch=64, **kwargs):
    """Start a distinct execution lineage on a new store, never alter a live one.

    The original cache ingestion, first-arrival handling and transactional
    tracker step are inherited unchanged. Binding the recipe and batch into
    the scorer binding prevents accidental resume as the serial protocol.
    Resume support is intentionally not part of this candidate factory.
    """
    if (tracker.n or tracker.db.execute('SELECT count(*) FROM events').fetchone()[0]
            or tracker.db.execute('SELECT count(*) FROM cache_receipts').fetchone()[0]
            or tracker.db.execute("SELECT count(*) FROM meta WHERE k IN ('cache_binding','scorer_binding')").fetchone()[0]):
        raise ValueError('batch candidate requires a new, unbound tracker')
    scorer = BatchedLearnedRowScorer(original_scorer, max_batch=max_batch)
    stream = PersistentForestCacheStream(cache, tracker, original_scorer, **kwargs)
    stream.scoring = BatchedRowContextScorer(tracker, scorer)
    stream.configuration_sha256 = digest(stream._configuration())
    return stream
