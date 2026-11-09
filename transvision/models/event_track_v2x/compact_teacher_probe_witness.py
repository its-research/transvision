"""New offline teacher storage version; frozen v1 producers remain unchanged."""
import time
from .exclusive_teacher_probe_witness import ExclusiveWitnessTeacher
from .compact_witness_storage import canonical, create_tables, store, FORMAT


class CompactWitnessTeacher(ExclusiveWitnessTeacher):
    SCHEMA = 'persistent_exclusive_completion_teacher_shared_probe_witness_v2'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        create_tables(self.db)

    def _store_witness(self, value):
        started = time.perf_counter()
        sha = store(self.db, value)
        self._witness_encoding_seconds += time.perf_counter() - started
        self._witness_peak_raw_encoding_bytes = max(self._witness_peak_raw_encoding_bytes, len(canonical(value)))
        return sha

    def _allocation_summary(self):
        summary = super()._allocation_summary()
        chunks, compressed = self.db.execute(
            'SELECT count(*),coalesce(sum(length(payload)),0) FROM priority_witness_chunks').fetchone()
        summary['raw_probe_witness_compressed_bytes'] += compressed
        summary['raw_probe_witness_storage'] = FORMAT
        summary['raw_probe_witness_shared_chunks'] = chunks
        # These are payload bytes, never a substitute for measured SQLite size.
        return summary
