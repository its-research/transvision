"""Wall-clock progress estimates; never interpreted as acceptance evidence."""
import datetime
import json
import os
import time
from collections import deque


class ExperimentProgress:
    def __init__(self, stage, total, *, interval_seconds=30, clock=time.monotonic,
                 emit=None, context=None, eta_scope='current stage only'):
        if type(total) is not int or total <= 0:
            raise ValueError('positive integer total required')
        self.stage, self.total = stage, total
        self.context = json.loads(json.dumps(context or {}, allow_nan=False))
        self.eta_scope = eta_scope
        self.progress_id = f'{os.getpid()}:{time.time_ns()}'
        self.clock, self.interval = clock, interval_seconds
        self.emit = emit or (lambda row: print(json.dumps(row, sort_keys=True), flush=True))
        self.started = self.last_time = clock()
        self.last_done = 0
        self.samples = deque([(0, self.started)], maxlen=4)
        self.update(0, force=True)

    def update(self, done, *, force=False):
        if type(done) is not int or not self.last_done <= done <= self.total:
            raise ValueError('progress must be monotonic and within total')
        now = self.clock()
        if not force and done != self.total and now - self.last_time < self.interval:
            return
        elapsed = now - self.started
        rate = done / elapsed if done > 0 and elapsed > 0 else None
        if done > self.samples[-1][0]:
            self.samples.append((done, now))
        oldest_done, oldest_time = self.samples[0]
        recent_elapsed = now - oldest_time
        recent_rate = ((done - oldest_done) / recent_elapsed
                       if done > oldest_done and recent_elapsed > 0 else None)
        eta_rate = min(rate, recent_rate) if rate and recent_rate else rate
        remaining = (self.total - done) / eta_rate if eta_rate else None
        row = dict(kind='rbf_experiment_progress_v1', stage=self.stage,
                   progress_id=self.progress_id, context=self.context,
                   eta_scope=self.eta_scope, whole_experiment_eta_seconds=None,
                   completed=done, total=self.total, progress_percent=100*done/self.total,
                   elapsed_seconds=elapsed, units_per_second=rate,
                   recent_units_per_second=recent_rate,
                   eta_seconds=remaining,
                   eta_status='warming_up' if rate is None else 'estimated',
                   eta_method='conservative_cumulative_recent_window_rate',
                   experiment_acceptance_proven=False)
        row['estimated_finish_utc'] = ((datetime.datetime.now(datetime.timezone.utc)
            + datetime.timedelta(seconds=remaining)).isoformat() if remaining is not None else None)
        self.emit(row)
        self.last_time, self.last_done = now, done
