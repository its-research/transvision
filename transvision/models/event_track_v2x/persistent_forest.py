"""Disk-backed joint identity forest with shared, recoverable history prefixes.

This is a long-sequence inference/state backend, not a trained scoring policy.
The caller supplies fixed-support raw-node factors; learned scoring, component
allocation and the official cache schedule are separate integrations. No old
identity is MAP-collapsed at a time boundary. Raw factors, unresolved prefixes,
branch states and immutable outputs remain in the task-local SQLite database.
"""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import sqlite3

import numpy as np

from .detection_cache_v2 import canonical
from .forest_tracking import ForestTrackingConfig, RawIdentityDetection
from .hypothesis_bank import logsumexp
from .identity_forest import IdentityNode, digest
from .recoverable_states import _state
from .tracking_v2 import ci, propagate


@dataclass(frozen=True)
class PersistentForestConfig:
    state: ForestTrackingConfig = ForestTrackingConfig()
    max_observations: int = 100_000
    max_new_observations: int = 256
    max_prefix_nodes: int = 500_000
    max_events: int = 100_000
    prefix_cache_entries: int = 4096
    sqlite_cache_kib: int = 8192
    max_decision_nodes: int = 4096

    def __post_init__(self):
        if type(self.state) is not ForestTrackingConfig or self.state.component_mode:
            raise ValueError('persistent backend currently uses one joint forest, not component allocation')
        if any(type(v) is not int or v < 1 for k, v in asdict(self).items() if k != 'state'):
            raise ValueError('positive integer persistence limits required')
        defaults = ForestTrackingConfig()
        for key in ('max_nodes', 'max_commits', 'max_discovered', 'action_budget', 'action_frontier',
                    'potential_context', 'max_component_nodes', 'max_components',
                    'max_total_frontier', 'max_total_discovered'):
            if getattr(self.state, key) != getattr(defaults, key):
                raise ValueError('finite-window option '+key+' is not used; configure persistent limits instead')


@dataclass(frozen=True)
class PersistentForestCommit:
    prediction_json: bytes
    audit_json: bytes

    @property
    def prediction(self):
        return json.loads(self.prediction_json)

    @property
    def audit(self):
        return json.loads(self.audit_json)


@dataclass(frozen=True)
class _Prefix:
    handle: int
    parent_handle: int | None
    depth: int
    choice: int | None
    root: int | None
    previous_root: int | None
    jumps: tuple[int, ...]
    sha256: str


def _path(value):
    path = Path(value).absolute()
    if any(p.is_symlink() for p in (path, *path.parents)):
        raise ValueError('persistent store may not traverse symlinks')
    return path


def _raw(value):
    data = json.loads(value)
    return RawIdentityDetection(**dict(data, node=IdentityNode(**data['node'])))


def _file_hash(path):
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b''):
            value.update(chunk)
    return value.hexdigest()


class PersistentForestTracker:
    """Append/rescore + inference + branch replay in ONE durable transaction.

``state.window_us`` selects the current decision-loss interval; it is NOT an
expiration time. Observation/prefix/event counts have separate hard storage
caps. Prefix metadata caches and SQLite page cache are bounded; row-bound arrays
use O(number of observations) scalar metadata, not O(history x hypotheses).
"""
    def __init__(self, path, *, sequence_id, config=None):
        self.path = _path(path)
        self.config = config or PersistentForestConfig()
        if not isinstance(self.config, PersistentForestConfig) or not isinstance(sequence_id, str) or not sequence_id:
            raise ValueError('validated config and nonempty sequence required')
        # Exclusive creation before opening SQLite prevents accidental replacement.
        with self.path.open('xb'):
            pass
        self.sequence_id = sequence_id
        self._connect()
        self.db.executescript('''
            CREATE TABLE meta(k TEXT PRIMARY KEY, v BLOB NOT NULL);
            CREATE TABLE observations(i INTEGER PRIMARY KEY, node_id TEXT UNIQUE NOT NULL,
                source INTEGER NOT NULL, frame TEXT NOT NULL, detection_index INTEGER NOT NULL,
                state_us INTEGER NOT NULL, arrival_us INTEGER NOT NULL, score REAL NOT NULL,
                raw BLOB NOT NULL, sha TEXT NOT NULL,
                UNIQUE(source, frame, detection_index));
            CREATE INDEX observation_slots ON observations(source,frame,i);
            CREATE INDEX observation_time ON observations(state_us,i);
            CREATE TABLE potentials(i INTEGER NOT NULL, p INTEGER NOT NULL, w REAL NOT NULL,
                PRIMARY KEY(i,p));
            CREATE TABLE prefixes(h INTEGER PRIMARY KEY, parent INTEGER, depth INTEGER NOT NULL,
                choice INTEGER, root INTEGER, previous_root INTEGER, jumps BLOB NOT NULL, sha TEXT NOT NULL,
                UNIQUE(parent,choice));
            CREATE TABLE weights(h INTEGER PRIMARY KEY, revision INTEGER NOT NULL, value REAL NOT NULL);
            CREATE TABLE states(h INTEGER PRIMARY KEY, payload BLOB NOT NULL);
            CREATE TABLE discovered(h INTEGER PRIMARY KEY, decision_us INTEGER NOT NULL);
            CREATE TABLE events(ordinal INTEGER PRIMARY KEY, event_id TEXT UNIQUE NOT NULL,
                request TEXT NOT NULL, prediction BLOB NOT NULL, audit BLOB NOT NULL);
            CREATE TABLE cache_receipts(side TEXT NOT NULL, frame TEXT NOT NULL,
                arrival_us INTEGER NOT NULL, frame_sha TEXT NOT NULL, PRIMARY KEY(side,frame));
        ''')
        self.db.execute('INSERT INTO prefixes VALUES(0,NULL,0,NULL,NULL,NULL,?,?)',
                        (canonical([]), digest(['persistent-forest-root', sequence_id])))
        self._set('schema', 'persistent_identity_forest_v1')
        self._set('sequence_id', sequence_id)
        self._set('config', asdict(self.config))
        self._set('state', dict(n=0, revision=0, frontier=[0], active=[], output=0,
            decision_us=-1, reference_us=-1, events=0, prediction_sha256='0'*64, audit_sha256='0'*64))
        self._load()

    def _connect(self):
        self.configuration_sha256 = digest(asdict(self.config))
        self.db = sqlite3.connect(self.path, isolation_level=None)
        self.db.execute('PRAGMA journal_mode=DELETE')
        self.db.execute('PRAGMA synchronous=FULL')
        self.db.execute('PRAGMA temp_store=FILE')
        self.db.execute('PRAGMA cache_size='+str(-self.config.sqlite_cache_kib))
        self.cache = OrderedDict()
        self.row_cache = OrderedDict()

    @classmethod
    def open(cls, path, *, expected_prediction_sha256, expected_database_sha256):
        path = _path(path)
        if (not path.is_file() or any(not isinstance(v, str) or len(v) != 64
                or any(c not in '0123456789abcdef' for c in v)
                for v in (expected_prediction_sha256, expected_database_sha256))
                or _file_hash(path) != expected_database_sha256):
            raise ValueError('existing database, sealed file hash and expected output commit required')
        # mode=rw cannot create a missing database. Read config before connecting
        # with the declared cache limits; never use SQLite's implicit create mode.
        db = sqlite3.connect(path.as_uri()+'?mode=ro', uri=True)
        try:
            values = {k: json.loads(v) for k, v in db.execute('SELECT k,v FROM meta')}
        finally:
            db.close()
        if (values.get('schema') != 'persistent_identity_forest_v1'
                or values['state']['prediction_sha256'] != expected_prediction_sha256):
            raise ValueError('unexpected persistent store identity or output head')
        obj = cls.__new__(cls)
        obj.path, obj.sequence_id = path, values['sequence_id']
        configuration = values['config']
        obj.config = PersistentForestConfig(**dict(configuration,
            state=ForestTrackingConfig(**configuration['state'])))
        obj._connect()
        try:
            if obj.db.execute('PRAGMA integrity_check').fetchone()[0] != 'ok':
                raise ValueError('persistent SQLite integrity failed')
            obj._load()
            if obj.meta['events']:
                record = obj.db.execute('SELECT prediction,audit FROM events ORDER BY ordinal DESC LIMIT 1').fetchone()
                result = PersistentForestCommit(*record)
                prediction = result.prediction
                recorded = prediction.pop('commit_sha256')
                if (digest(prediction) != recorded or recorded != expected_prediction_sha256
                        or digest(result.audit) != obj.meta['audit_sha256']
                        or result.audit['factor_rows_sha256'] != obj.factor_digest()
                        or result.audit['configuration_sha256'] != digest(asdict(obj.config))):
                    raise ValueError('persistent output, raw factors or configuration changed')
            return obj
        except BaseException:
            obj.close()
            raise

    def close(self):
        self.db.close()
        self.cache.clear()
        self.row_cache.clear()
        return _file_hash(self.path)

    def _set(self, key, value):
        self.db.execute('INSERT OR REPLACE INTO meta VALUES(?,?)', (key, canonical(value)))

    def _load(self):
        self.meta = json.loads(self.db.execute("SELECT v FROM meta WHERE k='state'").fetchone()[0])
        self.n = self.meta['n']
        self.revision = self.meta['revision']
        self.frontier, self.active = set(self.meta['frontier']), set(self.meta['active'])
        self.prefix_count = self.db.execute('SELECT count(*) FROM prefixes').fetchone()[0]
        self.cache.clear()
        self.row_cache.clear()
        self._bounds()

    def _bounds(self):
        sums = np.fromiter((logsumexp(w for _, w in self._row(i)) for i in range(self.n)),
                           dtype=float, count=self.n)
        self.suffix = np.r_[np.cumsum(sums[::-1])[::-1], 0.]
        self.suffix_abs = np.r_[np.cumsum(np.abs(sums[::-1]))[::-1], 0.]

    def _row(self, index):
        if index not in self.row_cache:
            self.row_cache[index] = tuple(self.db.execute('SELECT p,w FROM potentials WHERE i=? ORDER BY p', (index,)))
            if len(self.row_cache) > self.config.prefix_cache_entries:
                self.row_cache.popitem(last=False)
        self.row_cache.move_to_end(index)
        return self.row_cache[index]

    def _prefix(self, handle):
        if handle not in self.cache:
            row = self.db.execute('SELECT * FROM prefixes WHERE h=?', (handle,)).fetchone()
            if row is None:
                raise ValueError('missing recoverable prefix')
            self.cache[handle] = _Prefix(
                row[0], row[1], row[2], row[3], row[4], row[5], tuple(json.loads(row[6])), row[7])
            if len(self.cache) > self.config.prefix_cache_entries:
                self.cache.popitem(last=False)
        self.cache.move_to_end(handle)
        return self.cache[handle]

    def ancestor(self, handle, depth):
        prefix = self._prefix(handle)
        if type(depth) is not int or not 0 <= depth <= prefix.depth:
            raise ValueError('invalid ancestor depth')
        gap = prefix.depth-depth
        bit = 0
        while gap:
            if gap & 1:
                handle = self._prefix(handle).jumps[bit]
            gap >>= 1
            bit += 1
        return handle

    def parents(self, handle, *, maximum=10_000):
        """Bounded diagnostic materialization, not used for live inference."""
        prefix = self._prefix(handle)
        if prefix.depth > maximum:
            raise ValueError('diagnostic parent materialization cap exceeded')
        result = []
        while handle:
            p = self._prefix(handle)
            result.append(p.choice)
            handle = p.parent_handle
        return tuple(reversed(result))

    def _weight(self, handle):
        pending = []
        while handle:
            row = self.db.execute('SELECT revision,value FROM weights WHERE h=?', (handle,)).fetchone()
            if row is not None and row[0] == self.revision:
                value = row[1]
                break
            pending.append(handle)
            handle = self._prefix(handle).parent_handle
        else:
            value = 0.
        for h in reversed(pending):
            prefix = self._prefix(h)
            value = math.fsum([value, dict(self._row(prefix.depth-1))[prefix.choice]])
            self.db.execute('INSERT OR REPLACE INTO weights VALUES(?,?,?)', (h, self.revision, value))
        return value

    def _upper(self, handle):
        prefix, value = self._prefix(handle), self._weight(handle)
        # Conservative floating margin, not a formal interval certificate.
        margin = 64*np.finfo(float).eps*(self.n-prefix.depth+1)*max(1., abs(value), self.suffix_abs[prefix.depth])
        return float(value+self.suffix[prefix.depth]+margin)

    def _choices(self, handle):
        depth = self._prefix(handle).depth
        if depth == self.n:
            return ()
        source, frame = self.db.execute('SELECT source,frame FROM observations WHERE i=?', (depth,)).fetchone()
        forbidden = {self._prefix(self.ancestor(handle, i+1)).root for i, in self.db.execute(
            'SELECT i FROM observations WHERE source=? AND frame=? AND i<?', (source, frame, depth))}
        return tuple(p for p, _ in self._row(depth) if p < 0 or
                     self._prefix(self.ancestor(handle, p+1)).root not in forbidden)

    def _child(self, handle, choice):
        row = self.db.execute('SELECT h FROM prefixes WHERE parent=? AND choice=?', (handle, choice)).fetchone()
        if row:
            return row[0]
        if self.prefix_count >= self.config.max_prefix_nodes:
            raise ValueError('persistent prefix storage exhausted without deleting any hypothesis')
        parent = self._prefix(handle)
        root = parent.depth if choice < 0 else self._prefix(self.ancestor(handle, choice+1)).root
        previous, cursor = None, handle
        if choice >= 0:
            while cursor:
                candidate = self._prefix(cursor)
                if candidate.root == root:
                    previous = cursor
                    break
                cursor = candidate.parent_handle
        jumps = [handle]
        bit = 0
        while bit < len(self._prefix(jumps[-1]).jumps):
            jumps.append(self._prefix(jumps[-1]).jumps[bit])
            bit += 1
        sha = digest([parent.sha256, parent.depth, choice, root])
        cursor = self.db.execute('INSERT INTO prefixes(parent,depth,choice,root,previous_root,jumps,sha) VALUES(?,?,?,?,?,?,?)',
            (handle, parent.depth+1, choice, root, previous, canonical(jumps), sha))
        self.prefix_count += 1
        return cursor.lastrowid

    def _promote(self, decision):
        terminals = {h for h in self.frontier if self._prefix(h).depth == self.n}
        for h in terminals:
            self.db.execute('INSERT OR IGNORE INTO discovered VALUES(?,?)', (h, decision))
        ranked = sorted(self.active | terminals, key=lambda h: (-self._weight(h), self._prefix(h).sha256))
        retained = set(ranked[:self.config.state.active_limit])
        self.frontier.update(self.active-retained)
        self.frontier.difference_update(retained)
        self.active = retained

    def _refine(self, decision, *, budget=None, frontier_cap=None, prefix_cap=None):
        budget = self.config.state.expansion_budget if budget is None else budget
        frontier_cap = self.config.state.max_frontier if frontier_cap is None else frontier_cap
        prefix_cap = self.config.max_prefix_nodes if prefix_cap is None else prefix_cap
        if (type(budget) is not int or budget < 0 or type(frontier_cap) is not int or frontier_cap < 1
                or type(prefix_cap) is not int or prefix_cap < 1):
            raise ValueError('invalid shared refinement budget or storage cap')
        frontier_cap = min(frontier_cap, self.config.state.max_frontier)
        prefix_cap = min(prefix_cap, self.config.max_prefix_nodes)
        done, limited = 0, False
        for _ in range(budget):
            candidates = sorted((h for h in self.frontier if self._prefix(h).depth < self.n),
                                key=lambda h: (-self._upper(h), self._prefix(h).sha256))
            chosen = None
            for handle in candidates:
                choices = self._choices(handle)
                missing = sum(self.db.execute('SELECT 1 FROM prefixes WHERE parent=? AND choice=?',
                                              (handle, p)).fetchone() is None for p in choices)
                if (len(self.frontier)-1+len(choices) > frontier_cap
                        or self.prefix_count+missing > prefix_cap):
                    limited = True
                    continue
                chosen = handle, choices
                break
            if chosen is None:
                break
            handle, choices = chosen
            self.frontier.remove(handle)
            self.frontier.update(self._child(handle, p) for p in choices)
            self._promote(decision)
            done += 1
        return done, limited

    def _mass(self):
        active = sorted(self.active, key=lambda h: (-self._weight(h), self._prefix(h).sha256))
        weights = [self._weight(h) for h in active]
        retained = logsumexp(weights)
        upper = logsumexp([retained, logsumexp((self._upper(h) for h in self.frontier), upper=True)], upper=True)
        eta = (0. if not self.frontier else 1. if not active else
               max(0., min(1., -math.expm1(min(0., retained-upper)))))
        return active, weights, retained, upper, eta

    def _decode(self, active, weights, retained, eta, indices, fallback, *,
                fallback_on_risk=True, force_fallback=False):
        if not indices:
            return active[0] if active and not force_fallback else fallback, dict(risk_bound=0., conditional_risk=0.,
                relaxed_bayes_lower=0., optimization_gap=0., empty_loss_scope=True)
        if not active:
            return fallback, dict(risk_bound=1., conditional_risk=None, relaxed_bayes_lower=None,
                                   optimization_gap=None, empty_loss_scope=False)
        probabilities = [math.exp(w-retained) for w in weights]
        marginal = [{} for _ in indices]
        def roots(h):
            return tuple(self._prefix(self.ancestor(h, i+1)).root for i in indices)
        for h, probability in zip(active, probabilities):
            for row, root in zip(marginal, roots(h)):
                row[root] = row.get(root, 0.)+probability
        def risk(h):
            return math.fsum(1-row.get(root, 0.) for row, root in zip(marginal, roots(h)))/len(indices)
        lower = max(0., math.fsum(1-max(row.values()) for row in marginal)/len(indices))
        chosen = fallback if force_fallback else min(set(active) | {fallback}, key=lambda h: (risk(h), self._prefix(h).sha256))
        gap = max(0., risk(chosen)-lower)
        bound = min(1., eta+gap)
        if fallback_on_risk and bound > self.config.state.max_model_regret:
            chosen = fallback
            gap = max(0., risk(chosen)-lower)
            bound = min(1., eta+gap)
        return chosen, dict(risk_bound=bound, conditional_risk=risk(chosen), relaxed_bayes_lower=lower,
                            optimization_gap=gap, empty_loss_scope=False)

    def _observation(self, index):
        return _raw(self.db.execute('SELECT raw FROM observations WHERE i=?', (index,)).fetchone()[0])

    def _charge_state(self):
        self.state_updates += 1
        if self.state_updates > self.config.state.max_replay_operations:
            raise ValueError('branch-state work cap exceeded; transaction not committed')

    def _ensure_state(self, handle):
        pending = []
        cursor = handle
        while cursor is not None:
            row = self.db.execute('SELECT payload FROM states WHERE h=?', (cursor,)).fetchone()
            if row is not None:
                state = json.loads(row[0])
                break
            pending.append(cursor)
            cursor = self._prefix(cursor).previous_root
        else:
            state = None
        for h in reversed(pending):
            prefix = self._prefix(h)
            obs = self._observation(prefix.depth-1)
            order = [obs.state_us, obs.node.source_id, obs.node.node_id]
            if state is None:
                self._charge_state()
                mean, covariance = np.asarray(obs.mean), np.asarray(obs.covariance)
                first, last, max_score = obs.state_us, obs.state_us, obs.score
            elif order > state['last_order']:
                self._charge_state()
                mean, covariance = propagate(np.asarray(state['mean']), np.asarray(state['covariance']),
                    (obs.state_us-state['last_us'])/1e6, self.config.state.process_noise)
                mean, covariance = ci(mean, covariance, np.asarray(obs.mean), np.asarray(obs.covariance),
                                      self.config.state.ci_weight)
                first, last, max_score = state['first_us'], obs.state_us, state['max_score']
            else:
                # External ordering in SQLite keeps late replay's raw payloads
                # off the Python heap. Do not propagate an averaged state back.
                query = '''WITH RECURSIVE chain(h) AS (SELECT ? UNION ALL
                    SELECT p.previous_root FROM prefixes p JOIN chain c ON p.h=c.h WHERE p.previous_root IS NOT NULL)
                    SELECT o.raw FROM chain c JOIN prefixes p ON p.h=c.h JOIN observations o ON o.i=p.depth-1
                    ORDER BY o.state_us,o.source,o.node_id'''
                mean = covariance = time = None
                for raw, in self.db.execute(query, (h,)):
                    current = _raw(raw)
                    self._charge_state()
                    if mean is None:
                        mean, covariance = np.asarray(current.mean), np.asarray(current.covariance)
                    else:
                        mean, covariance = propagate(mean, covariance, (current.state_us-time)/1e6,
                                                      self.config.state.process_noise)
                        mean, covariance = ci(mean, covariance, np.asarray(current.mean), np.asarray(current.covariance),
                                              self.config.state.ci_weight)
                    time = current.state_us
                first, last = min(state['first_us'], obs.state_us), max(state['last_us'], obs.state_us)
                max_score = state['max_score']
                order = max(order, state['last_order'])
            mean, covariance = _state(mean, covariance, obs.score)
            state = dict(mean=mean, covariance=covariance, first_us=first, last_us=last,
                last_order=order, max_score=max(max_score, obs.score))
            self.db.execute('INSERT INTO states VALUES(?,?)', (h, canonical(state)))
        return state

    def _predict(self, handle, reference):
        latest, cursor = {}, handle
        while cursor:
            prefix = self._prefix(cursor)
            latest.setdefault(prefix.root, cursor)
            cursor = prefix.parent_handle
        predictions = []
        for root, h in sorted(latest.items()):
            state = self._ensure_state(h)
            c = self.config.state
            # Stream exact scalar witnesses through root links; storing a copied
            # witness list in EVERY prefix state would grow quadratically.
            query = '''WITH RECURSIVE chain(h) AS (SELECT ? UNION ALL
                SELECT p.previous_root FROM prefixes p JOIN chain c ON p.h=c.h WHERE p.previous_root IS NOT NULL)
                SELECT o.score,o.state_us FROM chain c JOIN prefixes p ON p.h=c.h
                JOIN observations o ON o.i=p.depth-1'''
            score = max(s*c.survival_per_second**(max(0, reference-t)/1e6)
                        for s, t in self.db.execute(query, (h,)))
            if (state['max_score'] < c.birth_score or score < c.prune_score
                    or max(0, reference-state['last_us']) > c.max_age_us):
                continue
            mean, covariance = propagate(np.asarray(state['mean']), np.asarray(state['covariance']),
                                          (reference-state['last_us'])/1e6, c.process_noise)
            mean, covariance = _state(mean, covariance, score)
            node_id = self.db.execute('SELECT node_id FROM observations WHERE i=?', (root,)).fetchone()[0]
            predictions.append(dict(track_id=self.sequence_id+':'+digest(['forest-birth', node_id])[:24],
                class_label='car', mean=mean, covariance=covariance, score=score))
        return predictions

    def factor_digest(self):
        h = hashlib.sha256()
        for i, raw, sha in self.db.execute('SELECT i,raw,sha FROM observations ORDER BY i'):
            if hashlib.sha256(raw).hexdigest() != sha:
                raise ValueError('stored raw observation changed')
            h.update(canonical([i, sha, list(self.db.execute('SELECT p,w FROM potentials WHERE i=? ORDER BY p', (i,)))])+b'\n')
        return h.hexdigest()

    def observations_between(self, start_us, end_us, *, maximum):
        """Bounded scorer input, not an implicit gate or silent truncation."""
        count = self.db.execute('SELECT count(*) FROM observations WHERE state_us BETWEEN ? AND ?',
                                (start_us, end_us)).fetchone()[0]
        if type(maximum) is not int or maximum < 1 or count > maximum:
            raise ValueError('scorer context capacity exceeded; no old observation was silently removed')
        return tuple((i, _raw(raw)) for i, raw in self.db.execute(
            'SELECT i,raw FROM observations WHERE state_us BETWEEN ? AND ? ORDER BY i', (start_us, end_us)))

    def _validate_row(self, index, row, *, rescore=False):
        if not isinstance(row, (tuple, list)) or len(row) > self.config.state.parent_limit+1:
            raise ValueError('parent support exceeds configured candidate capacity; no edge was removed')
        row = tuple(sorted((p, float(w)) for p, w in row))
        parents = tuple(p for p, _ in row)
        if (not row or len(set(parents)) != len(parents) or -1 not in parents
                or any(type(p) is not int or not -1 <= p < index for p in parents)
                or any(not math.isfinite(w) or abs(w) > 1000 for _, w in row)
                or rescore and parents != tuple(p for p, _ in self._row(index))):
            raise ValueError('invalid potentials or attempted rewrite of old candidate support')
        return row

    def _finalize_audit(self, audit):
        """Allow an inference backend to label its actual semantics before sealing."""
        return audit

    def step(self, observations, rows, *, frame_id, reference_us, decision_us, event_id,
             rescored_rows=(), decision_indices=None, cache_ingestion=None, scorer_binding=None):
        new, rows, rescored = tuple(observations), tuple(rows), tuple(rescored_rows)
        if (len(new) != len(rows) or len(new) > self.config.max_new_observations
                or any(type(o) is not RawIdentityDetection for o in new)
                or any(not isinstance(v, str) or not v for v in (frame_id, event_id))
                or type(reference_us) is not int or type(decision_us) is not int or not 0 <= reference_us <= decision_us):
            raise ValueError('invalid persistent event, raw detections, rows or time')
        if digest(asdict(self.config)) != self.configuration_sha256:
            raise ValueError('persistent tracker configuration changed')
        request = digest([frame_id, reference_us, decision_us, [asdict(o) for o in new], rows, rescored,
                          decision_indices, cache_ingestion, scorer_binding])
        old_event = self.db.execute('SELECT request,prediction,audit FROM events WHERE event_id=?', (event_id,)).fetchone()
        if old_event:
            if old_event[0] != request:
                raise ValueError('conflicting duplicate persistent event')
            return PersistentForestCommit(old_event[1], old_event[2])
        if (reference_us <= self.meta['reference_us'] or decision_us < self.meta['decision_us']
                or self.n+len(new) > self.config.max_observations or self.meta['events'] >= self.config.max_events):
            raise ValueError('nonmonotonic output or persistent observation/event capacity exceeded')
        previous_arrival = self.db.execute('SELECT max(arrival_us) FROM observations').fetchone()[0] or -1
        for obs in new:
            if (obs.sequence_id != self.sequence_id or obs.node.arrival_us > decision_us
                    or obs.node.arrival_us < max(previous_arrival, self.meta['decision_us'])):
                raise ValueError('future, withheld, reordered or cross-sequence observation')
            previous_arrival = obs.node.arrival_us
        old_n, old_active, old_decision = self.n, set(self.active), self.meta['decision_us']
        validated = tuple(self._validate_row(old_n+i, row) for i, row in enumerate(rows))
        if len({i for i, _ in rescored}) != len(rescored):
            raise ValueError('duplicate rescored row')
        updates = []
        for i, row in rescored:
            if type(i) is not int or not 0 <= i < old_n:
                raise ValueError('rescore index outside old observations')
            updates.append((i, self._validate_row(i, row, rescore=True)))
        deliveries = ()
        if cache_ingestion is not None:
            from .forest_cache_stream import CacheDelivery
            if (type(cache_ingestion) is not dict or cache_ingestion.get('kind') != 'persistent_cache_ingestion_v1'
                    or not isinstance(cache_ingestion.get('configuration_sha256'), str)):
                raise ValueError('invalid persistent cache ingestion audit')
            deliveries = tuple(CacheDelivery(**d) for d in cache_ingestion['new_deliveries'])
            receipt_map = {(d.side, d.frame_id): d for d in deliveries}
            if len(receipt_map) != len(deliveries) or any(d.sequence_id != self.sequence_id
                    or d.arrival_us > decision_us or d.arrival_us < self.meta['decision_us'] for d in deliveries):
                raise ValueError('invalid cache receipt cohort or time')
            for obs in new:
                key = ('vehicle-side' if obs.node.source_id == 0 else 'infrastructure-side', obs.node.frame_id)
                d = receipt_map.get(key)
                if d is None or (d.arrival_us, d.frame_sha256) != (obs.node.arrival_us, obs.source_cache_sha256):
                    raise ValueError('new raw observation not bound to new cache receipt')
        bindings = {'cache_binding': cache_ingestion['configuration_sha256'] if cache_ingestion is not None else None,
                    'scorer_binding': scorer_binding}
        for key, value in bindings.items():
            existing = self.db.execute('SELECT v FROM meta WHERE k=?', (key,)).fetchone()
            if existing is not None and json.loads(existing[0]) != value:
                raise ValueError('persistent cache/scoring protocol changed')
        self.state_updates = 0
        self.db.execute('BEGIN IMMEDIATE')
        try:
            for key, value in bindings.items():
                if value is not None:
                    self._set(key, value)
            for d in deliveries:
                self.db.execute('INSERT INTO cache_receipts VALUES(?,?,?,?)',
                                (d.side, d.frame_id, d.arrival_us, d.frame_sha256))
            for i, (obs, row) in enumerate(zip(new, validated), old_n):
                raw = canonical(asdict(obs))
                self.db.execute('INSERT INTO observations VALUES(?,?,?,?,?,?,?,?,?,?)',
                    (i, obs.node.node_id, obs.node.source_id, obs.node.frame_id, obs.detection_index,
                     obs.state_us, obs.node.arrival_us, obs.score, raw, hashlib.sha256(raw).hexdigest()))
                self.db.executemany('INSERT INTO potentials VALUES(?,?,?)', ((i, p, w) for p, w in row))
            for i, row in updates:
                self.db.executemany('UPDATE potentials SET w=? WHERE i=? AND p=?', ((w, i, p) for p, w in row))
            self.n += len(new)
            self.revision += bool(updates)
            self.row_cache.clear()
            self._bounds()
            fallback = self.meta['output']
            for _ in new:
                fallback = self._child(fallback, -1)
            restarted = False
            if new:
                self.frontier.update(self.active)
                self.active.clear()
                if len(self.frontier) > self.config.state.max_frontier:
                    self.frontier = {0}
                    restarted = True
            self._promote(decision_us)
            expansions, limited = self._refine(decision_us)
            active, weights, retained, upper, eta = self._mass()
            if decision_indices is None:
                indices = tuple(i for i, in self.db.execute('SELECT i FROM observations WHERE state_us>=? ORDER BY i',
                    (max(0, reference_us-self.config.state.window_us),)))
            else:
                indices = tuple(decision_indices)
            if (len(indices) > self.config.max_decision_nodes or len(set(indices)) != len(indices)
                    or any(type(i) is not int or not 0 <= i < self.n for i in indices)):
                raise ValueError('invalid identity-loss scope or decision-node cap exceeded')
            chosen, decision = self._decode(active, weights, retained, eta, indices, fallback)
            branch_states = []
            predictions = None
            for h in sorted(set(active) | {chosen}):
                states = self._predict(h, reference_us)
                branch_states.append(dict(handle=h, sha256=self._prefix(h).sha256, state_sha256=digest(states)))
                if h == chosen:
                    predictions = states
            restored = []
            if new and old_n and old_active:
                for h in active:
                    ancestor = self.ancestor(h, old_n)
                    if ancestor not in old_active:
                        known = self.db.execute('SELECT decision_us FROM discovered WHERE h=?', (ancestor,)).fetchone()
                        restored.append(dict(ancestor_handle=ancestor, ancestor_sha256=self._prefix(ancestor).sha256,
                            previously_discovered=known is not None and known[0] <= old_decision,
                            was_previous_output=ancestor == self.meta['output']))
            prediction = dict(sequence_id=self.sequence_id, frame_id=frame_id,
                box_reference_timestamp_us=reference_us, decision_timestamp_us=decision_us,
                coordinate_frame='world', state_layout='gravity_xyz_length_width_height_yaw_vxy',
                predictions=predictions, previous_commit_sha256=self.meta['prediction_sha256'])
            prediction['commit_sha256'] = digest(prediction)
            audit = dict(kind='persistent_recoverable_forest_v1', sequence_id=self.sequence_id,
                event_id=event_id, configuration_sha256=digest(asdict(self.config)),
                factor_rows_sha256=self.factor_digest(), observation_count=self.n,
                new_observations=len(new), appended_rows=validated, rescored_rows=updates,
                active=[dict(handle=h, sha256=self._prefix(h).sha256, log_weight=w) for h, w in zip(active, weights)],
                frontier=[dict(handle=h, sha256=self._prefix(h).sha256, depth=self._prefix(h).depth,
                               log_upper=self._upper(h)) for h in sorted(self.frontier)],
                log_partition_upper=upper, eta_upper=eta, decision_indices=indices, decision=decision,
                output_handle=chosen, output_sha256=self._prefix(chosen).sha256,
                restored_ancestors=restored, branches=branch_states, state_updates=self.state_updates,
                expansions=expansions, resource_limited=limited, frontier_restarted_from_root=restarted,
                prefix_node_count=self.prefix_count, prefix_cache_entries=len(self.cache),
                row_cache_entries=len(self.row_cache), row_bound_metadata_bytes=self.suffix.nbytes+self.suffix_abs.nbytes,
                raw_history_retained_on_disk=True, historical_identity_map_compressed=False,
                component_allocation_integrated=False, scorer_binding=scorer_binding, cache_ingestion=cache_ingestion,
                formal_numeric_certificate=False, true_posterior_or_metric_bound=False,
                previous_audit_sha256=self.meta['audit_sha256'], prediction_sha256=prediction['commit_sha256'])
            audit = self._finalize_audit(audit)
            result = PersistentForestCommit(canonical(prediction), canonical(audit))
            self.db.execute('INSERT INTO events VALUES(?,?,?,?,?)',
                (self.meta['events'], event_id, request, result.prediction_json, result.audit_json))
            self.meta = dict(n=self.n, revision=self.revision, frontier=sorted(self.frontier), active=sorted(self.active),
                output=chosen, decision_us=decision_us, reference_us=reference_us, events=self.meta['events']+1,
                prediction_sha256=prediction['commit_sha256'], audit_sha256=digest(audit))
            self._set('state', self.meta)
            self.db.execute('COMMIT')
            return result
        except BaseException:
            self.db.execute('ROLLBACK')
            self._load()
            raise
