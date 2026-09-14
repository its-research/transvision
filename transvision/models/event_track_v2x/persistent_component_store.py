"""Transactional component maps over shared raw observations and fixed support.

Component-specific inference tables reference the single original raw/factor store through read-only views. A bridge rebuilds the COMPLETE joined support; old prefix tables and
membership maps remain recoverable. This storage layer does not itself select actions or claim end-to-end component tracking.
"""
from __future__ import annotations
import json
import math
import re
from collections import OrderedDict
from dataclasses import asdict, dataclass

from .detection_cache_v2 import canonical
from .hypothesis_bank import logsumexp
from .identity_forest import digest
from .persistent_forest import PersistentForestConfig, PersistentForestTracker

_TABLES = frozenset(('meta', 'observations', 'potentials', 'prefixes', 'weights', 'states', 'discovered', 'events', 'cache_receipts', 'observation_slots', 'observation_time'))
_IDENTIFIERS = re.compile(r'\b(' + ('|'.join(sorted(_TABLES, key=len, reverse=True))) + r')\b')


class _Namespace:
    """Internal kernel SQL only; values stay bound parameters, never SQL text.

    Transaction control belongs to the coordinator. Rejecting it here avoids accidental SQLite executescript commits or independent component commits.
    """

    def __init__(self, connection, component):
        if type(component) is not int or component < 1:
            raise ValueError('positive internal component namespace required')
        self.connection, self.prefix = connection, f'pc{component}_'

    @property
    def in_transaction(self):
        return self.connection.in_transaction

    def _sql(self, sql):
        if re.match(r'\s*(BEGIN|COMMIT|ROLLBACK|SAVEPOINT|RELEASE|PRAGMA|ATTACH|DETACH)\b', sql, re.I):
            raise ValueError('component kernel cannot control shared transaction or connection')
        return _IDENTIFIERS.sub(lambda m: self.prefix + m.group(0), sql)

    def execute(self, sql, parameters=()):
        return self.connection.execute(self._sql(sql), parameters)

    def executemany(self, sql, parameters):
        return self.connection.executemany(self._sql(sql), parameters)


@dataclass(frozen=True)
class ComponentChange:
    component: int
    predecessors: tuple
    added_indices: tuple
    merge_restart: bool


class _RootPartitionKernel(PersistentForestTracker):
    """Exactly sum parent paths that encode the SAME full identity partition.

    For a fixed root-assignment prefix, linking to any candidate parent in root r produces the same identity history and conditional CI state. Its outgoing class weight is the SUM
    of those parent potentials, not their maximum. A canonical minimum parent stores the class; raw factors retain all paths.
    """

    def _choices(self, handle):
        choices = super()._choices(handle)
        groups = {}
        for parent in choices:
            root = -1 if parent < 0 else self._prefix(self.ancestor(handle, parent + 1)).root
            groups.setdefault(root, parent)
        return tuple(sorted(groups.values()))

    def transition_log_weight(self, handle, choice):
        row = self._row(self._prefix(handle).depth)
        if choice < 0:
            return dict(row)[-1]
        root = self._prefix(self.ancestor(handle, choice + 1)).root
        return logsumexp(w for p, w in row if p >= 0 and self._prefix(self.ancestor(handle, p + 1)).root == root)

    def _weight(self, handle):
        pending = []
        while handle:
            row = self.db.execute('SELECT revision,value FROM weights WHERE h=?', (handle, )).fetchone()
            if row is not None and row[0] == self.revision:
                value = row[1]
                break
            pending.append(handle)
            handle = self._prefix(handle).parent_handle
        else:
            value = 0.
        for h in reversed(pending):
            prefix = self._prefix(h)
            transition = self.transition_log_weight(prefix.parent_handle, prefix.choice)
            value = math.fsum([value, transition])
            self.db.execute('INSERT OR REPLACE INTO weights VALUES(?,?,?)', (h, self.revision, value))
        return value

    def _covered(self, handle):
        if 0 in self.frontier:
            return True
        return any(self.ancestor(handle, self._prefix(p).depth) == p for p in self.frontier)

    def _promote(self, decision):
        terminals = {h for h in self.frontier if self._prefix(h).depth == self.n}
        for h in terminals:
            self.db.execute('INSERT OR IGNORE INTO discovered VALUES(?,?)', (h, decision))
        ranked = sorted(self.active | terminals, key=lambda h: (-self._weight(h), self._prefix(h).sha256))
        retained = set(ranked[:self.config.state.active_limit])
        # A seeded complete class may already lie inside an unresolved prefix.
        # Do not insert a second overlapping frontier region when it is evicted.
        for handle in self.active - retained:
            if not self._covered(handle):
                self.frontier.add(handle)
        self.frontier.difference_update(retained)
        self.active = retained

    def seed_complete_action(self, handle, decision_us):
        if self._prefix(handle).depth != self.n:
            raise ValueError('only complete identity classes can seed retained mass')
        self.db.execute('INSERT OR IGNORE INTO discovered VALUES(?,?)', (handle, decision_us))
        self.active.add(handle)
        self._promote(decision_us)

    def _mass(self):
        active = sorted(self.active, key=lambda h: (-self._weight(h), self._prefix(h).sha256))
        weights = [self._weight(h) for h in active]
        retained = logsumexp(weights)
        # Frontier regions are disjoint. Some seeded active classes may already
        # be included in those regions: do not add their mass a second time.
        upper = logsumexp([self._upper(p) for p in self.frontier] + [w for h, w in zip(active, weights) if not self._covered(h)], upper=True)
        eta = 0. if not self.frontier else 1. if not active else max(0., min(1., -math.expm1(min(0., retained - upper))))
        return active, weights, retained, upper, eta

    def _decode(self, active, weights, retained, eta, indices, fallback, *, fallback_on_risk=True, force_fallback=False):
        if self.config.state.decision_mode in {'all-legal-hamming', 'map'}:
            return super()._decode(active, weights, retained, eta, indices, fallback, fallback_on_risk=fallback_on_risk, force_fallback=force_fallback)
        if self.frontier or not active:
            chosen, decision = super()._decode(active, weights, retained, eta, indices, fallback, fallback_on_risk=fallback_on_risk, force_fallback=force_fallback)
            decision['conditional_bayes_lower_kind'] = 'independent_root_relaxation'
            return chosen, decision
        # Empty frontier means every supported legal root class is active. The
        # best enumerated action is then the exact conditional Bayes action for
        # this model/action set, even if independent root minima are infeasible.
        best, optimal = super()._decode(active, weights, retained, 0., indices, fallback, fallback_on_risk=False)
        if force_fallback:
            chosen, decision = super()._decode(active, weights, retained, 0., indices, fallback, force_fallback=True)
        else:
            chosen, decision = best, optimal
        lower = optimal['conditional_risk']
        gap = max(0., decision['conditional_risk'] - lower)
        decision = dict(decision, relaxed_bayes_lower=lower, optimization_gap=gap, risk_bound=min(1., gap), conditional_bayes_lower_kind='complete_supported_action_enumeration')
        return chosen, decision


class PersistentComponentStore:
    """Exact append-only edge components; all mutations require an outer txn.

    Limits count archived membership maps too. A merger may lose enumeration work but never deletes support. Parent supports must already be committed to the shared transaction's
    observations/potentials tables by its owner.
    """

    def __init__(self, owner, *, max_components=4096, max_member_rows=1_000_000):
        if (not isinstance(owner, PersistentForestTracker) or any(type(v) is not int or v < 1 for v in (max_components, max_member_rows))):
            raise ValueError('persistent owner and positive component limits required')
        self.owner, self.db = owner, owner.db
        self.max_components, self.max_member_rows = max_components, max_member_rows

    def initialize(self):
        self._transaction()
        if self.db.execute('SELECT count(*) FROM observations').fetchone()[0]:
            raise ValueError('component schema must be installed before observations arrive')
        self.db.execute('CREATE TABLE component_catalog(component INTEGER PRIMARY KEY, live INTEGER NOT NULL, '
                        'predecessors BLOB NOT NULL, created_us INTEGER NOT NULL)')
        self.db.execute('CREATE TABLE component_members(component INTEGER NOT NULL, local_i INTEGER NOT NULL, '
                        'global_i INTEGER NOT NULL, PRIMARY KEY(component,local_i), UNIQUE(component,global_i))')
        self.db.execute('CREATE TABLE component_owners(global_i INTEGER PRIMARY KEY, component INTEGER NOT NULL)')
        self.db.execute('CREATE TABLE component_control(k TEXT PRIMARY KEY, value BLOB NOT NULL)')
        self.db.executemany('INSERT INTO component_control VALUES(?,?)', [('limits', canonical([self.max_components, self.max_member_rows])), ('partition_n', canonical(0))])

    def _transaction(self):
        if not self.db.in_transaction:
            raise ValueError('shared transaction required for component mutations')

    def _check_limits(self):
        row = self.db.execute("SELECT value FROM component_control WHERE k='limits'").fetchone()
        if row is None or json.loads(row[0]) != [self.max_components, self.max_member_rows]:
            raise ValueError('persistent component resource limits changed')

    def members(self, component):
        return tuple(i for i, in self.db.execute('SELECT global_i FROM component_members WHERE component=? ORDER BY local_i', (component, )))

    def live(self):
        return tuple(c for c, in self.db.execute('SELECT component FROM component_catalog WHERE live=1 ORDER BY component'))

    def append_partition(self, old_n, *, decision_us):
        self._transaction()
        self._check_limits()
        recorded = json.loads(self.db.execute("SELECT value FROM component_control WHERE k='partition_n'").fetchone()[0])
        count = self.db.execute('SELECT count(*) FROM observations').fetchone()[0]
        if (type(old_n) is not int or old_n != recorded or old_n > count or type(decision_us) is not int or decision_us < 0):
            raise ValueError('partition append differs from previous raw history')
        if count and self.db.execute('SELECT min(i),max(i) FROM observations').fetchone() != (0, count - 1):
            raise ValueError('shared raw indices are not contiguous')
        latest_arrival = self.db.execute('SELECT max(arrival_us) FROM observations').fetchone()[0]
        if latest_arrival is not None and latest_arrival > decision_us:
            raise ValueError('future raw observation in component store')
        parents = {}

        def root(key):
            parents.setdefault(key, key)
            while parents[key] != key:
                parents[key] = parents[parents[key]]
                key = parents[key]
            return key

        for i in range(old_n, count):
            root(('new', i))
            row = tuple(self.db.execute('SELECT p,w FROM potentials WHERE i=? ORDER BY p', (i, )))
            self.owner._validate_row(i, row)
            for p, _ in row:
                if p < 0:
                    continue
                if p < old_n:
                    found = self.db.execute('SELECT component FROM component_owners WHERE global_i=?', (p, )).fetchone()
                    if found is None:
                        raise ValueError('old raw observation lost its component owner')
                    key = ('old', found[0])
                else:
                    key = ('new', p)
                a, b = root(('new', i)), root(key)
                parents[max(a, b)] = min(a, b)
        groups = {}
        for key in tuple(parents):
            groups.setdefault(root(key), []).append(key)
        plans = []
        for group in groups.values():
            prior = tuple(sorted(v for kind, v in group if kind == 'old'))
            added = tuple(sorted(v for kind, v in group if kind == 'new'))
            plans.append((prior, added))
        plans.sort(key=lambda p: p[1][0])
        old_live = len(self.live())
        new_live = old_live + sum(1 - len(prior) for prior, _ in plans)
        # Existing one-predecessor maps append only. New/merged maps include all
        # predecessor rows; archived maps remain counted and are never deleted.
        extra = 0
        for prior, added in plans:
            extra += len(added)
            if len(prior) > 1:
                extra += sum(self.db.execute('SELECT count(*) FROM component_members WHERE component=?', (c, )).fetchone()[0] for c in prior)
        if (new_live > self.max_components or self.db.execute('SELECT count(*) FROM component_members').fetchone()[0] + extra > self.max_member_rows):
            raise ValueError('component/map capacity exhausted without dropping support')
        changes = []
        for prior, added in plans:
            if len(prior) == 1:
                component = prior[0]
                start = self.db.execute('SELECT count(*) FROM component_members WHERE component=?', (component, )).fetchone()[0]
                self.db.executemany('INSERT INTO component_members VALUES(?,?,?)', ((component, start + offset, i) for offset, i in enumerate(added)))
            else:
                component = self.db.execute('INSERT INTO component_catalog(live,predecessors,created_us) VALUES(1,?,?)', (canonical(prior), decision_us)).lastrowid
                if prior:
                    placeholders = ','.join('?' for _ in prior)
                    indices = (i for i, in self.db.execute('SELECT global_i FROM component_members WHERE component IN (' + placeholders + ') ORDER BY global_i', prior))
                    self.db.executemany('INSERT INTO component_members VALUES(?,?,?)', ((component, local, i) for local, i in enumerate(indices)))
                    self.db.executemany('UPDATE component_catalog SET live=0 WHERE component=?', ((c, ) for c in prior))
                start = self.db.execute('SELECT count(*) FROM component_members WHERE component=?', (component, )).fetchone()[0]
                self.db.executemany('INSERT INTO component_members VALUES(?,?,?)', ((component, start + offset, i) for offset, i in enumerate(added)))
            self.db.execute('INSERT OR REPLACE INTO component_owners SELECT global_i,component FROM component_members WHERE component=?', (component, ))
            changes.append(ComponentChange(component, prior, added, len(prior) > 1))
        self.db.execute("UPDATE component_control SET value=? WHERE k='partition_n'", (canonical(count), ))
        self.validate()
        return tuple(changes)

    def validate(self):
        count = self.db.execute('SELECT count(*) FROM observations').fetchone()[0]
        if self.db.execute('SELECT count(*) FROM component_owners').fetchone()[0] != count:
            raise ValueError('component ownership is incomplete')
        if self.db.execute('''SELECT 1 FROM component_owners o LEFT JOIN component_catalog c ON c.component=o.component
                LEFT JOIN component_members m ON m.component=o.component AND m.global_i=o.global_i
                LEFT JOIN observations raw ON raw.i=o.global_i
                WHERE c.live IS NOT 1 OR m.global_i IS NULL OR raw.i IS NULL LIMIT 1''').fetchone():
            raise ValueError('invalid live component ownership')
        if self.db.execute('''SELECT 1 FROM potentials p JOIN component_owners a ON a.global_i=p.i
                JOIN component_owners b ON b.global_i=p.p WHERE p.p>=0 AND a.component!=b.component LIMIT 1''').fetchone():
            raise ValueError('candidate edge crosses supposedly independent components')
        if self.db.execute('''SELECT 1 FROM component_members GROUP BY component
                HAVING min(local_i)!=0 OR max(local_i)!=count(*)-1 LIMIT 1''').fetchone():
            raise ValueError('noncontiguous local component order')

    def create_kernel(self, component, config, *, caches=None):
        """Create shared-raw views and empty-prefix inference state in this
        txn."""
        self._transaction()
        if type(config) is not PersistentForestConfig or component not in self.live():
            raise ValueError('live component and validated monolithic kernel config required')
        prefix = _Namespace(self.db, component).prefix
        # The raw table is global; its payload is NOT copied during a merge.
        self.db.execute(f'''CREATE VIEW {prefix}observations AS SELECT m.local_i AS i,o.node_id,o.source,o.frame,
            o.detection_index,o.state_us,o.arrival_us,o.score,o.raw,o.sha FROM component_members m
            JOIN observations o ON o.i=m.global_i WHERE m.component={component}''')
        self.db.execute(f'''CREATE VIEW {prefix}potentials AS SELECT a.local_i AS i,
            CASE WHEN p.p=-1 THEN -1 ELSE b.local_i END AS p,p.w FROM component_members a
            JOIN potentials p ON p.i=a.global_i LEFT JOIN component_members b
            ON b.component=a.component AND b.global_i=p.p WHERE a.component={component}''')
        sql = _Namespace(self.db, component)
        for statement in (
                'CREATE TABLE meta(k TEXT PRIMARY KEY,v BLOB NOT NULL)',
                'CREATE TABLE prefixes(h INTEGER PRIMARY KEY,parent INTEGER,depth INTEGER NOT NULL,choice INTEGER,root INTEGER,'
                'previous_root INTEGER,jumps BLOB NOT NULL,sha TEXT NOT NULL,UNIQUE(parent,choice))',
                'CREATE TABLE weights(h INTEGER PRIMARY KEY,revision INTEGER NOT NULL,value REAL NOT NULL)',
                'CREATE TABLE states(h INTEGER PRIMARY KEY,payload BLOB NOT NULL)',
                'CREATE TABLE discovered(h INTEGER PRIMARY KEY,decision_us INTEGER NOT NULL)',
        ):
            sql.execute(statement)
        sql.execute('INSERT INTO prefixes VALUES(0,NULL,0,NULL,NULL,NULL,?,?)', (canonical([]), digest(['persistent-forest-root', self.owner.sequence_id])))
        sql.executemany(
            'INSERT INTO meta VALUES(?,?)',
            [('config', canonical(asdict(config))), ('representation', canonical('exact_root_partition_classes_with_seeded_lower_mass_v1')),
             ('state',
              canonical(dict(n=0, revision=0, frontier=[0], active=[], output=0, decision_us=-1, reference_us=-1, events=0, prediction_sha256='0' * 64, audit_sha256='0' * 64)))])
        return self.open_kernel(component, config, caches=caches)

    def open_kernel(self, component, config, *, caches=None):
        if type(config) is not PersistentForestConfig:
            raise TypeError('validated kernel configuration required')
        sql = _Namespace(self.db, component)
        stored = sql.execute("SELECT v FROM meta WHERE k='config'").fetchone()
        representation = sql.execute("SELECT v FROM meta WHERE k='representation'").fetchone()
        if (stored is None or json.loads(stored[0]) != asdict(config) or representation is None
                or json.loads(representation[0]) != 'exact_root_partition_classes_with_seeded_lower_mass_v1'):
            raise ValueError('component kernel configuration differs')
        kernel = _RootPartitionKernel.__new__(_RootPartitionKernel)
        kernel.path, kernel.sequence_id, kernel.config = self.owner.path, self.owner.sequence_id, config
        kernel.db = sql
        kernel.cache, kernel.row_cache = (OrderedDict(), OrderedDict()) if caches is None else caches
        kernel.configuration_sha256 = digest(asdict(config))
        kernel._load()
        return kernel

    def merged_fallback(self, component, predecessors):
        """Lift previous complete outputs, not their averaged states or
        Top-K."""
        members = self.members(component)
        inverse = {global_i: local for local, global_i in enumerate(members)}
        parents = [-1] * len(members)
        for previous in predecessors:
            old_members = self.members(previous)
            sql = _Namespace(self.db, previous)
            meta = json.loads(sql.execute("SELECT v FROM meta WHERE k='state'").fetchone()[0])
            handle = meta['output']
            if meta['n'] != len(old_members):
                raise ValueError('predecessor output does not cover its complete raw history')
            depth = len(old_members)
            while handle:
                old = sql.execute('SELECT parent,depth,choice FROM prefixes WHERE h=?', (handle, )).fetchone()
                if old is None or old[1] != depth:
                    raise ValueError('incomplete predecessor output prefix')
                parent, local, choice = old[0], old[1] - 1, old[2]
                parents[inverse[old_members[local]]] = -1 if choice < 0 else inverse[old_members[choice]]
                handle, depth = parent, depth - 1
            if depth:
                raise ValueError('predecessor output lost identity history')
        return tuple(parents)

    def activate_kernel(self, kernel, *, decision_us, rescore=False, fallback_parents=None):
        """Refresh a component from shared views before shared-budget
        inference."""
        self._transaction()
        if not isinstance(kernel.db, _Namespace) or kernel.db.connection is not self.db:
            raise ValueError('kernel must belong to this shared transaction')
        if type(rescore) is not bool or type(decision_us) is not int or decision_us < kernel.meta['decision_us']:
            raise ValueError('invalid component rescore/decision')
        old_n, old_active = kernel.n, set(kernel.active)
        count, arrival = kernel.db.execute('SELECT count(*),max(arrival_us) FROM observations').fetchone()
        if (count < old_n or count > kernel.config.max_observations or arrival is not None and arrival > decision_us):
            raise ValueError('component history shrank, exceeded capacity, or includes future input')
        kernel.n = count
        kernel.revision += rescore
        if count != old_n or rescore:
            kernel.row_cache.clear()
            kernel._bounds()
        kernel.state_updates = 0
        if fallback_parents is None:
            fallback = kernel.meta['output']
            if kernel._prefix(fallback).depth != old_n:
                raise ValueError('previous component output has incorrect depth')
            for _ in range(count - old_n):
                fallback = kernel._child(fallback, -1)
        else:
            if old_n or len(fallback_parents) != count:
                raise ValueError('explicit joined fallback is only for a new complete component')
            fallback = 0
            for choice in fallback_parents:
                if type(choice) is not int or choice not in kernel._choices(fallback):
                    raise ValueError('joined predecessor output is not a legal component action')
                fallback = kernel._child(fallback, choice)
        restarted = False
        if count > old_n:
            for handle in kernel.active:
                if not kernel._covered(handle):
                    kernel.frontier.add(handle)
            kernel.active.clear()
            if len(kernel.frontier) > kernel.config.state.max_frontier:
                kernel.frontier, restarted = {0}, True
        kernel._promote(decision_us)
        return dict(old_n=old_n, old_active=old_active, fallback=fallback, frontier_restarted_from_root=restarted, old_decision_us=kernel.meta['decision_us'])

    def save_kernel(self, kernel, *, chosen, reference_us, decision_us):
        self._transaction()
        if (not isinstance(kernel.db, _Namespace) or kernel.db.connection is not self.db or kernel._prefix(chosen).depth != kernel.n or type(reference_us) is not int
                or type(decision_us) is not int or not 0 <= reference_us <= decision_us or reference_us <= kernel.meta['reference_us'] or decision_us < kernel.meta['decision_us']):
            raise ValueError('invalid component output or nonmonotonic decision')
        kernel.meta = dict(
            kernel.meta,
            n=kernel.n,
            revision=kernel.revision,
            frontier=sorted(kernel.frontier),
            active=sorted(kernel.active),
            output=chosen,
            decision_us=decision_us,
            reference_us=reference_us,
            events=kernel.meta['events'] + 1)
        kernel._set('state', kernel.meta)

    def output_recoveries(self, component, kernel, chosen, context, change):
        """Audit actual output classes recovered outside predecessor retained
        sets.

        Projection is reconstructed from current prefix + immutable member maps; complete parent histories are not duplicated into every audit record.
        """
        if change is not None and change.merge_restart:
            predecessors = change.predecessors
        elif context['old_n']:
            predecessors = (component, )
        else:
            return []
        if change is not None and change.merge_restart:
            current_members = self.members(component)
            current_local = {i: local for local, i in enumerate(current_members)}
        events = []
        for previous in predecessors:
            sql = _Namespace(self.db, previous)
            if previous == component:
                old_n = context['old_n']
                handle = kernel.ancestor(chosen, old_n)
                previous_meta = kernel.meta
                known_active = context['old_active']
                projection_sha = kernel._prefix(handle).sha256
                projection_nodes = old_n
            else:
                previous_meta = json.loads(sql.execute("SELECT v FROM meta WHERE k='state'").fetchone()[0])
                known_active = set(previous_meta['active'])
                old_members = self.members(previous)
                old_local = {i: local for local, i in enumerate(old_members)}
                parents = []
                for global_i in old_members:
                    prefix = kernel._prefix(kernel.ancestor(chosen, current_local[global_i] + 1))
                    parents.append(-1 if prefix.choice < 0 else old_local[current_members[prefix.choice]])
                projection_sha = digest(['root_partition_predecessor_projection_v1', previous, parents])
                projection_nodes = len(parents)
                handle = 0
                for choice in parents:
                    found = sql.execute('SELECT h FROM prefixes WHERE parent=? AND choice=?', (handle, choice)).fetchone()
                    if found is None:
                        handle = None
                        break
                    handle = found[0]
            if handle in known_active or handle == previous_meta['output']:
                continue
            discovered = None if handle is None else sql.execute('SELECT decision_us FROM discovered WHERE h=?', (handle, )).fetchone()
            events.append(
                dict(
                    previous_component=previous,
                    current_component=component,
                    output_handle=chosen,
                    predecessor_handle=handle,
                    projection_sha256=projection_sha,
                    previously_discovered=discovered is not None and discovered[0] <= previous_meta['decision_us'],
                    previously_unexpanded=discovered is None or discovered[0] > previous_meta['decision_us'],
                    projection_nodes=projection_nodes,
                    projection_recipe='restrict_current_class_to_predecessor_member_map',
                    scope='actual_output_class_outside_previous_active_and_output'))
        return events
