"""Qualify the new search oracle using an existing small, immutable output.

Copied SQLite mutations are local software controls, not new experiment runs.
All controls must reject; original artifacts and running auditors are untouched.
"""
import argparse
import datetime
import itertools
import json
import math
from pathlib import Path
import random
import shutil
import sqlite3
import tempfile
from types import SimpleNamespace

import independent_fixed_top1_search as oracle
from rbf_nested_seen_val_v2_common import R, new, register, sha

ADMISSION = R/'artifacts/rbf-original-joint-fixed-top1-GPU4-v1-readback-20261002/topk/seed2027/independent-byte-coverage-factor-admission.json'
DB = ADMISSION.parent/'rank3-unpack/rank-3/0087/sequence-0.sqlite'
DB_SHA = 'f43a3a003c238b7fa5a2d44331bf4ffe851577fd89eed420528f4341615fc521'
PRODUCTION = R/'artifacts/rbf-original-joint-coupled-forest-GPU4-v3-source-20261001/independent-cloud-readback/source.bytes'
PRODUCTION_SHA = '038fa8118c9540d91073fbb8bf594fb6abefe8347f13bb69dfb006d27fcfda03'


def mathematical_controls():
    root = oracle.load_root_oracle(); rng = random.Random(3407); products = 0
    for width in (1, 2, 3, 4):
        for trial in range(15):
            groups = [(c, [(rng.choice((0., -0.25, -2., -1000.)), h)
                            for h in range(1, rng.randint(1, width)+1)]) for c in range(1, 5)]
            got, _, _, evaluations = oracle.cartesian_topk(groups, width)
            # Full global Cartesian enumeration, independent of stage pruning.
            exhaustive = []
            for options in itertools.product(*(options for _, options in groups)):
                score = 0.
                for weight, _ in options: score = math.fsum((score, weight))
                selection = tuple((c, choice[1]) for (c, _), choice in zip(groups, options))
                exhaustive.append((score, selection))
            assert got == sorted(exhaustive, key=lambda v: (-v[0], v[1]))[:width]
            assert evaluations >= len(got)
            products += 1
    got, complete, stages, evaluations = oracle.cartesian_topk(
        [(1, [(0., 1), (0., 2)]), (2, [(0., 1), (0., 2)])], 2)
    assert got == [(0., ((1, 1), (2, 1))), (0., ((1, 1), (2, 2)))]
    assert not complete and evaluations == 6 and stages[-1]['dropped_at_stage'] == 2
    factors = [[(-1, 0.)], [(-1, 0.)], [(-1, 0.), (0, -1.)],
               [(-1, 0.), (1, -1.), (2, -1.)], [(-1, 0.)], [(-1, 0.), (4, -1.)]]
    assert oracle.partition_plans(2, 6, factors, {0: 1, 1: 2}) == [((1, 2), (2, 3)), ((), (4, 5))]
    assert oracle.partition_plans(2, 2, factors, {0: 1, 1: 2}) == []
    assert oracle.partition_plans(2, 3, factors, {0: 1, 1: 2}) == [((1,), (2,))]
    branch = oracle.Branch(3, (-1, 0, -1), (0, 0, 2))
    tree = SimpleNamespace(slots=[('v', '0'), ('i', '0'), ('v', '1'), ('i', '1')],
        slot_positions={('i', '1'): [3]}, factors=[[], [], [], [(-1, -2.), (0, -3.), (1, -4.), (2, -1.)]],
        p={3: (3, 2, 3, -1, 2, None, '[]', 'synthetic-prefix')}, weights={3: 0.}, allowed={3: (-1, 0, 2)})
    candidates = oracle.legal_candidates(tree, [branch], 3, root)
    by_choice = {v[3]: (v[0], v[4]) for v in candidates}
    assert set(by_choice) == {-1, 0, 2}
    assert by_choice[-1] == (-2., 3) and by_choice[2] == (-1., 2)
    assert abs(by_choice[0][0] - math.log(math.exp(-3.)+math.exp(-4.))) <= 1e-14
    tree.slots[3] = ('v', '0'); tree.slot_positions = {('v', '0'): [0, 3]}; tree.allowed = {3: (-1, 2)}
    assert {v[3] for v in oracle.legal_candidates(tree, [branch], 3, root)} == {-1, 2}
    return dict(exhaustive_global_product_cases=products, exact_tie_and_frontier_work_case=True,
                graph_partition_cases=3, equivalent_parent_root_class_case=True, same_source_frame_conflict_case=True)


def modify_audit(db, mutation):
    ordinal, blob = db.execute('SELECT ordinal,audit FROM events WHERE ordinal=1').fetchone()
    value = json.loads(blob)
    component = next(s for s in value['components'] if len(s['active']) == 1 and len(s['predecessors']) > 1 and s['pruning'] and s['merge_pruning_stages'])
    mutation(value, component)
    db.execute('UPDATE events SET audit=? WHERE ordinal=?', (json.dumps(value, sort_keys=True).encode(), ordinal))


def audit_mutation(fn):
    return lambda db: modify_audit(db, fn)


def slot_conflict(db):
    meta = {k: json.loads(v) for k, v in db.execute('SELECT k,v FROM meta')}
    audit = oracle.DatabaseAudit(db, oracle.load_root_oracle(), meta)
    for component, in db.execute('SELECT component FROM component_catalog'):
        tree = audit.tree(component)
        for handle, row in tree.p.items():
            if row[2] < 2: continue
            path = []
            while handle:
                path.append(tree.p[handle]); handle = tree.p[handle][1]
            roots = [r[4] for r in reversed(path)]
            for i in range(len(roots)):
                for j in range(i):
                    if roots[i] == roots[j] and tree.slots[i][0] == tree.slots[j][0] and tree.slots[i][1] != tree.slots[j][1]:
                        # Keep the fixture writable under the schema's unique
                        # detection key; the intended corruption is an illegal
                        # shared identity in the same source/frame slot.
                        index = db.execute('SELECT coalesce(max(detection_index),-1)+1 FROM observations WHERE source=? AND frame=?',
                                           (tree.slots[j][0], tree.slots[j][1])).fetchone()[0]
                        db.execute('UPDATE observations SET frame=?,detection_index=? WHERE i=?',
                                   (tree.slots[j][1], index, tree.full_members[i]))
                        return
    raise AssertionError('fixture contains no same-source distinct-frame root chain')


def promote_nonselected_raw_root_class(db):
    meta = {k: json.loads(v) for k, v in db.execute('SELECT k,v FROM meta')}
    audit = oracle.DatabaseAudit(db, oracle.load_root_oracle(), meta)
    for component, in db.execute('SELECT component FROM component_catalog ORDER BY component'):
        tree = audit.tree(component)
        for handle, row in sorted(tree.p.items()):
            if not handle:
                continue
            parent, depth, chosen = row[1], row[2]-1, row[3]
            alternatives = [p for p in tree.allowed[parent] if p != chosen]
            if not alternatives:
                continue
            other = alternatives[0]
            global_i = tree.full_members[depth]
            global_p = -1 if other < 0 else tree.full_members[other]
            # This alters an unchosen root class, leaving the chosen class's
            # raw mass untouched. Its new score must beat the stored K1 path.
            weight = max(w for _, w in tree.factors[depth])+20.
            cursor = db.execute('UPDATE potentials SET w=? WHERE i=? AND p=?', (weight, global_i, global_p))
            assert cursor.rowcount == 1
            return
    raise AssertionError('fixture has no competing legal root class')


def change_meta_width(db):
    value = json.loads(db.execute("SELECT v FROM meta WHERE k='config'").fetchone()[0])
    value['state']['active_limit'] = 4
    db.execute("UPDATE meta SET v=? WHERE k='config'", (json.dumps(value).encode(),))


def controls():
    def modify_weight(a, s): s['active'][0]['log_weight'] += 0.001
    def modify_product(a, s): s['merge_pruning_stages'][0]['kept'] += 1
    def modify_mass(a, s): s['pruning'][0]['generated_prefix_discarded_mass'] = 0.25
    def modify_scope(a, s): s['pruning'][0]['mass_scope'] = 'full_posterior'
    def modify_count(a, s): s['pruning'][0]['generated_classes'] += 1
    def modify_output(a, s):
        s['output_handle'] = 0
        s['output_sha256'] = oracle.load_root_oracle().digest(['persistent-forest-root', '0087'])
    return dict(
        promoted_unchosen_legal_root_class=promote_nonselected_raw_root_class,
        foreign_K4_width=change_meta_width,
        multiple_retained_classes_in_K1=audit_mutation(lambda a, s: s['active'].append(dict(s['active'][0]))),
        archived_predecessor_as_active=audit_mutation(lambda a, s: s['active'][0].update(handle=1)),
        equivalent_parent_mass= audit_mutation(modify_weight),
        generated_prefix_discarded_mass=audit_mutation(modify_mass),
        false_full_posterior_scope=audit_mutation(modify_scope),
        node_candidate_count=audit_mutation(modify_count),
        resurrected_support_complete=audit_mutation(lambda a, s: s.update(full_model_support_still_in_beam=not s['full_model_support_still_in_beam'])),
        merge_retained_product_count=audit_mutation(modify_product),
        predecessor_member_mapping=audit_mutation(lambda a, s: s['predecessors'].reverse()),
        beam_expansion_counter=audit_mutation(lambda a, s: a.update(beam_expansions=a['beam_expansions']+1)),
        merge_frontier_work_counter=audit_mutation(lambda a, s: a.update(merge_score_evaluations=a['merge_score_evaluations']+1)),
        archive_as_chosen_output=audit_mutation(modify_output),
        false_recovery_claim=audit_mutation(lambda a, s: a.update(recovery_enabled=True)),
        component_loss_scope_weight=audit_mutation(lambda a, s: s.update(weight=s['weight']+0.01)),
        stored_prefix_log_weight=lambda db: db.execute('UPDATE pc42_weights SET value=value+0.001 WHERE h=1'),
        catalog_live_owner=lambda db: db.execute('UPDATE component_catalog SET live=1-live WHERE component=1'),
        prefix_content_hash=lambda db: db.execute("UPDATE pc42_prefixes SET sha=? WHERE h=1", ('0'*64,)),
        same_source_frame_identity_conflict=slot_conflict,
    )


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists()
    assert sha(DB) == DB_SHA and sha(PRODUCTION) == PRODUCTION_SHA
    admission = json.loads(ADMISSION.read_bytes())
    entry = next(v for v in admission['sequences'] if v['sequence_id'] == '0087')
    assert entry['database_sha256'] == DB_SHA and entry['events'] == 20 and entry['nodes'] == 414
    args.output.mkdir(parents=True)
    binding = dict(kind='rbf_independent_fixed_Top1_search_oracle_qualification_v1',
        original_task_id='879429b762b74e3b8be42ddcdef34efe', original_database=str(DB), original_database_sha256=DB_SHA,
        original_admission_path=str(ADMISSION), original_admission_sha256=sha(ADMISSION),
        production_archive_path=str(PRODUCTION), production_archive_sha256=sha(PRODUCTION),
        root_oracle_sha256=oracle.ROOT_SHA, oracle_path=str(Path(oracle.__file__).resolve()),
        oracle_sha256=sha(Path(oracle.__file__)), qualification_source_sha256=sha(Path(__file__)),
        software_qualification_only=True, old_experiment_rerun=False,
        new_final_model_experiment_accepted=False, full_online_method_accepted=False,
        same_resource_performance_accepted=False, paper_performance_complete=False)
    new(args.output/'binding.json', binding)
    results = []
    try:
        math_controls = mathematical_controls()
        baseline = oracle.verify_database(DB, DB_SHA)
        assert baseline['beam_width']==1
        new(args.output/'existing-small-sequence-search-only.json', baseline)
        with tempfile.TemporaryDirectory(prefix='rbf-topK-oracle-controls-') as temp:
            for name, mutation in controls().items():
                path = Path(temp)/(name+'.sqlite'); shutil.copyfile(DB, path)
                db = sqlite3.connect(path)
                try: mutation(db); db.commit()
                finally: db.close()
                try:
                    oracle.verify_database(path, sha(path))
                except (AssertionError, KeyError, ValueError) as error:
                    result = dict(control=name, rejected=True, type=type(error).__name__, message=str(error),
                                  mutated_input_sha256=sha(path))
                    new(args.output/(name+'.json'), result); results.append(result)
                else: raise AssertionError('negative control was accepted: '+name)
        assert len(results) == len(controls()) and sha(DB) == DB_SHA
        final = dict(binding, mathematical_controls=math_controls, controls_rejected=len(results),
            qualified_existing_sequence_events=baseline['events'],
            qualified_existing_sequence_observations=baseline['observations'],
            max_abs_error=max(baseline['max_abs_error'].values()), fixed_atol=1e-8, fixed_rtol=1e-8,
            checked_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
        new(args.output/'qualification.json', final); register(args.output/'qualification.json', final['kind'])
        print(json.dumps(final), flush=True)
    except BaseException as error:
        failure = dict(binding, failure_type=type(error).__name__, failure_message=str(error),
                       software_qualified=False, completed_negative_controls=len(results))
        new(args.output/'failure.json', failure); register(args.output/'failure.json', 'rbf_topK_search_oracle_qualification_failure')
        raise


if __name__ == '__main__': main()
