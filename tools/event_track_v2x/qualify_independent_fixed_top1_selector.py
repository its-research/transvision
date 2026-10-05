"""Software-only selector qualification; no prior search experiment rerun."""
import argparse
import copy
import datetime
import json
import math
from pathlib import Path
import shutil
import sqlite3
import tempfile

import independent_fixed_top1_selector as oracle
from rbf_nested_seen_val_v2_common import R, new, register, sha

DB = R/'artifacts/rbf-original-joint-fixed-top1-GPU4-v1-readback-20261002/topk/seed2027/rank3-unpack/rank-3/0087/sequence-0.sqlite'
DB_SHA = 'f43a3a003c238b7fa5a2d44331bf4ffe851577fd89eed420528f4341615fc521'
SEARCH = R/'artifacts/rbf-independent-fixed-Top1-search-pruning-qualification-v1-20261004/existing-small-sequence-search-only.json'


def mathematical_controls():
    def check(roots, probabilities, shas, complete=False, eta=.2):
        weights = [math.log(p) for p in probabilities]
        chosen, decision, risks = oracle.normative_float64(roots, weights, 0., shas, complete=complete, eta=eta)
        pairwise, _ = oracle.pairwise_risks(roots, weights)
        assert max(abs(a-b) for a, b in zip(risks, pairwise)) < 1e-12
        return chosen, decision, risks
    chosen, decision, risks = check([(0,0), (0,1), (1,1)], [.4,.35,.25], ['a','b','c'])
    assert chosen == 1 and all(abs(a-b) < 1e-14 for a,b in zip(risks, (.425,.325,.575)))
    assert abs(decision['risk_bound']-.2) < 1e-14
    chosen, decision, _ = check([(0,0), (0,1), (1,1)], [.4,.35,.25], ['a','b','c'], True)
    assert chosen == 1 and decision['optimization_gap'] == decision['risk_bound'] == 0.
    roots = [(0,0,1), (0,1,0), (1,0,0)]
    _, partial, _ = check(roots, [.34,.33,.33], ['a','b','c'])
    _, complete, _ = check(roots, [.34,.33,.33], ['a','b','c'], True)
    assert partial['optimization_gap'] > .1 and complete['optimization_gap'] == 0.
    assert complete['relaxed_bayes_lower'] > partial['relaxed_bayes_lower']
    chosen, _, _ = check([(0,), (1,)], [.5,.5], ['b','a'])
    assert chosen == 1, 'exact tie must use SHA ordering'
    chosen, _, _ = check([(0,), (1,)], [.5-1e-13,.5+1e-13], ['a','b'])
    assert chosen == 1, 'near tie may not be collapsed by numerical tolerance'
    chosen, empty, _ = check([(),()], [.6,.4], ['b','a'])
    assert chosen == 0 and empty['empty_loss_scope'] and empty['risk_bound'] == 0.
    r, p = oracle.pairwise_risks([(0,), (1,), (2,)], [0.,-1000.,-2000.])
    assert r == (0.,1.,1.) and p == (1.,0.,0.)
    weights = [-1234., -1234.7]
    retained = max(weights)+math.log(math.fsum(math.exp(w-max(weights)) for w in weights))
    _, _, risks = oracle.normative_float64([(0,), (1,)], weights, retained, ['a','b'], complete=False, eta=.2)
    pairwise, _ = oracle.pairwise_risks([(0,), (1,)], weights)
    assert max(abs(a-b) for a,b in zip(risks,pairwise)) < 1e-12
    chosen, single, risks = check([(0,1)], [1.], ['a'], False, .2)
    assert chosen==0 and risks==(0.,) and single['conditional_risk']==single['optimization_gap']==0. and single['risk_bound']==.2
    _, complete_single, _ = check([(0,1)], [1.], ['a'], True)
    assert complete_single['risk_bound']==0.
    return dict(K1_single_supported_action_case=True, generic_multi_action_MAP_not_Bayes_example=True, complete_vs_relaxed_lower_example=True,
                infeasible_independent_root_minima_example=True, exact_SHA_tie_example=True,
                near_tie_not_collapsed_by_tolerance=True, empty_scope_uses_first_active_example=True,
                large_log_weight_and_underflow_examples=True, decimal_precision=70)


def mutate_selected(db, name):
    for ordinal, blob in db.execute('SELECT ordinal,audit FROM events ORDER BY ordinal'):
        value = json.loads(blob)
        for s in value['components']:
            if len(s['active']) != 1 or len(s['decision_indices']) < 2:
                continue
            if name == 'nonretained_output':
                digest = db.execute(f"SELECT sha FROM pc{s['component']}_prefixes WHERE h=0").fetchone()[0]
                s['output_handle'], s['output_sha256'] = 0, digest
            elif name == 'output_SHA': s['output_sha256'] = '0'*64
            elif name in ('conditional_risk','relaxed_bayes_lower','optimization_gap','risk_bound'):
                s['decision'][name] += .001
            elif name == 'wrong_complete_lower_kind': s['decision']['conditional_bayes_lower_kind'] = 'full_posterior_certified'
            elif name == 'empty_scope_flag': s['decision']['empty_loss_scope'] = not s['decision']['empty_loss_scope']
            elif name == 'global_model_regret': value['model_regret_upper'] += .001
            elif name == 'regret_fallback': value['global_fallback_used'] = True
            elif name == 'MAP_policy_label': value['output_policy'] = 'retained_posterior_MAP'
            elif name == 'query_scope': s['decision_indices'].reverse()
            elif name == 'multiple_active_classes': s['active'].append(dict(s['active'][0]))
            elif name == 'bad_conditional_normalizer': s['log_retained'] += .001
            else: raise AssertionError(name)
            db.execute('UPDATE events SET audit=? WHERE ordinal=?', (json.dumps(value,sort_keys=True).encode(), ordinal))
            return
    raise AssertionError('no applicable K1 decision mutation: '+name)


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(); assert not args.output.exists()
    assert sha(DB) == DB_SHA
    search_proof = json.loads(SEARCH.read_bytes())
    args.output.mkdir(parents=True)
    sources = args.output/'candidate-sources'; sources.mkdir()
    for path in (Path(oracle.__file__), Path(__file__)):
        shutil.copyfile(path, sources/path.name)
    binding = dict(kind='rbf_independent_fixed_Top1_conditional_selector_qualification_v1',
        database_path=str(DB), database_sha256=DB_SHA,
        search_proof_path=str(SEARCH), search_proof_sha256=sha(SEARCH),
        oracle_sha256=sha(oracle.__file__), qualification_source_sha256=sha(__file__),
        source_snapshots={p.name:sha(p) for p in sources.iterdir()},
        software_qualification_only=True, search_pruning_rerun=False,
        old_experiment_rerun=False, new_final_model_experiment_accepted=False,
        full_online_method_accepted=False, paper_performance_complete=False)
    new(args.output/'binding.json', binding)
    negatives = []
    try:
        math_controls = mathematical_controls()
        baseline = oracle.verify_database(DB, DB_SHA, search_proof)
        assert baseline['events'] == 20 and baseline['component_decisions'] == 229
        assert baseline['posterior_MAP_and_output_differ'] == 0 and baseline['scoped_identity_queries'] == 4794
        new(args.output/'existing-small-sequence-selector-only.json', baseline)
        names = ('nonretained_output','output_SHA','conditional_risk',
            'relaxed_bayes_lower','optimization_gap','risk_bound','wrong_complete_lower_kind',
            'empty_scope_flag','global_model_regret','regret_fallback','MAP_policy_label',
            'query_scope','multiple_active_classes','bad_conditional_normalizer')
        with tempfile.TemporaryDirectory(prefix='rbf-topK-selector-') as directory:
            for name in names:
                path = Path(directory)/(name+'.sqlite'); shutil.copyfile(DB,path)
                db = sqlite3.connect(path)
                try: mutate_selected(db,name); db.commit()
                finally: db.close()
                # Synthetic upstream digest only isolates selector logic in a
                # software negative control. No real search proof is altered.
                hypothetical = dict(search_proof,database_sha256=sha(path))
                try: oracle.verify_database(path,sha(path),hypothetical)
                except AssertionError as error:
                    result = dict(control=name,rejected=True,message=str(error),mutated_input_sha256=sha(path),
                        upstream_search_digest_rewritten_for_local_negative_fixture_only=True,
                        real_search_admission_or_new_model_acceptance=False)
                    new(args.output/(name+'.json'),result); negatives.append(result)
                else: raise AssertionError('negative selector control accepted: '+name)
        edits = {
            'foreign_K4_search_kind': lambda v:v.update(kind='rbf_independent_fixed_topK_raw_graph_search_pruning_audit_v1'),
            'foreign_K4_beam_width': lambda v:v.update(beam_width=4),
            'foreign_byte_identity': lambda v:v.update(database_sha256='a'*64),
            'foreign_sequence_identity': lambda v:v.update(sequence_id='other'),
            'partial_search_events': lambda v:v.update(events=19),
            'missing_root_search_gate': lambda v:v.update(all_node_local_legal_root_class_Top1_choices_verified=False),
            'missing_prefix_weight_gate': lambda v:v.update(component_loss_scope_weights_and_stored_prefix_weights_verified=False),
            'premature_selector_claim': lambda v:v.update(conditional_output_selector_independently_accepted=True),
        }
        for name, edit in edits.items():
            proof=copy.deepcopy(search_proof);edit(proof)
            try:oracle.validate_search_sequence(proof,DB_SHA,'0087',20,414)
            except AssertionError:
                result=dict(control=name,rejected=True,input_gate_only=True)
                new(args.output/(name+'.json'),result); negatives.append(result)
            else:raise AssertionError('input control accepted: '+name)
        assert len(negatives) == 22 and sha(DB) == DB_SHA
        final=dict(binding,mathematical_controls=math_controls,negative_controls_rejected=len(negatives),
            existing_small_sequence_events=20,existing_component_decisions=229,
            existing_MAP_and_Bayes_output_differences=0, sole_K1_supported_action_verified=True,existing_scoped_identity_queries=4794,
            max_abs_error=max(baseline['max_abs_error'].values()),atol=1e-8,rtol=1e-8,
            checked_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
        new(args.output/'qualification.json',final);register(args.output/'qualification.json',final['kind'])
        print(json.dumps(final),flush=True)
    except BaseException as error:
        failure=dict(binding,type=type(error).__name__,message=str(error),completed_negative_controls=len(negatives),software_qualified=False)
        new(args.output/'failure.json',failure);register(args.output/'failure.json','rbf_fixed_topK_selector_qualification_failure')
        raise


if __name__ == '__main__':main()
