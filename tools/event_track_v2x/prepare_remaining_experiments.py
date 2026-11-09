#!/usr/bin/env python3
"""Freeze a source inventory of the remaining RBF experiments without dispatch.

This checks files and Python syntax, not scientific readiness. Each downstream
experiment still requires its own data, lineage, runtime and independent gates.
"""
import argparse
import ast
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TOOLS = 'tools/event_track_v2x/'
MODELS = 'transvision/models/event_track_v2x/'

# Requirements are experiment families; these are not counts of accepted runs.
FAMILIES = (
    ('mechanism_M0_M4', (), ['run_paper_pair_baselines.py', 'audit_mechanism_replay_v2.py',
     'evaluate_mechanism_diagnostics_v2.py', 'verify_stage1_preserved_streams.py',
     'stage1_sequence_statistics.py', 'stage1_independent_evaluation.py',
     'stage1_identity_episode_oracle.py', 'audit_stage1_identity_episodes.py',
     'stage1_nuscenes_statistics_oracle.py', 'stage1_nuscenes_sufficient_statistics.py',
     'bootstrap_stage1_nuscenes.py', 'verify_stage1_nuscenes_bootstrap_summary.py',
     'stage1_fusion_numeric_oracle.py',
     'audit_stage1_fusion_numerics.py', 'stage1_discrete_replay_oracle.py',
     'verify_stage1_discrete_replay.py', 'plot_stage1_qualified_paired_effects.py'], [],
     ['Original first/replay byte parity including the preserved M3-seed2027 failure',
      'Full 78 original streams and all 26 output chains have actual byte/provenance readback; M3-seed2027 strict parity remains failed',
      'All 13 original metric files and TrackEval count aggregation are independently checked; fixed 1337/10000 shared sequence draws now cover HOTA/AssA/DetA/IDF1 and all15 actual nuScenes metrics',
      'All13 original official metric-engine results are independently recomputed from full native frames, with exact counts and frozen 1e-12 float tolerance; M3-seed2027 strict replay remains failed',
      'All13 full native observable identity episodes and structural source-component-node-lifecycle-output chains are audited; whole-sequence optimal-owner ambiguity and temporal censoring remain unknown, not zero error',
      'Road residual matched association mass is not executed paired CI mass; physical source identity and counterfactual causal effects remain unproved',
      'All13 full21-sequence actual author score-subset caches independently reproduce all40 original curves and 15 headlines; complete shared1337/10000 nuScenes bootstrap is executed',
      'Complete13x10000x15 actual draw values and all345 descriptive/interval scalar checks plus105 intervals have independent byte/statistics readback; original M3 failure and M1 single-run SD remain explicit, full K/paper acceptance remains false',
      'Full native conditional fusion numerical audits require exact original covariance SHA preimages before independent complete nine-state/cross-covariance comparisons; two-frame smoke and running batches are not admission',
      'Sealed discrete association/lifecycle conditioning and previous actual output in temporal one-step checks do not prove independent global assignment, full scratch replay or causal error counterfactuals',
      'Separate full native independent temporal replay solves rectangular assignment with per-frame primal/dual certificates and persists its own state; original lifecycle and previous actual outputs are comparison targets only, with no fallback after a discrete mismatch',
      'All13 full conditional fusion audits and all13 independent temporal replays have actual individual full cohort receipts and independently read-back registered archives; pair alternatives stay sealed and neural inference and causal counterfactuals are not newly executed',
      'Pinned paired AMOTA HOTA Frag figure has all9 bindings checked, color/grayscale visual review and registered byte readback; M1 remains single-run, M3 has no qualified interval and SPD remains seen-val exploratory']),
    ('exact_models', (), ['run_stage2_exact_random_matrix_l40.py',
     'run_stage2_posterior_update_matrix_l40.py', 'run_stage2_broad_exact_matrix_l40.py',
     'run_stage2_gate_and_budget_events.py', 'audit_stage2_gate_and_budget_events.py',
     'run_stage2_crossing_occlusion_events.py', 'audit_stage2_crossing_occlusion_events.py',
     'verify_stage2_causal_recovery_witnesses.py', 'plot_stage2_causal_event_traces.py',
     'run_stage2_causal_beam_baselines.py', 'audit_stage2_causal_beam_baselines.py',
     'audit_stage2_causal_branch_states.py', 'compare_stage2_causal_beam_recovery.py',
     'run_stage2_causal_probabilistic_mht.py', 'audit_stage2_causal_probabilistic_mht.py',
     'compare_stage2_causal_probabilistic_mht.py', 'run_stage2_fixed_history_expiry.py',
     'audit_stage2_fixed_history_expiry.py', 'compare_stage2_fixed_history_expiry.py'], [],
     ['Reuse accepted matrices; bind remaining event and resource comparisons separately',
      'Accepted shared14-event streams cover initial, before-deadline late disambiguation and12 compute pulses; their coverage and the four protocol boundaries alone do not qualify all named scenarios',
      'Nine original registered empty/no-edge cases now have full18-event exact posterior,12 complete9D birth-state and sealed private-copy readback; original cases are reused, current exclusive/learned comparison qualification is separate',
      'Eighteen new wrong-gate/exhausted-budget cases at three fixture seeds and K1/4/16 have full90-event independent exact partition/regret/residual/component-product and225 complete9D state admission; model regret and evaluator identity loss are separate',
      'Unused finite-window action_budget rejection is preserved; new valid cases use the supported persistent expansion budget1 with actual decoder limits unchanged, and the three completed wrong-gate outputs are reused',
      'All27 empty/gate/budget scoped cases,108 events and237 full9D states have registered archive member byte readback; they are separate from the later causal trajectories',
      'Eighteen causal projected-crossing and missing-observation/reappearance fixtures now have261 actual events,690 full9D states,1224 independently checked residual regions and27 direct positive-suffix Bayes updates; absence is not an existence-likelihood factor',
      'Fifteen cases have immutable-SQL verified formerly frontier-only current-output recovery; three occlusion K1 cases fail within the predeclared budget and remain negative results, with no cap or tolerance increase',
      'The causal trace scientific figure binds actual root/regret values and visually reviewed color/grayscale exports; fixture seeds are not trained model seeds and identical expansion budgets are not same-resource evidence',
      'Fifty-four same-raw/factor/schedule node/batch/joint irreversible beam runs have full783-event exhaustive class-ranking, canonical-SHA tie-breaking, retained-action and1800 full9D output admission; they are local beam variants, not reproduced MHT',
      'All72 reused RBF and new beam scopes have1224 cached root-chain9D states,5597 active/output branch byte digests and17995 complete branch9D predictions independently read back; original RBF experiments are not rerun',
      'The24-row same-input comparison retains mixed outcomes: crossing K1 RBF recovers after extra compute while K1 beams cannot restore pruned histories; wider beams are correct at new evidence arrival, and occlusion K1 RBF regret exceeds node-beam regret',
      'Thirty new exact-scan JPDA/PKF and local global-scan MHT K1/4/16 runs have435 complete events,1104 output9D states,5406 retained-branch9D predictions and300 SQL cache9D states independently checked; 102 same-input runs are compared without same-resource or native-public-method claims',
      'Twelve paired fixed-deadline actual initial-history MAP handoffs and twelve matched retained controls have372 complete events,900 output9D states,7173 branch9D predictions and423 cache states independently checked; all twelve expiry transactions remove previously retained B support and preserve fixed-budget negative results',
      'General expiry contracts, other scenario-specific baselines, learned allocation and complete same-resource comparisons remain; these declared finite fixed-deadline cases do not turn decision windows or output aging into history expiry, and formal Linux MHT remains separate']),
    ('final_forest', (), ['read_rbf_final_refit_forest_outputs.py',
     'prefetch_final_refit_rank_artifacts.py'],
     ['exclusive_paper_runtime.py', 'exclusive_completion_tracking.py'],
     ['All three final-refit independent full-cohort acceptances; no legacy inheritance',
      'Registered rank byte prefetch is transfer only; wait for its exit before terminal-task readback']),
    ('fixed_top1_topk', (), ['prepare_rbf_final_refit_top1.py', 'prepare_rbf_final_refit_topk.py',
     'accept_final_refit_top1_selector.py', 'accept_final_refit_topk_selector.py'], [],
     ['Use current frozen dispatchers and corrected receipt-row acceptors, not mutable copies',
      'Three seeds and all declared resource settings; physical GPU collision admission']),
    ('teacher', ('final_forest',), ['run_paper.py', 'train_paper_priority.py',
     'rbf_final_refit_teacher_prerequisites.py', 'prepare_rbf_final_refit_capacity_teacher.py',
     'rbf_final_refit_teacher_binding.py', 'publish_final_refit_teacher_prerequisite.py',
     'submit_final_refit_capacity_teacher.py', 'collect_final_refit_capacity_teacher.py',
     'accept_final_refit_capacity_teacher.py'],
     ['exclusive_completion_tracking.py', 'paper_runtime_selection.py'],
     ['Final-refit model-bound full train teacher cohort and independent target acceptance',
      'Use frozen prerequisite publication, GPU dispatch, native collection and independent target entries',
      'Model-bound improvement targets do not certify true identity risk']),
    ('learned_priority', ('teacher',), ['train_paper_priority.py', 'run_paper.py',
     'rbf_final_priority_admission.py', 'prepare_final_refit_priority_export.py',
     'run_final_refit_priority_after_targets.py', 'prepare_final_priority_runtime_probe.py',
     'control_final_priority_runtime_probe.py', 'audit_final_refit_priority_checkpoint.py',
     'prepare_rbf_final_refit_learned_replay.py', 'publish_final_refit_priority_checkpoint.py',
     'submit_final_refit_learned_replay.py', 'prepare_final_refit_learned_readback.py',
     'read_final_refit_learned_outputs.py', 'rbf_final_refit_learned_output_binding.py',
     'rbf_independent_learned_trajectory.py', 'check_learned_search_trajectory.py',
     'prepare_final_refit_learned_CPU.py', 'rbf_final_refit_learned_forest_binding.py',
     'accept_final_refit_learned_cohort.py'],
     ['allocation_training.py', 'allocation_policy.py', 'paper_runtime_selection.py'],
     ['Full accepted teacher manifest; train-only holdout selection and three checkpoints',
      'Frozen final-model export consumer and read-only live-provenance CPU bridge',
      'New consumer Linux probe is frozen; specific source upload authorization remains pending after automatic review rejection',
      'Actual completed Linux runtime and independent cloud readback still required before export or fit',
      'Local full-export and selected-checkpoint NumPy audit is prepared; cloud fit byte readback and learned replay remain separate',
      'Learned full-train GPU replay producer is prepared on unchanged core; actual checkpoint publication, runtime and full ordering/forest/cost admission remain required',
      'Three-file checkpoint publisher/readback and collision-safe learned dispatcher are prepared; no actual fit, checkpoint publication or learned replay has occurred',
      'Learned output byte/event/factor reader binds each rank and sequence to the admitted priority checkpoint; independent learned ordering, full states and resource costs remain separate',
      'Single-database learned trajectory checker reconstructs states/features and replays float64 selection with exact tie breaks',
      'Full 46-sequence learned acceptance entry connects qualified published checkpoints to the original five forest oracles and trajectory verification; real teacher/fit/replay outputs and measured resource comparison remain required']),
    ('resource_baselines', ('final_forest', 'fixed_top1_topk', 'learned_priority'),
     ['scan_paper_resources.py', 'run_paper.py'], ['paper_resources.py'],
     ['Same hardware, threads, batch, warm-up and total-cost accounting',
      'All-class MHT runtime contract and measured cost curves; CPU scan is not a GPU scan']),
    ('recovery_ablation', ('learned_priority',), ['run_paper.py',
     'prepare_recovery_off_full_independent_CPU.py', 'accept_recovery_off_final_cohort.py',
     'prepare_recovery_off_teacher_learned_code.py',
     'recovery_off_structure_oracle.py', 'recovery_off_action_oracle.py',
     'check_recovery_off_learned_search_trajectory.py', 'rbf_independent_recovery_off_learned_trajectory.py'],
     ['exclusive_completion_tracking.py', 'recovery_off_tracking.py', 'recovery_off_allocation.py', 'recovery_off_paper_runtime.py'],
     ['Bound, teacher and learned allocation use explicitly restricted historical support',
      'Bound full CPU oracles are qualified on original frozen producer fixtures; actual cohort outputs are still required',
      'Full independent forest/state acceptance, real cohorts and same-resource comparison remain required',
      'Learned teacher export/fit/load is separately source/config bound; no unrestricted checkpoint transfer',
      'Independent learned ordering uses immutable prior event history; real cohort verification and deployment freeze remain required']),
    ('spd_seen_val', ('learned_priority', 'resource_baselines'), ['run_paper.py', 'evaluate_paper.py',
      'publish_rbf_seen_val_forest_inputs_native_route_v2.py',
      'audit_seen_val_registered_cache_members.py', 'continue_seen_val_registered_member_audit.py',
      'prepare_seen_val_registered_URL_runtime_v2.py', 'submit_rbf_seen_val_bound_forest_native_route_v2.py',
      'continue_seen_val_native_registered_URL_dispatch_v2.py'],
     ['exclusive_paper_runtime.py', 'paper_protocol.py'],
     ['Matching accepted cache/checkpoints, frozen evaluation plan and complete predictions',
      'Specific1337/3407 complete-cache publications reuse accepted full train main gates; all seven registered HTTP byte reads and14379 independent cache member checks must complete before replay admission',
      'The native registered-URL worker is separately frozen, preserving all other original worker ASTs and literal patches; one actual6864-byte completed registered artifact and seven metadata/URL negatives qualify only the transport reader',
      'Native dispatcher freezes resolved imports and requires all seven completed publications and14379 registered member checks; qualified native-v2 training retains first idle GPU priority and all original physical-intersection checks',
      'Report seen-val exploratory scope']),
    ('v2v4real_identity', (), ['prepare_v2v4real_identity_labels.py', 'verify_v2v4real_identity_labels.py',
      'v2v4real_identity_source_geometry.py', 'v2v4real_identity_supervision.py',
      'prepare_v2v4real_identity_rows.py', 'v2v4real_identity_target_oracle.py', 'verify_v2v4real_identity_rows.py',
      'v2v4real_identity_supervision_native_v2.py', 'v2v4real_identity_target_oracle_native_v2.py',
      'retarget_v2v4real_identity_rows_native_v2.py', 'verify_v2v4real_identity_rows_native_v2.py',
      'v2v4real_identity_training_contract.py', 'train_v2v4real_joint_identity.py',
      'v2v4real_identity_training_contract_native_v2.py', 'train_v2v4real_joint_identity_native_v2.py',
      'prepare_v2v4real_identity_gpu_inputs_native_v2.py', 'run_v2v4real_identity_gpu_native_v2.py',
      'verify_v2v4real_identity_checkpoint_native_v2.py', 'submit_v2v4real_identity_native_v2.py',
      'prepare_v2v4real_identity_gpu_inputs.py', 'run_v2v4real_identity_gpu.py', 'verify_v2v4real_identity_checkpoint.py'], [],
     ['Reuse the six independently read-back native 9D cache scopes; bind each seed and original manifest SHA',
      'All 14210 train source labels and 137134 native objects have original-code geometry/ID admission; signed extents remain original corner operands',
      'Three-seed real row preparation and independent feature/parent/partial-target campaigns are running; no checkpoint qualification is inherited',
      'The first full 801957-node legacy row admission has zero cross-source positives and cannot qualify final joint training; preserve its numerical/support admission and original processes',
      'Native v2 links author-native mapped IDs at a paired frame with at least one explicit association and unique source keys; duplicate/association-conflict components remain unknown and two-fallback aliases do not create links',
      'All137134 annotations have independent native-v2 graph partition/status/crossflag agreement with34191 exact cross links; seed1337 all801957 rows are independently admitted with78029 cross and330042 temporal positive edges and15 unchanged arrays byte-identical; remaining seeds wait their own full admissions',
      'Original28-fit/4-holdout groups both have actual informative cross-source/temporal targets; native-v2 gates reject legacy proofs and a zero-positive head in either role, without changing the eight actual training/DDP function definitions',
      'Seed1337 corrected-target local training bundle has64 fully read-back members and an actual27-source import closure excluding CUDA branch-state/priority modules; cloud byte admission and disjoint live worker capacity precede dispatch, and all-row final-checkpoint NumPy admission still remains',
      'Current SPD final-refit checkpoints retain complete 677744-row qualification and are reused; V2V requires its own frozen train-name-group selection and full-train refit',
      'Native training requires complete independent real row admission, cloud input byte readback and disjoint physical 4/8-GPU admission before updates; L40S remains CPU-only',
      'Prepared selection/refit/forward code and software diagnostics do not qualify a V2V checkpoint; all real final rows must pass the unchanged SHA-bound float64 reference at 1e-4 tolerances']),
    ('v2v4real_official_test', ('learned_priority', 'resource_baselines', 'v2v4real_identity'),
     ['run_paper.py', 'evaluate_paper.py', 'prepare_v2v4real_ground_truth.py', 'v2v4real_gt_oracle.py',
      'merge_v2v4real_vehicle_ground_truth.py',
      'fit_v2v4real_vehicle_calibration.py', 'verify_v2v4real_vehicle_calibration.py',
      'build_native_paper_cache.py', 'build_v2v4real_raw_native_cache.py',
      'run_v2v4real_official_test_raw_features.py', 'evaluate_v2v4real_native_vehicle.py'],
     ['paper_protocol.py', 'paper_evaluation_policy.py', 'v2v4real_ground_truth.py', 'v2v4real_vehicle_calibration.py'],
     ['Authorized official split and merged vehicle evaluation: v2v4real-official-benchmark-vehicle-v1; physical session mapping is not required',
      'Vehicle calibration-role GT and three-seed existence calibration are independently accepted; reuse exact registered bytes',
      'Full tracking GT, fitted motion/full covariance and all six native 9D caches have separately named admissions; rebind their exact evidence before replay',
      'Official-test raw producer privately loads the pinned original native detector core; actual output bytes and numerical acceptance remain required',
      'Reuse accepted train-only causal motion/full covariance assets; first/unmatched velocity uses the declared population prior and remains explicitly unmeasured',
      'Native 3D IoU AMOTP is distinct from distance AMOTP; explicit coordinate and official sequence mapping assets are required',
      'Reuse applicable assets; new vehicle GT/calibration must not inherit strict-Car labels or metrics',
      'Frozen nominal 10 Hz and train-only selection remain; disclose known overlap, preserve overlap-v1, do not create a new v2 split']),
    ('public_baselines', (), ['run_public_baseline.py', 'public_native_outputs.py',
      'download_public_native_author_checkpoint.py', 'audit_public_native_author_checkpoints.py',
      'audit_public_native_author_source.py', 'download_public_native_feature_archive.py',
      'audit_DMSTrack_native_feature_archive.py', 'continue_DMSTrack_native_feature_audit.py',
      'prepare_DMSTrack_native_CPU_inputs.py', 'continue_DMSTrack_native_CPU_inputs.py',
      'audit_DMSTrack_native_detection_GT_inputs.py', 'audit_DMSTrack_official_test_native_GT_mapping.py',
      'continue_DMSTrack_native_CPU_inference.py'], [],
     ['Verified native assets/commits for CoopTrack, SparseCoop and DMSTrack',
      'Five actual author weights have complete byte acquisition and5744 finite checkpoint tensors audited; both DMSTrack native CAV covariance classes load all34 tensors each strictly with exact author configuration, but this is no native prediction or method acceptance',
      'DMSTrack pinned full source9696 Git blobs are independently read, author dataset symlink text is not followed; complete native train/val archive acquisition and all-member plus detection-aligned array audit receipts are separately bound',
      'Both native feature archives now have complete member and mandatory-array admissions; val41895 files11958 arrays1993 frames per CAV and train149339 files42630 arrays7105 frames per CAV, with exact public ZIP SHA bindings',
      'All60046 native detection rows and31419 GT objects match author NPY corners under predeclared float32/text rounding bounds; all1993 native GT frames match independently admitted official-test original YAML IDs, merged vehicle category and two-stage ROI, maximum corner error7.62939453125e-6m',
      'Isolated Python3.7 CPU tracking environment imports the complete unchanged native entry and passes pip check; this is author-feature tracking runtime preparation, not a complete CUDA detector environment or native prediction acceptance',
      'Portable native adapter uses identical original stdlib hashing and all six original adapter function ASTs preserve checkpoint/output/source guards; documented four author module roots replace the incomplete import path',
      'Exact qualified public native input transfer and nine-sequence inference continuation are separately frozen; no native Task is created before complete CPU inputs; native inference requires independent CPU disk readback; original num_epochs0 evaluation keeps diagnostic GT losses, with force-GT-prediction false and no detector/embedding rerun',
      'All three native output collectors bind actual paths; reported summary parsing is not independent metric acceptance',
      'Long-SCOPE verified implementation still missing; never substitute another algorithm']),
    ('independent_metrics', ('spd_seen_val', 'v2v4real_official_test'), ['evaluate_paper.py', 'evaluate_v2v4real_native_vehicle.py'],
     ['paper_reports.py', 'paper_calibration.py'],
     ['Independent GT envelopes and native metric definitions; no test-driven selection']),
    ('tables_figures', ('independent_metrics', 'recovery_ablation', 'public_baselines'),
     ['report_paper.py', 'plot_paper.py'], ['paper_reports.py'],
     ['All seeds and sequence-level paired statistics; recording clusters only when verified, otherwise report sequence units',
      'No placeholder or fixture performance in paper tables']),
    ('gpu_memory', ('fixed_top1_topk',), ['rbf_gpu_replay_measurement.py',
     'qualify_batched_branch_states.py', 'qualify_batched_branch_states_gpu.py',
     'submit_branch_state_cuda_candidate.py', 'read_branch_state_cuda_outputs.py',
     'accept_branch_state_cuda_outputs.py'],
     ['batched_row_context_scoring.py', 'batched_persistent_cache_stream.py', 'batched_branch_states.py'],
     ['Collision-safe real GPU measurement and independent numerical acceptance',
      'Specific admitted CUDA source/database package upload authorization remains pending',
      '75-80 percent per physical UUID must be observed; Worker group averages do not prove it']),
)


def inventory(root=ROOT):
    records = []
    for name, dependencies, tools, modules, gates in FAMILIES:
        sources = []
        for relative in [*(TOOLS + n for n in tools), *(MODELS + n for n in modules)]:
            path = root / relative
            entry = dict(path=relative, exists=path.is_file())
            if path.is_file():
                data = path.read_bytes()
                entry.update(sha256=hashlib.sha256(data).hexdigest(), bytes=len(data))
                try:
                    ast.parse(data, filename=str(path))
                    entry['syntax_valid'] = True
                except (SyntaxError, UnicodeError):
                    entry['syntax_valid'] = False
            sources.append(entry)
        records.append(dict(id=name, dependencies=dependencies, sources=sources,
                            source_files_present_and_parseable=all(s.get('syntax_valid', False) for s in sources),
                            required_gates=gates, experiment_ready=False, experiment_completed=False))
    return dict(kind='rbf_remaining_experiment_source_inventory_v1', families=records,
                all_experiment_code_ready=False, formal_acceptance=False,
                warning='Presence and syntax do not prove implemented semantics, runtime compatibility or acceptance.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = inventory()
    with args.output.open('x') as stream:
        json.dump(result, stream, indent=2, ensure_ascii=False)
        stream.write('\n')
    print(json.dumps(dict(output=str(args.output), families=len(result['families']),
                         all_sources_parseable=all(r['source_files_present_and_parseable'] for r in result['families']),
                         all_experiment_code_ready=False)))


if __name__ == '__main__':
    main()
