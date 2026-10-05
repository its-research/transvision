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
     'evaluate_mechanism_diagnostics_v2.py'], [],
     ['Original first/replay byte parity including the preserved M3-seed2027 failure',
      'Complete causal error-duration audit; existing anchored episodes are insufficient']),
    ('exact_models', (), ['run_stage2_exact_random_matrix_l40.py',
     'run_stage2_posterior_update_matrix_l40.py', 'run_stage2_broad_exact_matrix_l40.py'], [],
     ['Reuse accepted matrices; bind remaining event and resource comparisons separately']),
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
    ('spd_seen_val', ('learned_priority', 'resource_baselines'), ['run_paper.py', 'evaluate_paper.py'],
     ['exclusive_paper_runtime.py', 'paper_protocol.py'],
     ['Matching accepted cache/checkpoints, frozen evaluation plan and complete predictions',
      'Report seen-val exploratory scope']),
    ('v2v4real_official_test', ('learned_priority', 'resource_baselines'),
     ['run_paper.py', 'evaluate_paper.py', 'prepare_v2v4real_ground_truth.py', 'v2v4real_gt_oracle.py',
      'fit_v2v4real_vehicle_calibration.py', 'build_native_paper_cache.py', 'build_v2v4real_raw_native_cache.py',
      'run_v2v4real_official_test_raw_features.py', 'evaluate_v2v4real_native_vehicle.py'],
     ['paper_protocol.py', 'paper_evaluation_policy.py', 'v2v4real_ground_truth.py', 'v2v4real_vehicle_calibration.py'],
     ['Authorized official split and merged vehicle evaluation: v2v4real-official-benchmark-vehicle-v1; physical session mapping is not required',
      'Vehicle GT, train-only calibration and score-only raw-cache recalibration entries exist; actual artifacts remain unaccepted',
      'Official-test raw producer privately loads the pinned original native detector core; actual output bytes and numerical acceptance remain required',
      'GT-free raw train/official-test to 9D cache bridge requires explicit causal motion/full covariance assets; no implicit zero velocity',
      'Native 3D IoU AMOTP is distinct from distance AMOTP; explicit coordinate and official sequence mapping assets are required',
      'Reuse applicable assets; new vehicle GT/calibration must not inherit strict-Car labels or metrics',
      'Frozen nominal 10 Hz and train-only selection remain; disclose known overlap, preserve overlap-v1, do not create a new v2 split']),
    ('public_baselines', (), ['run_public_baseline.py', 'public_native_outputs.py'], [],
     ['Verified native assets/commits for CoopTrack, SparseCoop and DMSTrack',
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
