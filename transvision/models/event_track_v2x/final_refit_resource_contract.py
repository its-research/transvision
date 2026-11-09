"""Bind current final refits to full independent training/forward evidence.

Metadata admission only: does not load tensors, run inference, or certify a
new runtime. Original checkpoints and acceptance receipts remain immutable.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

SEEDS = (1337, 2027, 3407)
EVIDENCE_FIELDS = frozenset(("training", "forward", "numeric", "plan"))


def read_reference(reference):
    if type(reference) is not dict or set(reference) != {"path", "sha256"}:
        raise ValueError("exact path/SHA evidence reference required")
    path = Path(reference["path"]).absolute()
    if any(p.is_symlink() for p in (path, *path.parents)):
        raise ValueError("evidence symlink traversal")
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != reference["sha256"]:
        raise ValueError("evidence bytes differ")
    return json.loads(raw)


def validate_final_refit(checkpoint, checkpoint_sha256, evidence):
    if type(evidence) is not dict or set(evidence) != EVIDENCE_FIELDS:
        raise ValueError("complete final-refit evidence required")
    train, forward, numeric, plan = (read_reference(evidence[k])
                                    for k in ("training", "forward", "numeric", "plan"))
    seed = checkpoint.get("seed")
    if type(seed) is not int or seed not in SEEDS or any(v.get("seed") != seed
        for v in (train, forward, numeric)):
        raise ValueError("final-refit evidence seed differs")
    if (checkpoint.get("kind") != "persistent_forest_row_identity_checkpoint_v1"
        or checkpoint.get("data_split") != "train" or checkpoint.get("fixture") is not False
        or checkpoint.get("labels_in_model_inputs") is not False
        or checkpoint.get("validation_or_test_selection") is not False):
        raise ValueError("current non-fixture train checkpoint required")
    if (train.get("kind") != "rbf_new_final_refit_six_artifacts_independent_byte_metadata_admission_v1"
        or train.get("all_six_registered_artifact_bytes_independently_read") is not True
        or train.get("frozen_nested_epoch_and_full_train_recipe_verified") is not True
        or train.get("two_head_training_and_producer_CPU_reference_gradient_report_bound") is not True):
        raise ValueError("full final-refit training admission required")
    if (forward.get("kind") != "rbf_new_final_refit_all_row_independent_byte_coverage_admission_v1"
        or forward.get("all_registered_output_bytes_verified") is not True
        or forward.get("all_46_train_sequences_row_coverage_verified") is not True
        or forward.get("train_byte_admission_sha256") != evidence["training"]["sha256"]):
        raise ValueError("complete final-refit forward byte coverage required")
    if (numeric.get("kind") != "rbf_new_final_refit_full_independent_float64_joint_reference_v1"
        or numeric.get("full_independent_numeric_pass") is not True
        or numeric.get("all_rows_selected_without_GT") is not True
        or numeric.get("numeric_failed_rows") != 0
        or numeric.get("prediction_proof_sha256") != evidence["forward"]["sha256"]
        or numeric.get("atol") != 1e-4 or numeric.get("rtol") != 1e-4):
        raise ValueError("original full independent numeric admission required")
    counts = forward.get("sequence_counts", {})
    independent = numeric.get("sequences", [])
    if (len(counts) != 46 or len(independent) != 46
        or {v["sequence_id"]: v["rows"] for v in independent} != counts
        or any(v.get("failures") != 0 for v in independent)
        or type(forward.get("rows")) is not int or forward["rows"] <= 0
        or sum(counts.values()) != forward["rows"] or numeric.get("rows") != forward["rows"]):
        raise ValueError("full independent 46-sequence row coverage differs")
    weights = checkpoint["weights"]["sha256"]
    if any(v.get("weights_sha256") != weights for v in (train, forward, numeric)):
        raise ValueError("final-refit weight identity differs")
    if (train.get("model_sha256") != checkpoint.get("model_sha256")
        or train["artifacts"]["checkpoint"]["sha256"] != checkpoint_sha256
        or train["artifacts"]["checkpoint"]["task"] != train["task_id"]
        or train["artifacts"]["plan"]["sha256"] != evidence["plan"]["sha256"]):
        raise ValueError("final-refit checkpoint/plan identity differs")
    recipe = plan.get("recipe", {})
    canonical = json.dumps(recipe, sort_keys=True, separators=(",", ":")).encode()
    if (plan.get("kind") != "rbf_nested_selected_epoch_all_class_full_train_refit_candidate_v1"
        or recipe.get("seed") != seed
        or hashlib.sha256(canonical).hexdigest() != train.get("recipe_sha256")
        or recipe.get("fit_config", {}).get("epochs") != train.get("epochs")
        or train.get("epochs") != checkpoint.get("selected_epoch")
        or recipe.get("frozen_selection_receipt_sha256") is None):
        raise ValueError("frozen full-train recipe or train-only selection differs")
    return {"seed": seed, "rows": forward["rows"], "sequences": 46,
            "checkpoint_sha256": checkpoint_sha256, "model_sha256": checkpoint["model_sha256"],
            "weights_sha256": weights, "evidence": evidence,
            "current_final_refit_full_train_and_forward_bound": True,
            "new_runtime_numerically_qualified": False}
