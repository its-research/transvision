#!/usr/bin/env python3
"""Check finite-history Bayes omission updates against the frozen independent oracle.

This data-free Stage2 component does not establish a tracker or paper result.
"""
import datetime
from decimal import Decimal, localcontext
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import random
import tarfile
import tempfile
import time
from urllib.parse import urlparse

SOURCE_TASK = "a07847eb50c3447a8210bd3914a98bdd"
SOURCE_SHA = "4ee688b675df384926468baa6aadc4deb22559685acd0c96c4306eaf0efc901a"
PROJECT = "Thesis/Recover-Before-Fuse/Training"
SEEDS = (1337, 2027, 3407)
HOSTS = {"10.100.34.118:8081", "10.100.35.118:8081"}


def sha(data):
    return hashlib.sha256(data).hexdigest()


def source_bytes(artifact):
    from clearml.storage.helper import StorageHelper
    url = artifact.url
    if urlparse(url).netloc not in HOSTS:
        raise ValueError("unapproved ClearML artifact host")
    url = url.replace("10.100.34.118:8081", "10.100.35.118:8081", 1)
    data = b"".join(StorageHelper.get(url).download_as_stream(url))
    if len(data) != artifact.size or sha(data) != artifact.hash or sha(data) != SOURCE_SHA:
        raise ValueError("frozen oracle archive byte identity differs")
    return data


def load_oracle(data, root):
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as archive:
        member = archive.getmember("stage_two_exact_history_oracle.py")
        if not member.isfile():
            raise ValueError("oracle source is not a regular file")
        payload = archive.extractfile(member).read()
    path = root / member.name
    path.write_bytes(payload)
    spec = importlib.util.spec_from_file_location("frozen_stage2_posterior_oracle", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.oracle, sha(payload)


def node(node_id, source_id, frame_id, arrival, choices):
    return {"node_id": node_id, "source_id": source_id, "frame_id": frame_id,
            "information_us": arrival - 10, "arrival_us": arrival,
            "choices": [{"parent_id": parent, "log_potential": weight}
                        for parent, weight in choices]}


def cases(seed):
    rng = random.Random(seed)
    a = node("a", -1, "anchor", 100, [(None, 0.)])
    b = node("b", 0, "scan0", 200, [(None, 0.), ("a", rng.uniform(-.01, .01))])
    c = node("c", 0, "scan0", 210, [(None, 0.), ("a", rng.uniform(-.01, .01))])
    d = node("d", 1, "scan1", 300, [(None, 0.), ("a", .1), ("b", -.1)])
    return [
        ("both_sides_empty", []), ("birth_only", [a]),
        ("one_sided_candidate", [a, b]),
        ("same_slot_conflict_near_equal", [a, b, c]),
        ("rectangular_two_sources", [a, b, c, d]),
        ("wide_finite_log_span", [a, node("b", 0, "scan0", 200,
                                          [(None, -500.), ("a", 500.)]),
                                   node("c", 1, "scan1", 300,
                                        [(None, 500.), ("a", -500.), ("b", 0.)])]),
        ("later_disambiguation", [a, b, c, node("d", 1, "scan1", 300,
                                                [(None, 0.), ("a", -8.),
                                                 ("b", 8.), ("c", -8.)])]),
    ]


def check_update(result):
    histories = result["history_probabilities"]
    if len(histories) == 1:
        return {"status": "vacuous_single_history", "legal_histories": 1}
    with localcontext() as context:
        context.prec = 80
        ranked = sorted(histories, key=lambda h: (-Decimal(h["probability_decimal"]),
                                                   str(h["parents"])))
        retained = ranked[:max(1, len(ranked) // 2)]
        omitted = ranked[len(retained):]
        # Evidence is fixed from legal parent histories, never from GT or a bank.
        def likelihood(h):
            digest = hashlib.sha256(repr(h["parents"]).encode()).digest()
            return Decimal(25 + int.from_bytes(digest[:2], "big") % 376) / 100

        def mass(rows):
            return sum((Decimal(h["probability_decimal"]) for h in rows), Decimal(0))

        eta = mass(omitted)
        retained_mass = mass(retained)
        omitted_weight = sum((Decimal(h["probability_decimal"]) * likelihood(h)
                              for h in omitted), Decimal(0))
        retained_weight = sum((Decimal(h["probability_decimal"]) * likelihood(h)
                               for h in retained), Decimal(0))
        if not 0 < eta < 1 or retained_mass <= 0 or retained_weight <= 0:
            raise ValueError("proper positive retained/omitted partition required")
        exact = omitted_weight / (omitted_weight + retained_weight)
        omitted_mean = omitted_weight / eta
        retained_mean = retained_weight / retained_mass
        formula = eta * omitted_mean / (retained_mass * retained_mean + eta * omitted_mean)
        kappa = max(likelihood(h) for h in omitted) / min(likelihood(h) for h in retained)
        upper = eta * kappa / (retained_mass + eta * kappa)
        if abs(exact - formula) > Decimal("1e-65") or exact > upper + Decimal("1e-65"):
            raise ValueError("independent Bayes update or finite kappa bound differs")
        # Explicitly demonstrate why an unverified kappa=1 certificate is invalid.
        bad_omitted = eta * Decimal(1000)
        bad_retained = retained_mass / Decimal(1000)
        amplified = bad_omitted / (bad_omitted + bad_retained)
        if amplified <= eta:
            raise ValueError("invalid likelihood-ratio counterexample not realized")
        return {"status": "verified", "legal_histories": len(histories),
                "retained_histories": len(retained), "omitted_histories": len(omitted),
                "prior_omitted_mass_decimal": str(eta),
                "posterior_omitted_mass_decimal": str(exact),
                "bayes_formula_decimal": str(formula),
                "finite_kappa_decimal": str(kappa),
                "finite_kappa_upper_decimal": str(upper),
                "invalid_kappa_one_counterexample_posterior_decimal": str(amplified)}


def run(oracle, oracle_sha, task=None):
    started = time.monotonic()
    checks = []
    for seed in SEEDS:
        for name, nodes in cases(seed):
            exact = oracle(nodes, max_configurations=100000)
            if exact["production_modules_imported"] is not False:
                raise ValueError("oracle imported production code")
            row = check_update(exact)
            row.update({"seed": seed, "case": name,
                        "oracle_result_sha256": sha(json.dumps(exact, sort_keys=True,
                                                                 separators=(",", ":")).encode())})
            checks.append(row)
            elapsed = time.monotonic() - started
            eta = elapsed / len(checks) * (21 - len(checks))
            print(f"stage2 posterior update {len(checks)}/21 seed={seed} case={name} "
                  f"ETA={eta:.1f}s", flush=True)
            if task is not None:
                task.get_logger().report_scalar("posterior_matrix", "completed_cases",
                                                len(checks), iteration=len(checks))
    if len(checks) != 21 or sum(r["status"] == "verified" for r in checks) != 15:
        raise ValueError("posterior matrix coverage differs")
    return {"kind": "stage2_independent_finite_history_posterior_update_matrix_v1",
            "checked_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "seeds": SEEDS, "case_count": len(checks), "checks": checks,
            "oracle_source_task_id": SOURCE_TASK, "oracle_archive_sha256": SOURCE_SHA,
            "oracle_source_sha256": oracle_sha,
            "dataset_read": False, "ground_truth_read": False,
            "production_modules_imported": False, "parameter_training": False,
            "full_stage_two_complete": False, "same_resource_baselines_complete": False,
            "paper_performance_complete": False}


def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    from clearml import Task
    task = Task.init(project_name=PROJECT,
                     task_name="RBF stage2 posterior update matrix L40S CPU",
                     auto_connect_frameworks=False, auto_connect_arg_parser=False)
    task.reload()
    worker = task.data.last_worker or ""
    if not worker.startswith("10.100.35.121-L40S:"):
        raise ValueError("L40S CPU worker required")
    expected = task.get_parameters()["General/bootstrap_source_sha256"]
    if sha(Path(__file__).read_bytes()) != expected:
        raise ValueError("bootstrap source differs")
    source = Task.get_task(task_id=SOURCE_TASK)
    if source.status != "completed" or source.artifacts["source"].hash != SOURCE_SHA:
        raise ValueError("frozen oracle source task differs")
    with tempfile.TemporaryDirectory(prefix="rbf-stage2-posterior-matrix-") as temp:
        root = Path(temp)
        oracle, oracle_sha = load_oracle(source_bytes(source.artifacts["source"]), root)
        receipt = run(oracle, oracle_sha, task)
        receipt.update({"task_id": task.id, "worker_id": worker,
                        "bootstrap_source_sha256": expected, "device": "cpu"})
        path = root / "posterior-update-matrix-receipt.json"
        path.write_text(json.dumps(receipt, sort_keys=True, indent=2) + "\n")
        task.upload_artifact("posterior-update-matrix-receipt", artifact_object=str(path),
                             wait_on_upload=True)
        task.close()


if __name__ == "__main__":
    main()
