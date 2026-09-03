import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs/event_track_v2x/experiment_v1.json"


def test_frozen_experiment_config_contains_preregistered_matrix() -> None:
    value = json.loads(CONFIG.read_text(encoding="utf-8"))
    assert value["schema_version"] == 1
    assert value["primary_metric"] == "Robust-AssA@64k"
    assert value["primary_budget_bytes_per_second"] == 64000
    assert value["byte_budgets_per_second"] == [
        16000,
        32000,
        64000,
        128000,
        256000,
    ]
    assert value["network_conditions"] == [f"C{index}" for index in range(10)]
    assert value["network_seeds"] == list(range(1001, 1011))
    assert value["training_seeds"] == [1337, 2027, 3407]
    assert value["publication_statistics"] == {
        "alpha": 0.05,
        "random_seed": 1337,
        "resamples": 10000,
    }
    assert value["scheduler"]["token_bucket_burst_seconds"] == 1.0
    assert value["scheduler"]["candidate_id"] == "marginal_voi"
    assert len(value["scheduler"]["ids"]) == 6
    assert value["external_validation"]["dataset_id"] == "griffin-25m"
    assert value["external_validation"]["car_only"] is True
    assert len(value["methods"]) == len(set(value["methods"])) == 8
    assert value["data"]["tfd_included"] is False
    assert value["data"]["development"] == {
        "final_refit_sequence_count": 46,
        "fold_assignment": "sha256-sort-then-round-robin",
        "fold_count": 5,
        "fold_salt": "eventtrack-v2x-spd-development-5fold-v1",
        "fold_sizes": [10, 9, 9, 9, 9],
        "selection_estimate": "pooled-out-of-fold",
        "split": "train",
        "split_sha256_bound": True,
    }
    assert value["data"]["confirmatory"] == {
        "one_shot_after_seal": True,
        "sequence_count": 21,
        "split": "val",
    }
    assert value["data"]["official_test_policy"] == "excluded"
    assert value["failure_policy"]["missing_run_allowed"] is False
