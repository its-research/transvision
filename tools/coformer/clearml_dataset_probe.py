#!/usr/bin/env python3
"""Inspect candidate ClearML datasets without materializing their contents."""

from __future__ import annotations

import json
from collections import Counter

from clearml import Dataset, Task


DATASET_IDS = (
    "f24a09190df8449fa0192fb3fda6c25a",
    "7c032a13f91947cc8490c032d6e3325c",
    "0bf769e66d9a426bb08e8a2be7f170cb",
)

REQUIRED_SUFFIXES = (
    "cooperative/data_info.json",
    "vehicle-side/data_info.json",
    "infrastructure-side/data_info.json",
    "manifests/temporal_manifest_v2.json",
)


def _entry_size(entry: object) -> int:
    value = getattr(entry, "size", 0)
    return value if isinstance(value, int) and value >= 0 else 0


def _summarize(dataset_id: str) -> dict[str, object]:
    dataset = Dataset.get(dataset_id=dataset_id, only_completed=True)
    files = sorted(dataset.list_files())
    entries = dataset.file_entries_dict
    top_level = Counter(path.split("/", 1)[0] for path in files)
    markers = {
        suffix: [path for path in files if path.endswith(suffix)][:5]
        for suffix in REQUIRED_SUFFIXES
    }
    categories = {
        "vehicle_bin": sum(
            "/vehicle-side/velodyne/" in path and path.endswith(".bin")
            for path in files
        ),
        "infrastructure_bin": sum(
            "/infrastructure-side/velodyne/" in path and path.endswith(".bin")
            for path in files
        ),
        "vehicle_pcd": sum(
            "/vehicle-side/velodyne/" in path and path.endswith(".pcd")
            for path in files
        ),
        "infrastructure_pcd": sum(
            "/infrastructure-side/velodyne/" in path and path.endswith(".pcd")
            for path in files
        ),
        "vehicle_image": sum(
            "/vehicle-side/image/" in path and path.endswith(".jpg")
            for path in files
        ),
        "infrastructure_image": sum(
            "/infrastructure-side/image/" in path and path.endswith(".jpg")
            for path in files
        ),
        "world_label": sum(
            "/cooperative/label_world/" in path and path.endswith(".json")
            for path in files
        ),
        "dair_info_pickle": sum(
            path.endswith(("dair_infos_train.pkl", "dair_infos_val.pkl"))
            for path in files
        ),
        "dair_dbinfo": sum(path.endswith("dair_dbinfos_train.pkl") for path in files),
    }
    return {
        "id": dataset_id,
        "name": dataset.name,
        "project": dataset.project,
        "version": dataset.version,
        "file_count": len(files),
        "total_uncompressed_bytes": sum(_entry_size(entry) for entry in entries.values()),
        "top_level": dict(sorted(top_level.items())),
        "categories": categories,
        "required_markers": markers,
        "first_files": files[:20],
        "last_files": files[-20:],
    }


def main() -> int:
    task = Task.current_task() or Task.init(
        project_name="ResilientV2X/Training",
        task_name="CoFormer dataset metadata probe",
        task_type=Task.TaskTypes.data_processing,
    )
    summaries = []
    for dataset_id in DATASET_IDS:
        try:
            summaries.append(_summarize(dataset_id))
        except Exception as error:  # noqa: BLE001 - preserve per-dataset evidence
            summaries.append(
                {
                    "id": dataset_id,
                    "error_type": type(error).__name__,
                    "error": str(error),
                }
            )
    report = {"schema_version": 1, "datasets": summaries}
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True), flush=True)
    if not task.upload_artifact(
        "coformer_dataset_probe",
        artifact_object=report,
        wait_on_upload=True,
    ):
        raise RuntimeError("failed to upload CoFormer dataset probe artifact")
    task.flush(wait_for_uploads=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
