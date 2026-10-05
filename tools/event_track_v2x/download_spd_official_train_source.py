#!/usr/bin/env python3
"""Resume the frozen SPD official-train ClearML archive by authenticated ranges.

The output is create-once evidence. Failed ranges are truncated before retry;
neither a partial file nor ClearML's completed status is an accepted download.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time
from urllib.parse import urlparse

TASK_ID = "9a7e7a9213954b57a35403f777e74561"
ARTIFACT = "train-inputs"
EXPECTED_SHA256 = "c793bc74ec35140c894fd7f3d14fd2cce231f0b919eee3fdbeca7515b3d1e0c1"
EXPECTED_BYTES = 4070034766
FILE_HOST = "10.100.35.118:8081"
RANGE_BYTES = 64 * 1024 * 1024


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def download(output: Path, receipt: Path) -> dict:
    from clearml import Task
    from clearml.storage.helper import StorageHelper

    if receipt.exists():
        raise ValueError("create-once acceptance receipt already exists")
    task = Task.get_task(task_id=TASK_ID)
    artifact = task.artifacts[ARTIFACT]
    if (task.status != "completed" or artifact.hash != EXPECTED_SHA256
            or artifact.size != EXPECTED_BYTES):
        raise ValueError("frozen ClearML source identity differs")
    url = artifact.url.replace(urlparse(artifact.url).netloc, FILE_HOST, 1)
    if urlparse(url).netloc != FILE_HOST or urlparse(url).scheme != "http":
        raise ValueError("unapproved artifact host")
    driver = StorageHelper.get(url)._driver
    container = driver._containers["http://" + FILE_HOST]
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() and output.stat().st_size > EXPECTED_BYTES:
        raise ValueError("existing partial file exceeds declared size")
    initial_size = output.stat().st_size if output.exists() else 0
    started = time.monotonic()
    mode = "r+b" if output.exists() else "x+b"
    with output.open(mode) as stream:
        offset = stream.seek(0, 2)
        while offset < EXPECTED_BYTES:
            end = min(offset + RANGE_BYTES, EXPECTED_BYTES) - 1
            expected_length = end - offset + 1
            for attempt in range(1, 4):
                response = None
                try:
                    headers = dict(container.get_headers(url))
                    headers["Range"] = f"bytes={offset}-{end}"
                    response = container.session.get(url, headers=headers,
                                                     timeout=(10, 120), stream=True)
                    if (response.status_code != 206
                            or response.headers.get("Content-Range")
                            != f"bytes {offset}-{end}/{EXPECTED_BYTES}"
                            or int(response.headers.get("Content-Length", "-1")) != expected_length):
                        raise ValueError("file server did not honor the exact byte range")
                    stream.seek(offset)
                    received = 0
                    for chunk in response.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            stream.write(chunk)
                            received += len(chunk)
                    if received != expected_length:
                        raise ValueError("short range response")
                    stream.flush()
                    offset += received
                    if offset % (256 * 1024 * 1024) < RANGE_BYTES or offset == EXPECTED_BYTES:
                        elapsed = max(time.monotonic() - started, 0.001)
                        # This run's rate excludes bytes already present on resume.
                        run_bytes = offset - initial_size
                        eta = (EXPECTED_BYTES - offset) / (run_bytes / elapsed) if run_bytes else None
                        print(f"SPD source {offset}/{EXPECTED_BYTES} bytes ETA="
                              f"{eta:.1f}s" if eta is not None else "SPD source ETA=unknown",
                              flush=True)
                    break
                except Exception:
                    stream.seek(offset)
                    stream.truncate(offset)
                    stream.flush()
                    if attempt == 3:
                        raise
                    time.sleep(attempt)
                finally:
                    if response is not None:
                        response.close()
        stream.flush()
    if output.stat().st_size != EXPECTED_BYTES or digest(output) != EXPECTED_SHA256:
        raise ValueError("complete local archive byte identity differs")
    value = {"kind": "spd_official_train_source_clearml_range_readback_v1",
             "status": "byte_verified", "task_id": TASK_ID, "artifact": ARTIFACT,
             "bytes": EXPECTED_BYTES, "sha256": EXPECTED_SHA256,
             "local_path": str(output.absolute()),
             "checked_at_utc": datetime.now(timezone.utc).isoformat(),
             "fold_inputs_prepared": False, "detector_training_started": False}
    with receipt.open("x") as stream:
        json.dump(value, stream, ensure_ascii=False, sort_keys=True, indent=2)
        stream.write("\n")
    print("SPD source readback complete receipt_sha256=" + digest(receipt), flush=True)
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    download(args.output, args.receipt)


if __name__ == "__main__":
    main()
