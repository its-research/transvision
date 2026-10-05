"""Candidate bounded-chunk transfer; not promoted into frozen running jobs."""
import hashlib
import json
from pathlib import Path
import time
from urllib.parse import urlparse, urlunparse

def download_registered_artifact(artifact, expected_sha, expected_bytes, target):
    from clearml.storage.helper import StorageHelper
    from clearml.backend_api.session import Session
    registered = urlparse(artifact.url)
    active = urlparse(Session.get_files_server_host())
    hosts = {"10.100.35.118:8081", "10.100.34.118:8081"}
    if (registered.scheme != "http" or active.scheme != "http"
            or registered.netloc not in hosts or active.netloc not in hosts
            or registered.username or registered.query or registered.fragment):
        raise ValueError("unregistered artifact/file-service route")
    if artifact.hash != expected_sha or artifact.size != expected_bytes:
        raise ValueError("registered artifact identity differs")
    url = urlunparse(registered._replace(netloc=active.netloc))
    target = Path(target)
    digest = hashlib.sha256()
    received = 0
    started = time.monotonic()
    print("EVENTTRACK_DOWNLOAD_ETA " + json.dumps({"artifact": target.name,
          "eta_seconds": None, "eta_status": "unknown", "file_service": active.netloc}), flush=True)
    with target.open("xb") as stream:
        for block in StorageHelper.get(url).download_as_stream(url, chunk_size=1024 * 1024):
            stream.write(block)
            digest.update(block)
            prior = received
            received += len(block)
            if received > expected_bytes:
                raise ValueError("artifact longer than registered byte count")
            if received // (256 * 1024 * 1024) > prior // (256 * 1024 * 1024):
                elapsed = time.monotonic() - started
                print("EVENTTRACK_DOWNLOAD_ETA " + json.dumps({"artifact": target.name,
                      "received_bytes": received, "expected_bytes": expected_bytes,
                      "eta_seconds": (expected_bytes-received)*elapsed/received}), flush=True)
    if received != expected_bytes or digest.hexdigest() != expected_sha:
        raise ValueError("downloaded byte count or SHA differs")
    return target
