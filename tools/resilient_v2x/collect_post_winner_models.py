#!/usr/bin/env python3
"""Download the four verified Table VI final checkpoints exactly once."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import tempfile
from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path
from urllib.parse import unquote, urlsplit

from clearml import Task

try:
    from tools.resilient_v2x import collect_clearml_formal_models as formal
except ModuleNotFoundError as error:
    if error.name != "tools":
        raise
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from tools.resilient_v2x import collect_clearml_formal_models as formal


DOCUMENT_TYPE = "resilient_v2x_table_vi_local_final_model"
EXPECTED_P = ("0.0", "0.1", "0.3", "0.5")
DEFAULT_EVIDENCE = Path(
    "artifacts/resilient_v2x/paper-post-winner/formal-evidence.json"
)
DEFAULT_OUTPUT = Path("artifacts/trained_models/completed-live")


def _canonical(value: object) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


def _sealed(value: Mapping[str, object]) -> dict[str, object]:
    result = dict(value)
    result.pop("content_sha256", None)
    result["content_sha256"] = hashlib.sha256(
        _canonical(result).encode("utf-8")
    ).hexdigest()
    return result


def _read(path: Path) -> dict[str, object]:
    path = path.resolve(strict=True)
    if path.is_symlink() or not path.is_file():
        raise RuntimeError("post-winner evidence must be a regular file")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError("post-winner evidence must be an object")
    observed = str(value.get("seal_sha256") or "")
    detached = dict(value)
    detached.pop("seal_sha256", None)
    expected = hashlib.sha256(_canonical(detached).encode("utf-8")).hexdigest()
    if observed != expected:
        raise RuntimeError("post-winner evidence seal mismatch")
    return value


def _atomic_json(path: Path, value: object) -> None:
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


@contextmanager
def _lock(root: Path):
    path = root / ".post-winner-models.lock"
    descriptor = os.open(path, os.O_CREAT | os.O_RDWR, 0o600)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield
    except BlockingIOError as error:
        raise RuntimeError("post-winner model collection is already running") from error
    finally:
        os.close(descriptor)


def _model(task: object, model_id: str, url: str) -> object:
    models = task.get_models()
    if not isinstance(models, Mapping):
        raise RuntimeError("training task cannot expose output models")
    matches = [
        model
        for model in models.get("output") or []
        if str(getattr(model, "id", "") or "") == model_id
    ]
    if len(matches) != 1:
        raise RuntimeError("training task lacks the unique final OutputModel")
    model = matches[0]
    if (
        str(getattr(model, "task", "") or "") != str(getattr(task, "id", ""))
        or str(getattr(model, "url", "") or "") != url
    ):
        raise RuntimeError("final OutputModel identity drifted")
    return model


def collect(*, evidence_path: Path, output_root: Path) -> dict[str, object]:
    evidence = _read(evidence_path)
    rows = evidence.get("table_vi")
    if not isinstance(rows, list) or [str(row.get("p")) for row in rows if isinstance(row, Mapping)] != list(EXPECTED_P):
        raise RuntimeError("Table VI evidence inventory drifted")
    root = formal._safe_output_root(output_root)
    auth_provider, auth_refresher = formal._resolve_download_authentication(
        Task, auth_header_provider=None, auth_refresher=None
    )
    collected: list[dict[str, object]] = []
    with _lock(root):
        for row in rows:
            p = str(row["p"])
            training = row.get("training")
            if not isinstance(training, Mapping):
                raise RuntimeError(f"p={p} training evidence is invalid")
            task_id = formal._clearml_id(training.get("task_id"), f"p={p} task")
            checkpoint = training.get("checkpoint")
            if not isinstance(checkpoint, Mapping):
                raise RuntimeError(f"p={p} checkpoint evidence is invalid")
            model_id = formal._clearml_id(checkpoint.get("model_id"), f"p={p} model")
            checkpoint_sha = formal._sha256(checkpoint.get("sha256"), f"p={p} checkpoint")
            checkpoint_size = formal._positive_integer(checkpoint.get("size_bytes"), f"p={p} checkpoint size")
            url = str(checkpoint.get("url") or "")
            filename = Path(unquote(urlsplit(url).path)).name
            if not filename or formal.SAFE_FILENAME_PATTERN.fullmatch(filename) is None:
                raise RuntimeError(f"p={p} checkpoint filename is unsafe")
            task = Task.get_task(task_id=task_id)
            if str(task.status) != "completed":
                raise RuntimeError(f"p={p} training task is not completed")
            _model(task, model_id, url)
            directory = formal._safe_destination(root, f"table_vi_p_{p.replace('.', '_')}", "manifest.json").parent
            destination = formal._safe_destination(root, directory.name, filename)
            destination, reused = formal._download_verified(
                model_url=url,
                destination=destination,
                expected_size_bytes=checkpoint_size,
                expected_sha256=checkpoint_sha,
                auth_header_provider=auth_provider,
                auth_refresher=auth_refresher,
            )
            manifest = _sealed(
                {
                    "schema_version": 1,
                    "document_type": DOCUMENT_TYPE,
                    "p_lidar": float(p),
                    "p_camera": float(p),
                    "training_seed": evidence["training_seed"],
                    "checkpoint_policy": evidence["checkpoint_policy"],
                    "training_task_id": task_id,
                    "model_id": model_id,
                    "filename": destination.name,
                    "size_bytes": checkpoint_size,
                    "sha256": checkpoint_sha,
                    "source_evidence_seal": evidence["seal_sha256"],
                }
            )
            manifest_path = directory / "manifest.json"
            if manifest_path.exists():
                observed = json.loads(manifest_path.read_text(encoding="utf-8"))
                if observed != manifest:
                    raise RuntimeError(f"p={p} existing manifest drifted")
            else:
                _atomic_json(manifest_path, manifest)
            collected.append(
                {
                    "p": p,
                    "task_id": task_id,
                    "model_id": model_id,
                    "path": str(destination),
                    "size_bytes": checkpoint_size,
                    "sha256": checkpoint_sha,
                    "reused": reused,
                }
            )
    return {
        "status": "complete",
        "model_count": len(collected),
        "source_evidence_seal": evidence["seal_sha256"],
        "models": collected,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", type=Path, default=DEFAULT_EVIDENCE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    print(
        json.dumps(
            collect(evidence_path=args.evidence, output_root=args.output_dir),
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
