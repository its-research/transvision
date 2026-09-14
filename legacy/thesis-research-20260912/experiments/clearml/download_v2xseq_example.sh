#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "usage: $0 <spd|tfd> [destination]" >&2
}

if [[ ${V2XSEQ_ACCESS_ACKNOWLEDGED:-} != 1 ]]; then
  echo "refusing download: confirm the official V2X-Seq access boundary, then set V2XSEQ_ACCESS_ACKNOWLEDGED=1" >&2
  exit 2
fi

kind=${1:-}
destination=${2:-artifacts/v2xseq-official-examples}

case "$kind" in
  spd)
    file_id='1lyl1gdrbfDI-sTQq7EoU-SkdBcWJqFn7'
    filename='V2X-Seq-SPD-Example.zip'
    expected_bytes=585329481
    ;;
  tfd)
    file_id='1EI_INcDVkcDRlyfcj8PK1cf2JCfDFZyQ'
    filename='V2X-Seq-TFD-Example.zip'
    expected_bytes=285827698
    ;;
  *)
    usage
    exit 2
    ;;
esac

if ! python3 -m gdown --version >/dev/null 2>&1; then
  echo 'gdown 6.x is required; install with: python -m pip install "gdown>=6,<7"' >&2
  exit 2
fi

mkdir -p "$destination"
archive="$destination/$filename"
partial="$archive.part"

python3 -m gdown "$file_id" --continue -O "$partial"

actual_bytes=$(wc -c < "$partial" | tr -d '[:space:]')
if [[ "$actual_bytes" != "$expected_bytes" ]]; then
  echo "downloaded byte count mismatch: expected $expected_bytes, observed $actual_bytes; leaving $partial for diagnosis" >&2
  exit 1
fi

if ! unzip -tq "$partial"; then
  echo "download is not a valid ZIP archive; leaving $partial for diagnosis" >&2
  exit 1
fi

mv "$partial" "$archive"

python3 -c 'import hashlib, pathlib, sys; p = pathlib.Path(sys.argv[1]); h = hashlib.sha256(); f = p.open("rb"); [h.update(chunk) for chunk in iter(lambda: f.read(1024 * 1024), b"")]; print(f"{h.hexdigest()}  {p}")' "$archive"
