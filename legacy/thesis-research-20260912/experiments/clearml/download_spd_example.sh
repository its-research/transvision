#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
exec "$script_dir/download_v2xseq_example.sh" spd "${1:-artifacts/v2xseq-official-examples}"
