#!/usr/bin/env python3
"""List portable runtime binaries needing glibc newer than Ubuntu 20.04."""

import json
from pathlib import Path
import re
import subprocess


def main():
    inspected = 0
    newer = []
    for path in sorted(Path("/opt/cooptrack").rglob("*")):
        if not path.is_file() or path.is_symlink() or ".so" not in path.name:
            continue
        output = subprocess.run(["readelf", "-V", str(path)], stdout=subprocess.PIPE,
                                stderr=subprocess.DEVNULL, text=True).stdout
        versions = [tuple(int(part) for part in match.split("."))
                    for match in re.findall(r"\bGLIBC_([0-9.]+)\b", output)]
        if versions:
            inspected += 1
            if max(versions) > (2, 31):
                newer.append({"path": str(path), "max_glibc": ".".join(map(str, max(versions)))})
    print(json.dumps({"inspected": inspected, "requires_newer_than_2_31": newer}, indent=2))


if __name__ == "__main__":
    main()
