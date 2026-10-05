#!/usr/bin/env python3
"""Read-only byte audit of host 112's SPD raw train against ClearML inventory."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shlex
import subprocess


INVENTORY = Path("/Volumes/Data/test/recover-before-fuse/artifacts/"
                 "spd-official-train-source-20260930/verified-inventory/verified-inventory.bin")
INVENTORY_SHA256 = "e04f07ad6e4eeb7a505ca33eed31bafd27d4157e40cc34f73e9e961a07136ce0"
OUTPUT = Path("/Volumes/Data/test/recover-before-fuse/receipts/"
              "spd-official-oof-remote-112-source-readback-20260930.json")
HOST = "10.100.35.112"
REMOTE_ROOT = "/home/lbin/Desktop/eventtrack-cooptrack-runtime-20260911/full-train-inputs"

REMOTE_CODE = r'''
import hashlib,json,os,socket,stat,sys,time
root = "/home/lbin/Desktop/eventtrack-cooptrack-runtime-20260911/full-train-inputs"
inventory = json.load(sys.stdin)["entries"]
expected = {}
for row in inventory:
    name = row["path"]
    if not name.startswith("inputs/"):
        continue
    relative = name[len("inputs/"):]
    if not relative or relative.startswith("/") or ".." in relative.split("/") or "\\" in relative:
        raise ValueError("invalid expected input path")
    if relative in expected:
        raise ValueError("duplicate expected input path")
    expected[relative] = row
count = 0
byte_count = 0
started = time.monotonic()
for relative,row in expected.items():
    path = os.path.join(root, relative)
    if os.path.islink(path):
        raise ValueError("symlink input")
    info = os.stat(path)
    if not stat.S_ISREG(info.st_mode) or info.st_size != row["bytes"]:
        raise ValueError("input type or size differs: " + relative)
    digest = hashlib.sha256()
    with open(path,"rb") as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b""):
            digest.update(block)
    if digest.hexdigest() != row["sha256"]:
        raise ValueError("input hash differs: " + relative)
    count += 1
    byte_count += info.st_size
    if count % 20000 == 0:
        elapsed = max(time.monotonic()-started,0.001)
        eta = (len(expected)-count)*elapsed/count
        print("REMOTE_SPD_PROGRESS %d/%d ETA=%.1fs" % (count,len(expected),eta),flush=True)
print("REMOTE_SPD_RESULT " + json.dumps({"host":socket.gethostname(),
      "root":root,"input_file_count":count,"input_bytes":byte_count,
      "status":"all_clearml_inventory_inputs_match"},sort_keys=True),flush=True)
'''


def sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def audit() -> dict:
    if OUTPUT.exists():
        raise FileExistsError("remote source audit receipt is create-once")
    raw = INVENTORY.read_bytes()
    if sha256(raw) != INVENTORY_SHA256:
        raise ValueError("ClearML inventory bytes differ")
    command = ["ssh", "-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=yes",
               "-o", "ConnectTimeout=5", HOST, "python3 -u -c " + shlex.quote(REMOTE_CODE)]
    process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT, text=False)
    assert process.stdin is not None and process.stdout is not None
    process.stdin.write(raw)
    process.stdin.close()
    observed = None
    for line in process.stdout:
        text = line.decode(errors="replace").rstrip()
        if text.startswith("REMOTE_SPD_PROGRESS "):
            print(text, flush=True)
        elif text.startswith("REMOTE_SPD_RESULT "):
            observed = json.loads(text.removeprefix("REMOTE_SPD_RESULT "))
        else:
            print("REMOTE_SPD_DIAGNOSTIC " + text[:200], flush=True)
    status = process.wait()
    if status != 0 or observed is None:
        raise RuntimeError("remote inventory readback did not finish successfully")
    if (observed["status"] != "all_clearml_inventory_inputs_match"
            or observed["root"] != REMOTE_ROOT
            or observed["input_file_count"] != 106536
            or observed["input_bytes"] != 4186610538):
        raise ValueError("remote source summary differs from accepted ClearML inventory")
    receipt = {"kind": "spd_official_oof_remote_source_readback_v1",
               "status": observed["status"], "remote_host": HOST,
               "remote_hostname": observed["host"], "remote_root": REMOTE_ROOT,
               "input_file_count": observed["input_file_count"],
               "input_bytes": observed["input_bytes"],
               "clearml_inventory_sha256": INVENTORY_SHA256,
               "extra_remote_files_audited": False,
               "official_val_or_test_read": False,
               "checked_at_utc": datetime.now(timezone.utc).isoformat()}
    with OUTPUT.open("x") as stream:
        json.dump(receipt, stream, ensure_ascii=False, sort_keys=True, indent=2)
        stream.write("\n")
    print("REMOTE_SPD_RECEIPT " + sha256(OUTPUT.read_bytes()), flush=True)
    return receipt


if __name__ == "__main__":
    audit()
