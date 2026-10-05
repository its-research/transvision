#!/usr/bin/env python3
"""Add the frozen nominal-protocol gate to a diagnostic detector source tar.

This prepares a new source lineage. Existing ClearML tasks and tarballs remain
unchanged; this command does not publish or dispatch training.
"""
import argparse
import gzip
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import tarfile

BASE_SOURCE_SHA256 = {
    '93b3f63cb1058c725224e275c713566e2a11090812026060684c77867a3c3a69',
    '122157453fa481ce27e9adb97528990bab439c8b9e30d5687229da6a57744f1a',
}
PROTOCOL_SHA256 = '40479859c407db27cc7e609c889a447057319905f94e887769ef919d57004334'
RUNNER_NAME = 'project/tools/event_track_v2x/run_v2v4real_detector_clearml.py'
VALIDATOR_NAME = 'project/transvision/models/event_track_v2x/paper_nominal_clock.py'
CONFIG_NAME = 'project/configs/event_track_v2x/v2v4real-nominal-10hz-formal-v1.json'


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def upgrade(base: Path, validator: Path, config: Path, output: Path) -> dict:
    if sha(base.read_bytes()) not in BASE_SOURCE_SHA256:
        raise ValueError('base diagnostic source differs')
    if sha(config.read_bytes()) != PROTOCOL_SHA256:
        raise ValueError('frozen protocol bytes differ')
    if output.exists() or output.is_symlink():
        raise ValueError('fresh output required')
    files = {}
    with tarfile.open(base, 'r:gz') as archive:
        for member in archive:
            name = PurePosixPath(member.name)
            if (name.is_absolute() or '..' in name.parts or not member.isfile()
                    or name.as_posix() in files
                    or name.parts[0] not in {'project', 'official'}):
                raise ValueError('unsafe base source member')
            files[name.as_posix()] = archive.extractfile(member).read()
    if RUNNER_NAME not in files or VALIDATOR_NAME in files or CONFIG_NAME in files:
        raise ValueError('base source member set differs')
    text = files[RUNNER_NAME].decode()
    edits = (
        ('from tools.event_track_v2x.extract_v2v4real_archive import extract_volume',
         'from tools.event_track_v2x.extract_v2v4real_archive import extract_volume\n'
         'from transvision.models.event_track_v2x.paper_nominal_clock import validate_formal_variant'),
        ("MEMBERSHIP_SHA = '0f429069e56c1525484d32cb7c9ec4affd49bc5eee5a724d16f3cf44a6c9b4d9'",
         "MEMBERSHIP_SHA = '0f429069e56c1525484d32cb7c9ec4affd49bc5eee5a724d16f3cf44a6c9b4d9'\n"
         f"PROTOCOL_CONFIG_SHA = '{PROTOCOL_SHA256}'"),
        ("    task.reload(); p = task.get_parameters()",
         "    task.reload(); p = task.get_parameters()\n"
         "    protocol_path = (Path(__file__).resolve().parents[2] /\n"
         "        'configs/event_track_v2x/v2v4real-nominal-10hz-formal-v1.json')\n"
         "    if not protocol_path.is_file() or protocol_path.is_symlink() or sha(protocol_path) != PROTOCOL_CONFIG_SHA:\n"
         "        raise ValueError('frozen nominal protocol config differs')\n"
         "    protocol_config = json.loads(protocol_path.read_bytes())\n"
         "    validate_formal_variant(protocol_config)"),
        ("    seed = int(value('seed'))",
         "    if value('protocol_id') != PROTOCOL_ID:\n"
         "        raise ValueError('task protocol parameter differs')\n"
         "    seed = int(value('seed'))"),
        ("            membership_sha256=MEMBERSHIP_SHA,",
         "            membership_sha256=MEMBERSHIP_SHA,\n"
         "            protocol_config_sha256=PROTOCOL_CONFIG_SHA,"),
    )
    for before, after in edits:
        if text.count(before) != 1:
            raise ValueError(f'expected one source occurrence: {before!r}')
        text = text.replace(before, after)
    files[RUNNER_NAME] = text.encode()
    files[VALIDATOR_NAME] = validator.read_bytes()
    files[CONFIG_NAME] = config.read_bytes()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('xb') as raw:
        with gzip.GzipFile(filename='', mode='wb', fileobj=raw, compresslevel=9, mtime=0) as gz:
            with tarfile.open(fileobj=gz, mode='w') as archive:
                for name, payload in sorted(files.items()):
                    info = tarfile.TarInfo(name)
                    info.size = len(payload)
                    info.mode = 0o644
                    info.mtime = info.uid = info.gid = 0
                    info.uname = info.gname = ''
                    archive.addfile(info, io.BytesIO(payload))
    return {'source_sha256': sha(output.read_bytes()),
            'runner_sha256': sha(files[RUNNER_NAME]),
            'validator_sha256': sha(files[VALIDATOR_NAME]),
            'protocol_config_sha256': sha(files[CONFIG_NAME]),
            'member_count': len(files)}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base', required=True, type=Path)
    p.add_argument('--validator', required=True, type=Path)
    p.add_argument('--config', required=True, type=Path)
    p.add_argument('--output', required=True, type=Path)
    a = p.parse_args()
    print(json.dumps(upgrade(a.base, a.validator, a.config, a.output), sort_keys=True))


if __name__ == '__main__':
    main()
