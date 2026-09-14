"""Migration scope, recoverability and fail-before-removal checks."""
import json
from pathlib import Path
import sys

import pytest

from tools.event_track_v2x import migrate_thesis_code as migration
from tools.event_track_v2x.test_migrated_thesis import regular_file


def tree(tmp_path):
    root = tmp_path / 'paper'
    (root / '.git').mkdir(parents=True)
    contents = {
        'experiments/rtp_v2x/model/core.py': 'research source\n',
        'experiments/clearml/run.sh': '#!/bin/sh\nexit 0\n',
        'tests/test_publish_task.py': 'research test\n',
        'scripts/render_results.py': 'build tool\n',
        'experiments/rtp_v2x/provenance_seals.py': 'shared build dependency\n',
        'experiments/rtp_v2x/tests/test_provenance_seals.py': 'build test\n',
        'experiments/rtp_v2x/__pycache__/core.pyc': 'cache\n',
        'evidence/run/tool.py': 'historical evidence source\n',
        'Makefile': 'all:\n', 'latexmkrc': 'latex config\n',
        'main.tex': 'paper text\n', 'experiments/clearml/protocols/test.json': '{}\n',
    }
    for name, body in contents.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(body)
    (root / 'experiments/clearml/run.sh').chmod(0o755)
    return root, contents


def invoke(monkeypatch, root, archive, *, apply=True):
    monkeypatch.setattr(sys, 'argv', ['migration', '--source', str(root), '--archive', str(archive),
                                    '--receipt', str(root / 'docs/receipt.json')] + (['--apply'] if apply else []))
    monkeypatch.setattr(migration.subprocess, 'check_output', lambda *a, **kw: 'a' * 40)
    migration.main()


def test_exact_scope_moves_only_research_and_preserves_recovery(tmp_path, monkeypatch):
    root, contents = tree(tmp_path)
    archive = tmp_path / 'code/archive'
    before = migration.inventory(root)
    assert len(before) == 3
    invoke(monkeypatch, root, archive)
    receipt = json.loads((root / 'docs/receipt.json').read_text())
    assert receipt['status'] == 'migrated' and receipt['recoverable']
    for name, body in contents.items():
        if migration.is_research(Path(name)):
            assert not (root / name).exists()
            assert (archive / name).read_text() == body
        else:
            assert (root / name).read_text() == body
    assert (archive / 'experiments/clearml/run.sh').stat().st_mode & 0o777 == 0o755
    assert migration.inventory(root) == []
    assert json.loads((archive / 'copies-verified.json').read_text())['manifest_sha256'] == (
        migration.sha(archive / 'migration-manifest.json'))


def test_dry_run_never_creates_archive_or_removes_sources(tmp_path, monkeypatch):
    root, _ = tree(tmp_path)
    before = migration.inventory(root)
    archive = tmp_path / 'archive'
    invoke(monkeypatch, root, archive, apply=False)
    assert migration.inventory(root) == before and not archive.exists()
    assert not (root / 'docs/receipt.json').exists()


def test_corrupt_copy_stops_before_any_original_removal(tmp_path, monkeypatch):
    root, _ = tree(tmp_path)
    before = migration.inventory(root)
    real_copy = migration.shutil.copyfileobj
    def broken(src, dst):
        real_copy(src, dst)
        dst.write(b'corruption')
    monkeypatch.setattr(migration.shutil, 'copyfileobj', broken)
    with pytest.raises(ValueError, match='mismatch'):
        invoke(monkeypatch, root, tmp_path / 'archive')
    assert migration.inventory(root) == before


def test_existing_archive_never_overwritten(tmp_path, monkeypatch):
    root, _ = tree(tmp_path)
    before = migration.inventory(root)
    archive = tmp_path / 'archive'
    archive.mkdir()
    (archive / 'keep').write_text('keep')
    with pytest.raises(SystemExit):
        invoke(monkeypatch, root, archive)
    assert (archive / 'keep').read_text() == 'keep'
    assert migration.inventory(root) == before


def test_research_symlink_is_rejected_without_touching_target(tmp_path):
    root, _ = tree(tmp_path)
    external = tmp_path / 'external.py'
    external.write_text('external')
    (root / 'experiments/linked.py').symlink_to(external)
    with pytest.raises(ValueError, match='symlink'):
        migration.inventory(root)
    assert external.read_text() == 'external'


@pytest.mark.parametrize('relative', ['/absolute.py', '../outside.py', 'link.py', 'missing.py'])
def test_compatibility_snapshot_rejects_unsafe_or_missing_inputs(tmp_path, relative):
    target = tmp_path / 'safe.py'
    target.write_text('safe')
    (tmp_path / 'link.py').symlink_to(target)
    with pytest.raises(ValueError):
        regular_file(tmp_path, relative)
    assert regular_file(tmp_path, 'safe.py') == target
