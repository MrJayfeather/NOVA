import subprocess
from pathlib import Path

from nova.server.memory.sync import MemorySync


def make_remote(tmp_path) -> Path:
    bare = tmp_path / "remote.git"
    subprocess.run(["git", "init", "--bare", str(bare)], check=True,
                   capture_output=True)
    return bare


def test_ensure_repo_and_push(tmp_path):
    bare = make_remote(tmp_path)
    root = tmp_path / "mem"
    root.mkdir()
    sync = MemorySync(root, remote=str(bare))
    sync.ensure_repo()
    (root / "digest.md").write_text("конспект", encoding="utf-8")
    assert sync.push_now()
    log = subprocess.run(["git", "log", "--oneline"], cwd=bare,
                         capture_output=True, text=True).stdout
    assert "mem" in log


def test_push_now_no_changes_ok(tmp_path):
    bare = make_remote(tmp_path)
    root = tmp_path / "mem"
    root.mkdir()
    sync = MemorySync(root, remote=str(bare))
    sync.ensure_repo()
    assert sync.push_now() in (True, False)   # пусто — не падает


def test_push_survives_no_remote(tmp_path):
    root = tmp_path / "mem"
    root.mkdir()
    sync = MemorySync(root, remote=None)
    sync.ensure_repo()
    (root / "x.md").write_text("а", encoding="utf-8")
    assert sync.push_now() is False            # некуда, но без исключений
