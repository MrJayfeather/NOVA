import asyncio
import subprocess
import time
from pathlib import Path


def _git(root: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["git", "-C", str(root), *args],
                          capture_output=True, text=True)


class MemorySync:
    """Память в git: пуш после каждого обмена (дебаунс), пулл на старте.
    Любая ошибка сети — печать и продолжаем жить: память локально цела."""

    def __init__(self, root: Path, remote: str | None = None):
        self._root = Path(root)
        self._remote = remote
        self._poke = asyncio.Event()
        self._last_push = 0.0

    def ensure_repo(self) -> None:
        if not (self._root / ".git").exists():
            _git(self._root, "init")
        _git(self._root, "config", "user.name", "NOVA")
        _git(self._root, "config", "user.email", "nova@local")
        if self._remote:
            _git(self._root, "remote", "remove", "origin")
            _git(self._root, "remote", "add", "origin", self._remote)
            _git(self._root, "pull", "--rebase", "origin", "master")

    def push_now(self, msg: str = "mem") -> bool:
        _git(self._root, "add", "-A")
        _git(self._root, "commit", "-m", msg)
        # remote мог приехать и с клоном (onstart) — параметр не обязателен
        has_remote = self._remote or _git(self._root, "remote").stdout.strip()
        if not has_remote:
            return False
        _git(self._root, "pull", "--rebase", "origin", "master")
        r = _git(self._root, "push", "origin", "HEAD:master")
        if r.returncode != 0:
            print(f"[nova] память: пуш не прошёл ({r.stderr[:80]!r}) — позже")
            return False
        return True

    def request_push(self) -> None:
        self._poke.set()

    async def pusher_loop(self, min_interval_s: float = 30.0) -> None:
        while True:
            await self._poke.wait()
            self._poke.clear()
            wait = self._last_push + min_interval_s - time.time()
            if wait > 0:
                await asyncio.sleep(wait)
            self._last_push = time.time()
            await asyncio.to_thread(self.push_now)
