import difflib
import re
import time
from datetime import datetime
from pathlib import Path


def clip_to_tokens(text: str, max_tokens: int) -> str:
    """Жёсткий бюджет: токены ≈ символы/3 (русский). Режем СНИЗУ —
    в digest свежее сверху, старьё уходит первым (оно есть в дневнике)."""
    limit = max_tokens * 3
    if len(text) <= limit:
        return text
    cut = text[:limit]
    nl = cut.rfind("\n")
    return cut[: nl if nl > 0 else limit]


def _norm(s: str) -> str:
    return re.sub(r"[^\wёа-яЁА-Я ]", "", s.lower()).strip()


class MemoryStore:
    """Файловая память: diary/*.md (append-only), digest.md, facts.md,
    index.md, .cursor. Никаких нейронок — только диск."""

    DIGEST_TOKENS = 2500
    FACTS_TOKENS = 1500

    def __init__(self, root: Path):
        self.root = Path(root)
        (self.root / "diary").mkdir(parents=True, exist_ok=True)
        self._last_seen: str = ""
        self._last_seen_day: str = ""
        self._last_seen_offset: int | None = None
        self._last_seen_start: str = ""

    # ---- дневник ----

    def _day_file(self, ts: float) -> tuple[Path, str, str]:
        dt = datetime.fromtimestamp(ts)
        day = dt.strftime("%Y-%m-%d")
        return self.root / "diary" / f"{day}.md", day, dt.strftime("%H:%M")

    def _append_line(self, ts: float, line: str) -> None:
        path, _, _ = self._day_file(ts)
        with path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")

    def append_reply(self, who: str, text: str, ts: float | None = None) -> None:
        ts = ts or time.time()
        _, _, hm = self._day_file(ts)
        self._append_line(ts, f"{hm} [{who}] {text.strip()}")
        self._last_seen_offset = None  # между [видела] вклинилась реплика

    def append_event(self, text: str, ts: float | None = None) -> None:
        ts = ts or time.time()
        _, _, hm = self._day_file(ts)
        self._append_line(ts, f"{hm} [событие] {text.strip()}")
        self._last_seen_offset = None

    def append_seen(self, text: str, ts: float | None = None) -> bool:
        """Летопись глаз с дедупом: почти то же описание -> обновляем
        конец интервала времени ПОСЛЕДНЕЙ [видела]-строки."""
        ts = ts or time.time()
        path, day, hm = self._day_file(ts)
        text = text.strip()
        same = (
            self._last_seen
            and day == self._last_seen_day
            and difflib.SequenceMatcher(
                None, _norm(text), _norm(self._last_seen)).ratio() >= 0.82
        )
        if same and self._last_seen_offset is not None and path.exists():
            with path.open("r+", encoding="utf-8") as f:
                f.seek(self._last_seen_offset)
                f.truncate()
                f.write(f"{self._last_seen_start}-{hm} [видела] "
                        f"{self._last_seen}\n")
            return False
        with path.open("a", encoding="utf-8") as f:
            self._last_seen_offset = f.tell()
            f.write(f"{hm} [видела] {text}\n")
        self._last_seen = text
        self._last_seen_day = day
        self._last_seen_start = hm
        return True

    def diary_days(self) -> list[str]:
        return sorted(p.stem for p in (self.root / "diary").glob("*.md"))

    def read_day(self, day: str) -> str:
        p = self.root / "diary" / f"{day}.md"
        return p.read_text(encoding="utf-8") if p.exists() else ""

    # ---- digest / facts / index ----

    def _read(self, name: str) -> str:
        p = self.root / name
        return p.read_text(encoding="utf-8") if p.exists() else ""

    def read_digest(self) -> str:
        return self._read("digest.md")

    def read_facts(self) -> str:
        return self._read("facts.md")

    def read_index(self) -> str:
        return self._read("index.md")

    def write_digest(self, text: str) -> None:
        (self.root / "digest.md").write_text(
            clip_to_tokens(text.strip(), self.DIGEST_TOKENS), encoding="utf-8")

    def write_facts(self, text: str) -> None:
        (self.root / "facts.md").write_text(
            clip_to_tokens(text.strip(), self.FACTS_TOKENS), encoding="utf-8")

    def set_index_line(self, day: str, line: str) -> None:
        lines = [l for l in self.read_index().splitlines()
                 if l.strip() and not l.startswith(day)]
        lines.append(line.strip())
        (self.root / "index.md").write_text(
            "\n".join(sorted(lines)) + "\n", encoding="utf-8")

    def system_context(self) -> str:
        parts = []
        if self.read_digest():
            parts.append("ТВОЯ ПАМЯТЬ О ПОСЛЕДНИХ ДНЯХ:\n" + self.read_digest())
        if self.read_facts():
            parts.append("ЧТО ТЫ ЗНАЕШЬ (факты):\n" + self.read_facts())
        return "\n\n".join(parts)

    # ---- курсор конденсера ----

    def cursor(self) -> str:
        return self._read(".cursor").strip()

    def set_cursor(self, mark: str) -> None:
        (self.root / ".cursor").write_text(mark, encoding="utf-8")

    def unprocessed_tail(self, max_chars: int = 20000) -> tuple[str, str]:
        """Дневник после курсора. Курсор = «day:номер_строки»."""
        cur_day, _, cur_line = self.cursor().partition(":")
        cur_n = int(cur_line) if cur_line.isdigit() else 0
        out: list[str] = []
        last_day, last_n = cur_day, cur_n
        for day in self.diary_days():
            if cur_day and day < cur_day:
                continue
            lines = self.read_day(day).splitlines()
            start = cur_n if day == cur_day else 0
            for i, line in enumerate(lines[start:], start=start):
                out.append(f"{day} {line}")
                last_day, last_n = day, i + 1
            if sum(len(l) for l in out) > max_chars:
                break
        return "\n".join(out), f"{last_day}:{last_n}"
