# План 3А: Память NOVA

> Спека: docs/specs/2026-07-05-nova-memory-design.md. Задачи с чекбоксами,
> каждая — свой тест-цикл и коммит. Ветка memory3a от master.

**Цель:** бессмертная память NOVA — дневник-плёнка в git, конспект в голове,
картотека фактов с повадками, четырёхуровневое вспоминание (включая
«эффект Светки»), летопись экрана по пульсу 10с, конденсер в паузах мозга.

**Архитектура:** пакет nova/server/memory (store — файлы и дедуп, recall —
уровни 1/1.5, condenser — сжатие Qwen'ом, sync — git). Оркестратор пишет
дневник и вставляет воспоминания; GeminiEyes отдаёт описания через хук
on_seen; QwenVLM получает context_provider (digest+facts в системный
промпт) и complete() для конденсера. Хранилище — приватный репо
nova-memory, клонится в onstart, пушится после каждого обмена.

**Стек:** stdlib (pathlib, difflib, datetime, re, asyncio), git CLI.
Никаких новых pip-зависимостей.

## Глобальные ограничения

- НИКАКИХ упоминаний Claude/Anthropic/AI-инструментов в коде/коммитах.
- Ветка memory3a от master; merge без PR; инстанс видит только master.
- Русские комментарии, стиль кода как в nova/server/models/*.
- `uv run pytest -q` до начала: 131 passed, 2 skipped; после каждой
  задачи — столько же + новые, ноль красных.
- Дневник append-only: никакая операция не удаляет и не переписывает
  прошлые записи (единственное исключение — схлопывание времени
  ПОСЛЕДНЕЙ [видела]-строки при дедупе).
- Бюджеты: digest ~2500 ток, facts ~1500 ток, recall-вставка ~1500 ток
  (токены ≈ символы/3 для русского — грубая, но достаточная оценка).
- Крутилки env: NOVA_MEMORY (1|0), NOVA_MEMORY_DIR, NOVA_CHRONICLE_S=10,
  NOVA_CONDENSE_IDLE_S=600, NOVA_MEMORY_LLM (brain|gemini),
  NOVA_RECALL_COOLDOWN_D=5, NOVA_MEMORY_TOKEN, NOVA_MEMORY_REPO.
- Секреты: токен только в .env и /workspace/memory_token.

## Карта файлов

| Файл | Роль |
|---|---|
| nova/server/memory/__init__.py (новый) | пустой |
| nova/server/memory/store.py (новый) | дневник/digest/facts/index/.cursor, дедуп [видела], бюджеты |
| nova/server/memory/recall.py (новый) | триггеры, парс дат, выбор дней, окна; ассоциации + .recalled |
| nova/server/memory/condenser.py (новый) | промпт/парсер конденсера, раннер с прерыванием |
| nova/server/memory/sync.py (новый) | git pull/push, дебаунс 30с |
| nova/server/models/qwen_llm.py | + context_provider, + complete() |
| nova/server/models/gemini_vision.py | + on_seen-хук, + complete_text() |
| nova/server/orchestrator.py | дневник, recall/assoc-вставки, пульс летописи |
| nova/server/main.py | сборка memory, конденсер-цикл, [событие], пуш |
| deploy/onstart.sh, deploy/runner.sh | клон nova-memory, токен, env |
| scripts/vast.py, NOVA_START.bat | проброс env, pull на ноут |
| tests/test_memory_store.py и др. (новые) | по модулю на файл |

---

### Задача 1: MemoryStore — файлы, дедуп, курсор

**Файлы:** создать `nova/server/memory/__init__.py` (пустой),
`nova/server/memory/store.py`, `tests/test_memory_store.py`.

**Производит:**
`MemoryStore(root: Path)`;
`append_reply(who: str, text: str, ts: float | None = None)`;
`append_seen(text, ts=None) -> bool` (False = дедуп-схлопывание);
`append_event(text, ts=None)`;
`read_digest()/read_facts()/read_index() -> str` («» если нет);
`write_digest(text)/write_facts(text)` (с жёстким клипом бюджета);
`set_index_line(day: str, line: str)` (day «YYYY-MM-DD», замена/добавление);
`cursor() -> str` / `set_cursor(mark: str)` (файл .cursor);
`unprocessed_tail(max_chars=20000) -> tuple[str, str]` (текст, новый курсор);
`diary_days() -> list[str]`; `read_day(day: str) -> str`;
`system_context() -> str` (digest+facts блоком);
`clip_to_tokens(text, max_tokens) -> str` (модульная функция, режет
СТАРОЕ: у digest старое внизу — режем снизу).

- [ ] **Шаг 1.1: тесты (падают)** — `tests/test_memory_store.py`:

```python
import time
from pathlib import Path

from nova.server.memory.store import MemoryStore, clip_to_tokens


def make_store(tmp_path) -> MemoryStore:
    return MemoryStore(tmp_path)


def test_append_reply_writes_dated_line(tmp_path):
    st = make_store(tmp_path)
    ts = time.mktime((2026, 7, 5, 21, 14, 0, 0, 0, -1))
    st.append_reply("Джей", "Смотри, какой анлак!", ts=ts)
    st.append_reply("NOVA", "Это дар.", ts=ts + 30)
    day = (tmp_path / "diary" / "2026-07-05.md").read_text(encoding="utf-8")
    assert "21:14 [Джей] Смотри, какой анлак!" in day
    assert "21:14 [NOVA] Это дар." in day


def test_append_seen_dedup_collapses_time(tmp_path):
    st = make_store(tmp_path)
    t0 = time.mktime((2026, 7, 5, 21, 10, 0, 0, 0, -1))
    assert st.append_seen("Джей читает доку про Rivals", ts=t0)
    # почти то же описание минутой позже — не новая строка,
    # а обновление времени последней
    assert not st.append_seen("Джей читает доку про Rivals.", ts=t0 + 60)
    day = (tmp_path / "diary" / "2026-07-05.md").read_text(encoding="utf-8")
    assert day.count("[видела]") == 1
    assert "21:10-21:11 [видела]" in day
    # существенно другое — новая строка
    assert st.append_seen("Матч начался, счёт 0:0", ts=t0 + 120)
    day = (tmp_path / "diary" / "2026-07-05.md").read_text(encoding="utf-8")
    assert day.count("[видела]") == 2


def test_append_event_and_day_rollover(tmp_path):
    st = make_store(tmp_path)
    t1 = time.mktime((2026, 7, 5, 23, 59, 0, 0, 0, -1))
    t2 = time.mktime((2026, 7, 6, 0, 1, 0, 0, 0, -1))
    st.append_event("клиент подключился", ts=t1)
    st.append_event("клиент отключился", ts=t2)
    assert (tmp_path / "diary" / "2026-07-05.md").exists()
    assert (tmp_path / "diary" / "2026-07-06.md").exists()
    assert st.diary_days() == ["2026-07-05", "2026-07-06"]


def test_digest_facts_roundtrip_and_budget(tmp_path):
    st = make_store(tmp_path)
    assert st.read_digest() == ""
    st.write_digest("СЕГОДНЯ: дела.\n" + "старьё\n" * 5000)
    saved = st.read_digest()
    assert saved.startswith("СЕГОДНЯ")            # свежее сверху уцелело
    assert len(saved) <= 2500 * 3 + 100           # клип по бюджету
    st.write_facts("## Люди\n- Джей любит Rivals")
    assert "Rivals" in st.read_facts()
    assert "Джей любит Rivals" in st.system_context()


def test_index_line_replace(tmp_path):
    st = make_store(tmp_path)
    st.set_index_line("2026-07-05", "2026-07-05: черновик")
    st.set_index_line("2026-07-05", "2026-07-05 ★: турнир | сущности: Rivals")
    idx = st.read_index()
    assert idx.count("2026-07-05") == 1
    assert "★" in idx


def test_cursor_and_unprocessed_tail(tmp_path):
    st = make_store(tmp_path)
    t0 = time.mktime((2026, 7, 5, 12, 0, 0, 0, 0, -1))
    st.append_reply("Джей", "раз", ts=t0)
    st.append_reply("NOVA", "два", ts=t0 + 5)
    tail, new_cursor = st.unprocessed_tail()
    assert "раз" in tail and "два" in tail
    st.set_cursor(new_cursor)
    st.append_reply("Джей", "три", ts=t0 + 10)
    tail2, _ = st.unprocessed_tail()
    assert "три" in tail2 and "раз" not in tail2


def test_clip_to_tokens_cuts_bottom():
    text = "верх\n" + "\n".join(f"строка {i}" for i in range(1000))
    out = clip_to_tokens(text, 50)
    assert out.startswith("верх")
    assert len(out) <= 50 * 3 + 20
```

- [ ] **Шаг 1.2:** `uv run pytest tests/test_memory_store.py -q` → FAIL
  (ModuleNotFoundError).

- [ ] **Шаг 1.3: реализация** — `nova/server/memory/store.py`:

```python
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
```

- [ ] **Шаг 1.4:** `uv run pytest tests/test_memory_store.py -q` → passed.

- [ ] **Шаг 1.5:**

```bash
git add nova/server/memory tests/test_memory_store.py
git commit -m "feat: memory store - diary, digest, facts, index, cursor"
```

---

### Задача 2: recall уровень 1 — вспоминание по вопросу

**Файлы:** создать `nova/server/memory/recall.py`,
`tests/test_memory_recall.py`.

**Потребляет:** MemoryStore (read_index, diary_days, read_day).

**Производит:**
`wants_recall(text) -> bool`;
`parse_period(text, today: str) -> tuple[str, str] | None` (даты YYYY-MM-DD);
`keywords(text) -> list[str]`;
`pick_days(index_text, keys, period, limit=3) -> list[str]`;
`extract_windows(day_text, keys, radius=5, max_chars=4000) -> str`;
`recall(store, question, today) -> str` («» если нечего).

- [ ] **Шаг 2.1: тесты (падают)** — `tests/test_memory_recall.py`:

```python
from nova.server.memory.recall import (
    extract_windows, keywords, parse_period, pick_days, recall, wants_recall,
)
from nova.server.memory.store import MemoryStore

TODAY = "2026-07-05"


def test_wants_recall_triggers():
    assert wants_recall("а помнишь, была ситуация?")
    assert wants_recall("вспомни, что тогда было")
    assert wants_recall("мы же на прошлой неделе смотрели")
    assert wants_recall("что было вчера?")
    assert not wants_recall("как дела?")


def test_parse_period_relative():
    assert parse_period("что было вчера", TODAY) == ("2026-07-04", "2026-07-04")
    assert parse_period("неделю назад", TODAY) == ("2026-06-24", "2026-07-01")
    # «1-2 недели назад» — объединённый диапазон
    assert parse_period("недели две назад был анлак", TODAY) == (
        "2026-06-14", "2026-06-28")
    assert parse_period("месяц назад", TODAY) == ("2026-05-29", "2026-06-12")
    assert parse_period("просто помнишь?", TODAY) is None


def test_keywords_drops_stopwords():
    keys = keywords("А помнишь как Джефф съел троих в турнире?")
    assert "джефф" in keys and "турнире" in keys
    assert "как" not in keys and "троих" in keys


def test_pick_days_by_keys_and_period():
    index = "\n".join([
        "2026-06-20 ★: турнир Rivals, Джефф съел троих | сущности: Rivals, Джефф",
        "2026-06-25: чинили голос | сущности: голос",
        "2026-07-01: болтали про аниме | сущности: аниме",
    ])
    days = pick_days(index, ["джефф", "анлак"], None)
    assert days == ["2026-06-20"]
    # период сужает даже без совпадения слов
    days = pick_days(index, ["ерунда"], ("2026-06-30", "2026-07-02"))
    assert days == ["2026-07-01"]


def test_extract_windows_around_matches():
    day = "\n".join(f"21:{i:02d} [видела] строка {i}" for i in range(40))
    day = day.replace("строка 20", "Джефф съел троих")
    out = extract_windows(day, ["джефф"], radius=2)
    assert "Джефф съел троих" in out
    assert "строка 17" not in out and "строка 23" not in out
    assert "строка 18" in out and "строка 22" in out


def test_recall_end_to_end(tmp_path):
    st = MemoryStore(tmp_path)
    import time
    ts = time.mktime((2026, 6, 20, 21, 14, 0, 0, 0, -1))
    st.append_seen("Турнир Rivals: Джефф съел троих, счёт 0:1", ts=ts)
    st.append_reply("Джей", "Смотри, какой анлак!", ts=ts + 10)
    st.set_index_line("2026-06-20",
                      "2026-06-20 ★: турнир, Джефф съел троих | сущности: Джефф")
    out = recall(st, "помнишь, недели две назад Джефф съел троих?", TODAY)
    assert "[из дневника за 2026-06-20" in out
    assert "анлак" in out


def test_recall_empty_when_nothing(tmp_path):
    st = MemoryStore(tmp_path)
    assert recall(st, "помнишь про козявок?", TODAY) == ""
```

- [ ] **Шаг 2.2:** `uv run pytest tests/test_memory_recall.py -q` → FAIL.

- [ ] **Шаг 2.3: реализация** — `nova/server/memory/recall.py`:

```python
import difflib
import re
from datetime import date, timedelta

from nova.server.memory.store import MemoryStore

TRIGGERS = ("помнишь", "вспомни", "тогда", "в прошлый раз", "недавно",
            "на прошлой неделе", "назад", "вчера", "позавчера", "было")

_STOP = {"а", "и", "в", "на", "как", "что", "это", "ты", "я", "мы", "же",
         "не", "ну", "у", "с", "к", "по", "за", "из", "был", "была",
         "было", "были", "помнишь", "вспомни", "тогда", "недавно"}


def wants_recall(text: str) -> bool:
    t = text.lower()
    return any(w in t for w in TRIGGERS)


def _d(s: str) -> date:
    return date.fromisoformat(s)


def parse_period(text: str, today: str) -> tuple[str, str] | None:
    """«вчера», «N дней/недель/месяцев назад» -> диапазон дат с запасом
    (люди помнят время неточно — берём широкое окно)."""
    t = text.lower()
    base = _d(today)
    if "позавчера" in t:
        d = base - timedelta(days=2)
        return str(d), str(d)
    if "вчера" in t:
        d = base - timedelta(days=1)
        return str(d), str(d)
    nums = {"один": 1, "одну": 1, "два": 2, "две": 2, "три": 3,
            "четыре": 4, "пять": 5, "пару": 2}

    def amount(m: re.Match) -> int:
        w = m.group(1)
        return int(w) if w.isdigit() else nums.get(w, 1)

    m = re.search(r"(\d+|неделю|недел\w*|" + "|".join(nums) + r")?\s*недел\w*\s*назад", t)
    if m or "на прошлой неделе" in t:
        n = amount(m) if m and m.group(1) else 1
        centre = base - timedelta(weeks=n)
        return str(centre - timedelta(days=7 * max(1, n) // 1)), \
               str(centre + timedelta(days=7))
    m = re.search(r"(\d+|" + "|".join(nums) + r")?\s*месяц\w*\s*назад", t)
    if m:
        n = amount(m) if m.group(1) else 1
        centre = base - timedelta(days=30 * n)
        return str(centre - timedelta(days=7)), str(centre + timedelta(days=7))
    m = re.search(r"(\d+|" + "|".join(nums) + r")?\s*(дня|дней|день)\s*назад", t)
    if m:
        n = amount(m) if m.group(1) else 1
        centre = base - timedelta(days=n)
        return str(centre - timedelta(days=1)), str(centre + timedelta(days=1))
    return None


def keywords(text: str) -> list[str]:
    words = re.findall(r"[\wёЁа-яА-Я]{4,}", text.lower())
    return [w for w in words if w not in _STOP]


def _match(word: str, line: str) -> bool:
    lw = line.lower()
    if word in lw:
        return True
    # морфология: «джефф/джеффа/джеффом» — нечётко по словам строки
    return any(difflib.SequenceMatcher(None, word, w).ratio() >= 0.75
               for w in re.findall(r"[\wёа-я]{4,}", lw))


def pick_days(index_text: str, keys: list[str],
              period: tuple[str, str] | None, limit: int = 3) -> list[str]:
    scored: list[tuple[float, str]] = []
    for line in index_text.splitlines():
        m = re.match(r"(\d{4}-\d{2}-\d{2})", line.strip())
        if not m:
            continue
        day = m.group(1)
        in_period = period and period[0] <= day <= period[1]
        hits = sum(1 for k in keys if _match(k, line))
        score = hits * 2 + (1.5 if in_period else 0) + (0.5 if "★" in line else 0)
        if period and not in_period and hits == 0:
            continue
        if score > 0:
            scored.append((score, day))
    scored.sort(reverse=True)
    return [d for _, d in scored[:limit]]


def extract_windows(day_text: str, keys: list[str], radius: int = 5,
                    max_chars: int = 4000) -> str:
    lines = day_text.splitlines()
    hit_rows = [i for i, l in enumerate(lines)
                if any(_match(k, l) for k in keys)]
    take: set[int] = set()
    for i in hit_rows:
        take.update(range(max(0, i - radius), min(len(lines), i + radius + 1)))
    out: list[str] = []
    total = 0
    for i in sorted(take):
        total += len(lines[i])
        if total > max_chars:
            break
        out.append(lines[i])
    return "\n".join(out)


def recall(store: MemoryStore, question: str, today: str) -> str:
    keys = keywords(question)
    period = parse_period(question, today)
    days = pick_days(store.read_index(), keys, period)
    if not days and period:
        # оглавление могло не успеть — берём дни диапазона напрямую
        days = [d for d in store.diary_days()
                if period[0] <= d <= period[1]][-3:]
    chunks = []
    for day in days:
        got = extract_windows(store.read_day(day), keys)
        if got:
            chunks.append(f"[из дневника за {day}:\n{got}]")
    return "\n".join(chunks)
```

- [ ] **Шаг 2.4:** `uv run pytest tests/test_memory_recall.py -q` → passed.
  ВНИМАНИЕ: парс дат — самое хрупкое место; если тест диапазонов упал,
  правь регэкспы до зелёного, ожидания теста не менять (они из спеки:
  окна с запасом).

- [ ] **Шаг 2.5:**

```bash
git add nova/server/memory/recall.py tests/test_memory_recall.py
git commit -m "feat: recall level 1 - dates, day picking, diary windows"
```

---

### Задача 3: recall уровень 1.5 — эффект Светки

**Файлы:** дополнить `nova/server/memory/recall.py`,
`tests/test_memory_recall.py`.

**Производит:** `associate(store, context_keys: list[str], today: str,
min_age_days=3, cooldown_days=5, min_score=4.0) -> str` («» если нет
сильного кандидата). Журнал `.recalled`: строки «YYYY-MM-DD day», где
первая дата — когда предложено, вторая — какой день дневника.

- [ ] **Шаг 3.1: тесты (падают)** — добавить в `tests/test_memory_recall.py`:

```python
def _assoc_store(tmp_path):
    import time
    st = MemoryStore(tmp_path)
    ts = time.mktime((2026, 3, 1, 20, 0, 0, 0, 0, -1))
    st.append_seen("Игра X: NPC Барни ляпнул смешную фразу про сыр", ts=ts)
    st.append_reply("Джей", "ахахах этот Барни", ts=ts + 5)
    st.set_index_line("2026-03-01",
                      "2026-03-01 ★: игра X, NPC Барни и сыр | сущности: X, Барни")
    st.set_index_line("2026-07-04",
                      "2026-07-04: вчера болтали | сущности: разное")
    return st


def test_associate_finds_old_bright_day(tmp_path):
    from nova.server.memory.recall import associate

    st = _assoc_store(tmp_path)
    out = associate(st, ["барни", "игра"], TODAY)
    assert "2026-03-01" in out
    assert "если к месту" in out          # предложение, не приказ


def test_associate_respects_cooldown(tmp_path):
    from nova.server.memory.recall import associate

    st = _assoc_store(tmp_path)
    assert associate(st, ["барни"], TODAY)          # первый раз — нашла
    assert associate(st, ["барни"], TODAY) == ""    # повтор — молчим


def test_associate_skips_recent_days(tmp_path):
    from nova.server.memory.recall import associate

    st = _assoc_store(tmp_path)
    # вчерашний день не годится для «а помнишь» — слишком свежо
    assert associate(st, ["болтали", "разное"], TODAY) == ""


def test_associate_needs_strong_match(tmp_path):
    from nova.server.memory.recall import associate

    st = _assoc_store(tmp_path)
    assert associate(st, ["случайное"], TODAY) == ""
```

- [ ] **Шаг 3.2:** `uv run pytest tests/test_memory_recall.py -q` → FAIL.

- [ ] **Шаг 3.3: реализация** — дополнить recall.py:

```python
def associate(store: MemoryStore, context_keys: list[str], today: str,
              min_age_days: int = 3, cooldown_days: int = 5,
              min_score: float = 4.0) -> str:
    """Эффект Светки: текущий момент ассоциативно цепляет старый день.
    Возвращает ПРЕДЛОЖЕНИЕ мозгу (не приказ) или «»."""
    if not context_keys:
        return ""
    recalled_path = store.root / ".recalled"
    recalled = recalled_path.read_text(encoding="utf-8").splitlines() \
        if recalled_path.exists() else []
    base = _d(today)
    best: tuple[float, str] | None = None
    for line in store.read_index().splitlines():
        m = re.match(r"(\d{4}-\d{2}-\d{2})", line.strip())
        if not m:
            continue
        day = m.group(1)
        age = (base - _d(day)).days
        if age < min_age_days:
            continue  # свежее и так в конспекте — не «а помнишь»
        if any(day in r and (base - _d(r[:10])).days < cooldown_days
               for r in recalled):
            continue  # бабушкина история — уже вспоминала недавно
        hits = sum(1 for k in context_keys if _match(k, line))
        if hits < 2:
            continue
        score = hits * 2 + (1.0 if "★" in line else 0) + min(age / 30, 2.0)
        if score >= min_score and (best is None or score > best[0]):
            best = (score, day)
    if best is None:
        return ""
    day = best[1]
    got = extract_windows(store.read_day(day), context_keys)
    if not got:
        return ""
    with recalled_path.open("a", encoding="utf-8") as f:
        f.write(f"{today} {day}\n")
    return (f"[из дневника за {day} — если к месту, можешь сама это "
            f"вспомнить, но не обязана:\n{got}]")
```

- [ ] **Шаг 3.4:** `uv run pytest tests/test_memory_recall.py -q` → passed.

- [ ] **Шаг 3.5:**

```bash
git add nova/server/memory/recall.py tests/test_memory_recall.py
git commit -m "feat: recall level 1.5 - associative memory with repeat guard"
```

---

### Задача 4: конденсер + QwenVLM.complete

**Файлы:** создать `nova/server/memory/condenser.py`,
`tests/test_memory_condenser.py`; изменить
`nova/server/models/qwen_llm.py`, `tests/test_qwen_llm.py`.

**Потребляет:** MemoryStore (unprocessed_tail, write_digest, write_facts,
set_index_line, set_cursor, read_digest, read_facts).

**Производит:** `build_prompt(digest, facts, tail, today) -> str`;
`parse_output(text) -> tuple[str, str, str] | None` (digest, facts,
index_line); `class Condenser(store, chat, idle_s=600)`:
`should_run(last_activity: float, now: float) -> bool`,
`async run_once(today: str) -> bool`, `interrupt()`;
`QwenVLM.complete(system: str, user: str, max_tokens=1200) -> str`
(async, temperature 0.3, без штрафов и персоны).

- [ ] **Шаг 4.1: тесты (падают)** — `tests/test_memory_condenser.py`:

```python
import asyncio
import time

from nova.server.memory.condenser import Condenser, build_prompt, parse_output
from nova.server.memory.store import MemoryStore

OUT = """===DIGEST===
СЕГОДНЯ (2026-07-05): турнир, Джефф съел троих, шутка про склероз.
===FACTS===
## Люди
- Джей любит Rivals (с 05.07.2026)
## Повадки и стили
- Джей играет пацифистом (с 05.07.2026)
===INDEX===
2026-07-05 ★: турнир, Джефф | сущности: Rivals, Джефф
"""


def test_build_prompt_mentions_budgets_and_habits():
    p = build_prompt("старый конспект", "старые факты", "21:14 [Джей] хех", "2026-07-05")
    assert "повадк" in p.lower()          # охота за стилями — в инструкции
    assert "старый конспект" in p and "хех" in p
    assert "===DIGEST===" in p            # формат ответа показан


def test_parse_output_roundtrip():
    digest, facts, index_line = parse_output(OUT)
    assert digest.startswith("СЕГОДНЯ")
    assert "Повадки" in facts
    assert index_line.startswith("2026-07-05 ★")


def test_parse_output_garbage_returns_none():
    assert parse_output("модель понесло не туда") is None


async def test_condenser_run_once_updates_store(tmp_path):
    st = MemoryStore(tmp_path)
    st.append_reply("Джей", "Смотри, Джефф съел троих!",
                    ts=time.mktime((2026, 7, 5, 21, 14, 0, 0, 0, -1)))

    async def fake_chat(system, user, max_tokens=1200):
        return OUT

    c = Condenser(st, chat=fake_chat)
    assert await c.run_once("2026-07-05")
    assert "Джефф" in st.read_digest()
    assert "пацифистом" in st.read_facts()
    assert "2026-07-05" in st.read_index()
    # курсор сдвинулся — второй прогон без новых записей не работает
    assert not await c.run_once("2026-07-05")


async def test_condenser_interrupt_cancels(tmp_path):
    st = MemoryStore(tmp_path)
    st.append_reply("Джей", "раз", ts=time.time())
    started = asyncio.Event()

    async def slow_chat(system, user, max_tokens=1200):
        started.set()
        await asyncio.sleep(30)
        return OUT

    c = Condenser(st, chat=slow_chat)
    task = asyncio.create_task(c.run_once("2026-07-05"))
    await started.wait()
    c.interrupt()                          # реплика Джея всегда главнее
    assert await task is False
    assert st.read_digest() == ""          # ничего не записано


def test_should_run_idle_gate():
    c = Condenser.__new__(Condenser)
    c._idle_s = 600
    now = time.time()
    assert not c.should_run(last_activity=now - 100, now=now)
    assert c.should_run(last_activity=now - 700, now=now)
```

И в `tests/test_qwen_llm.py` добавить:

```python
async def test_complete_plain_call(monkeypatch):
    from nova.server.models.qwen_llm import QwenVLM

    sent = {}

    class FakeResp:
        def raise_for_status(self):
            pass

        def json(self):
            return {"choices": [{"message": {"content": "выжимка"}}]}

    class FakeClient:
        async def post(self, url, json=None):
            sent.update(json)
            return FakeResp()

    llm = QwenVLM.__new__(QwenVLM)
    llm._model = "m"
    llm._client = FakeClient()
    out = await llm.complete("ты конденсер", "сожми это")
    assert out == "выжимка"
    assert sent["messages"][0]["role"] == "system"
    assert sent["temperature"] == 0.3     # ровная выжимка, не креатив
    assert "presence_penalty" not in sent
```

- [ ] **Шаг 4.2:** оба файла → FAIL.

- [ ] **Шаг 4.3: реализация.** `nova/server/memory/condenser.py`:

```python
import asyncio

from nova.server.memory.store import MemoryStore

PROMPT = """Ты — память NOVA (ИИ-компаньонка Джея). Ниже её текущий
конспект, картотека фактов и НОВЫЕ записи дневника. Обнови всё три.

Правила:
1. КОНСПЕКТ: свежее — подробно и сверху; старые дни ужимай по градиенту
   (вчера короче, неделя — строка на день). Держи ~2500 токенов.
2. ФАКТЫ по разделам: Люди / Повадки и стили / События / NOVA о себе.
   ОСОБО охоться за повадками: манера игры, реакции, вкусы — обобщения
   поверх эпизодов («играет пацифистом», «в хоррорах орёт»). Новое —
   с датой «(с {today})». Устаревшее — удаляй. Ничего не смягчай:
   мат и грязные шутки сохраняются дословно, это летопись, не отчёт.
3. INDEX: одна строка про день {today}: «{today} [★ если был яркий
   момент]: суть дня | сущности: игры, люди, NPC через запятую».

ТЕКУЩИЙ КОНСПЕКТ:
{digest}

ТЕКУЩИЕ ФАКТЫ:
{facts}

НОВЫЕ ЗАПИСИ ДНЕВНИКА:
{tail}

Ответ СТРОГО в формате:
===DIGEST===
<новый конспект целиком>
===FACTS===
<новые факты целиком>
===INDEX===
<одна строка индекса за {today}>"""


def build_prompt(digest: str, facts: str, tail: str, today: str) -> str:
    return PROMPT.format(digest=digest or "(пусто)",
                         facts=facts or "(пусто)", tail=tail, today=today)


def parse_output(text: str) -> tuple[str, str, str] | None:
    try:
        _, rest = text.split("===DIGEST===", 1)
        digest, rest = rest.split("===FACTS===", 1)
        facts, index_line = rest.split("===INDEX===", 1)
    except ValueError:
        return None
    digest, facts = digest.strip(), facts.strip()
    index_line = index_line.strip().splitlines()[0] if index_line.strip() else ""
    if not digest or not index_line:
        return None
    return digest, facts, index_line


class Condenser:
    """Сжатие дневника в конспект/факты — мозгом, в паузах. Реплика
    пользователя прерывает (interrupt), доделаем в следующую паузу."""

    def __init__(self, store: MemoryStore, chat, idle_s: float = 600.0):
        self._store = store
        self._chat = chat  # async (system, user, max_tokens) -> str
        self._idle_s = idle_s
        self._task: asyncio.Task | None = None

    def should_run(self, last_activity: float, now: float) -> bool:
        return now - last_activity >= self._idle_s

    def interrupt(self) -> None:
        if self._task and not self._task.done():
            self._task.cancel()

    async def run_once(self, today: str) -> bool:
        tail, new_cursor = self._store.unprocessed_tail()
        if not tail.strip():
            return False
        prompt = build_prompt(self._store.read_digest(),
                              self._store.read_facts(), tail, today)
        self._task = asyncio.current_task()
        try:
            out = await self._chat("Ты аккуратный летописец.", prompt,
                                   max_tokens=2000)
        except asyncio.CancelledError:
            return False
        parsed = parse_output(out)
        if parsed is None:
            print("[nova] конденсер: ответ не разобран, пропуск до след. паузы")
            return False
        digest, facts, index_line = parsed
        self._store.write_digest(digest)
        if facts:
            self._store.write_facts(facts)
        self._store.set_index_line(today, index_line)
        self._store.set_cursor(new_cursor)
        return True
```

В `nova/server/models/qwen_llm.py` добавить метод в QwenVLM:

```python
    async def complete(self, system: str, user: str,
                       max_tokens: int = 1200) -> str:
        """Служебный вызов без персоны и штрафов — для конденсера памяти."""
        r = await self._client.post(
            "/chat/completions",
            json={
                "model": self._model,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                "max_tokens": max_tokens,
                "temperature": 0.3,
                "chat_template_kwargs": {"enable_thinking": False},
            },
        )
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"]
```

ВНИМАНИЕ: у Condenser.run_once `self._task = asyncio.current_task()` —
interrupt() из другого таска отменяет ИМЕННО текущий прогон; тест
interrupt проверяет, что store не тронут.

- [ ] **Шаг 4.4:** оба тест-файла → passed.

- [ ] **Шаг 4.5:**

```bash
git add nova/server/memory/condenser.py nova/server/models/qwen_llm.py \
  tests/test_memory_condenser.py tests/test_qwen_llm.py
git commit -m "feat: memory condenser - digest, facts, habits via brain"
```

---

### Задача 5: git-синхронизация

**Файлы:** создать `nova/server/memory/sync.py`, `tests/test_memory_sync.py`.

**Производит:** `class MemorySync(root: Path, remote: str | None = None)`:
`ensure_repo()` (git init + identity NOVA, remote при наличии),
`push_now(msg="mem") -> bool`, `async pusher_loop(poke: asyncio.Event,
min_interval_s=30)` — ждёт poke, пушит не чаще interval;
`request_push()` (ставит Event).

- [ ] **Шаг 5.1: тесты (падают)** — `tests/test_memory_sync.py`:

```python
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
```

- [ ] **Шаг 5.2:** → FAIL.

- [ ] **Шаг 5.3: реализация** — `nova/server/memory/sync.py`:

```python
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
```

- [ ] **Шаг 5.4:** → passed. Если на машине git требует master/main —
  тесты создают bare с дефолтной веткой; push HEAD:master задаёт её сам.

- [ ] **Шаг 5.5:**

```bash
git add nova/server/memory/sync.py tests/test_memory_sync.py
git commit -m "feat: memory git sync - push per exchange with debounce"
```

---

### Задача 6: хуки моделей — on_seen и context_provider

**Файлы:** изменить `nova/server/models/gemini_vision.py`,
`nova/server/models/qwen_llm.py`; тесты в `tests/test_gemini_vision.py`,
`tests/test_qwen_llm.py`.

**Производит:** `GeminiEyes(..., on_seen: Callable[[str], None] | None)` —
вызывается с каждым СВЕЖИМ (не из кэша) описанием из describe() и
describe_for_question(); поле публичное `eyes.on_seen = fn` тоже ок;
`GeminiEyes.complete_text(prompt) -> str` (текстовый вызов без кадров —
резерв конденсера NOVA_MEMORY_LLM=gemini);
`QwenVLM(..., context_provider: Callable[[], str] | None)` — добавка к
системному промпту в обоих build_*_messages.

- [ ] **Шаг 6.1: тесты (падают).** В `tests/test_gemini_vision.py`:

```python
async def test_on_seen_gets_fresh_descriptions():
    eyes = make_eyes(FakeInner(), describe_text="1: замес у точки")
    seen = []
    eyes.on_seen = seen.append
    await eyes.describe([b"j1"])
    await eyes.describe([b"j1"])           # кэш — хук молчит
    assert seen == ["замес у точки"]
    await eyes.reply_to_user("что там?", [b"j2"], [])
    assert len(seen) == 2                  # прицельное описание тоже пишется
```

В `tests/test_qwen_llm.py`:

```python
def test_context_provider_extends_system():
    from nova.server.models.qwen_llm import QwenVLM

    llm = QwenVLM.__new__(QwenVLM)
    llm._persona = "ты NOVA"
    llm._context_provider = lambda: "ПАМЯТЬ: вчера был турнир"
    msgs = llm.build_reply_messages("привет", [], [])
    assert msgs[0]["role"] == "system"
    assert "ты NOVA" in msgs[0]["content"]
    assert "вчера был турнир" in msgs[0]["content"]
```

- [ ] **Шаг 6.2:** → FAIL.

- [ ] **Шаг 6.3: реализация.** В GeminiEyes.__init__ добавить
  `self.on_seen = on_seen` (параметр `on_seen=None`). В describe(): после
  разбора строк свежих кадров — `if self.on_seen and lines:
  self.on_seen("; ".join(lines))`. В describe_for_question(): после
  успешного ответа — `if self.on_seen: self.on_seen(desc)` (где desc —
  результат, не BAD_SCREEN). Добавить:

```python
    async def complete_text(self, prompt: str) -> str:
        """Текстовый вызов без кадров — резервный конденсер памяти."""
        return await self._call_gemini([], prompt)
```

В QwenVLM.__init__ добавить параметр `context_provider=None` →
`self._context_provider = context_provider`; в build_reply_messages и
build_comment_messages системное сообщение собирать так:

```python
        system = self._persona
        if self._context_provider:
            extra = self._context_provider()
            if extra:
                system = f"{self._persona}\n\n{extra}"
```

и использовать `system` вместо self._persona. В make_eyes (тестовая
фабрика в test_gemini_vision.py) добавить строку `eyes.on_seen = None`.

- [ ] **Шаг 6.4:** оба файла тестов → passed (все старые тоже).

- [ ] **Шаг 6.5:**

```bash
git add nova/server/models/gemini_vision.py nova/server/models/qwen_llm.py \
  tests/test_gemini_vision.py tests/test_qwen_llm.py
git commit -m "feat: eyes on_seen hook, brain context provider"
```

---

### Задача 7: интеграция — оркестратор и main

**Файлы:** изменить `nova/server/orchestrator.py`, `nova/server/main.py`;
тесты в `tests/test_orchestrator.py`.

**Потребляет:** всё из задач 1–6.

**Производит:** `Session(..., memory: Memory | None = None)`;
`class Memory` (в orchestrator.py — лёгкий контейнер):

```python
class Memory:
    def __init__(self, store, condenser=None, sync=None,
                 chronicle_s: float = 10.0):
        self.store = store
        self.condenser = condenser
        self.sync = sync
        self.chronicle_s = chronicle_s
```

Поведение Session при memory:
- AudioSegment: `store.append_reply("Джей", text)`; если
  `wants_recall(text)` → `rec = recall(store, text, today)`; иначе
  `rec = associate(store, context_keys, today)` где context_keys =
  keywords(text) + keywords(последняя [видела] из store._last_seen);
  непустой rec → `text_llm = f"{rec}\n{text}"`; конденсер
  `memory.condenser.interrupt()` в НАЧАЛЕ обработки (реплика главнее);
  после ответа: `store.append_reply("NOVA", strip_markers(reply))`,
  `sync.request_push()`.
- _comment: после успешного комментария — append_reply("NOVA", ...) и
  request_push().
- Frame: пульс летописи — если прошло ≥ chronicle_s с прошлого раза и у
  self._llm есть describe: `desc = await self._llm.describe([кадр])`;
  запись идёт сама через on_seen-хук (система соберётся в main), Session
  только следит за таймером и дёргает describe.
- Session.close(): store.append_event("клиент отключился") — вызывает
  main в finally; при коннекте main пишет «клиент подключился».

main.build_models/create_app:
- если NOVA_MEMORY != "0": root = Path(NOVA_MEMORY_DIR или
  "/workspace/nova-memory"); store = MemoryStore(root);
  sync = MemorySync(root, remote=None) — remote настраивает деплой
  (репо уже клонирован с origin), поэтому sync.ensure_repo() только
  identity; условие: если .git есть — remote не трогать (доработка:
  ensure_repo() при remote=None и существующем .git НЕ добавляет origin);
- brain = QwenVLM(..., context_provider=store.system_context);
  llm = wrap_eyes(brain); если это GeminiEyes — llm.on_seen =
  store.append_seen;
- condenser = Condenser(store, chat=brain.complete, idle_s из env
  NOVA_CONDENSE_IDLE_S); NOVA_MEMORY_LLM=gemini и глаза облачные →
  chat=lambda s, u, max_tokens=2000: llm.complete_text(s + "\n" + u);
- startup: task пушера sync.pusher_loop() и task конденсер-цикла:

```python
        async def _memory_loop():
            while True:
                await asyncio.sleep(30)
                now = time.time()
                if condenser.should_run(app.state.last_activity, now):
                    today = datetime.now().strftime("%Y-%m-%d")
                    await condenser.run_once(today)
```

- ws_endpoint: memory=Memory(...) в Session; append_event
  подключился/отключился; каждое входящее сообщение —
  condenser.interrupt() уже делает Session.

- [ ] **Шаг 7.1: тесты (падают)** — добавить в `tests/test_orchestrator.py`
  (там уже есть моки Send/ASR/LLM/TTS — использовать их фабрики; ниже
  каркас, подгони имена фикстур по соседним тестам файла):

```python
async def test_memory_writes_dialog_and_recall(tmp_path):
    from nova.server.memory.store import MemoryStore
    from nova.server.orchestrator import Memory, Session
    import time as _t

    st = MemoryStore(tmp_path)
    # старый день с Джеффом — для вспоминания
    ts = _t.mktime((2026, 6, 20, 21, 0, 0, 0, 0, -1))
    st.append_seen("Джефф съел троих на турнире", ts=ts)
    st.set_index_line("2026-06-20", "2026-06-20 ★: Джефф | сущности: Джефф")

    got_texts = []

    class EchoLLM:
        async def reply_to_user(self, text, frames, history):
            got_texts.append(text)
            return "Помню, конечно!"

        async def comment_on_event(self, event, frames, history):
            return "PASS"

    sess = Session(send=async_noop, engine=quiet_engine(), asr=FixedASR(
        "помнишь как Джефф съел троих?"), llm=EchoLLM(), tts=NullTTS(),
        memory=Memory(store=st))
    await sess.handle(audio_segment())
    day = st.read_day(_t.strftime("%Y-%m-%d"))
    assert "[Джей] помнишь как Джефф съел троих?" in day
    assert "[NOVA] Помню, конечно!" in day
    assert "[из дневника за 2026-06-20" in got_texts[0]   # recall вшит


async def test_chronicle_pulse_throttles(tmp_path):
    from nova.server.memory.store import MemoryStore
    from nova.server.orchestrator import Memory, Session

    calls = []

    class EyesLLM(EchoLLM):
        async def describe(self, frames):
            calls.append(1)
            return "экран"

    st = MemoryStore(tmp_path)
    sess = Session(send=async_noop, engine=quiet_engine(), asr=FixedASR("x"),
                   llm=EyesLLM(), tts=NullTTS(),
                   memory=Memory(store=st, chronicle_s=10.0))
    now = time.time()
    await sess.handle(frame_msg())          # первый кадр — описали
    await sess.handle(frame_msg())          # сразу второй — пульс молчит
    assert len(calls) == 1
    sess._last_chronicle -= 11              # «прошло» 11 секунд
    await sess.handle(frame_msg())
    assert len(calls) == 2
```

(вспомогательные фабрики `async_noop`, `quiet_engine`, `FixedASR`,
`NullTTS`, `audio_segment`, `frame_msg` — взять/сделать по образцу уже
существующих в test_orchestrator.py; NullTTS.synthesize — пустой
async-генератор, sample_rate=16000.)

- [ ] **Шаг 7.2:** → FAIL.

- [ ] **Шаг 7.3: реализация** в orchestrator.py: класс Memory (см. выше),
  у Session: параметр `memory=None`, поля `self._memory = memory`,
  `self._last_chronicle = 0.0`. В handle(AudioSegment) после transcribe:

```python
            mem = self._memory
            text_llm = text
            if mem:
                if mem.condenser:
                    mem.condenser.interrupt()
                mem.store.append_reply("Джей", text)
                from nova.server.memory.recall import (
                    associate, keywords, recall, wants_recall)
                today = time.strftime("%Y-%m-%d")
                rec = recall(mem.store, text, today) \
                    if wants_recall(text) else ""
                if not rec:
                    ctx = keywords(text) + keywords(mem.store._last_seen)
                    rec = associate(mem.store, ctx, today, cooldown_days=int(
                        os.environ.get("NOVA_RECALL_COOLDOWN_D", "5")))
                if rec:
                    text_llm = f"{rec}\n{text}"
```

`reply = await self._llm.reply_to_user(text_llm, frames, ...)` (история
и speak — по-прежнему с чистым text). После reply:

```python
            if mem:
                mem.store.append_reply("NOVA", strip_markers(reply))
                if mem.sync:
                    mem.sync.request_push()
```

В _comment после речи — аналогичные две строки. В handle(Frame) в конце:

```python
        if self._memory and time.time() - self._last_chronicle >= \
                self._memory.chronicle_s:
            describe = getattr(self._llm, "describe", None)
            if describe:
                self._last_chronicle = time.time()
                try:
                    await describe([self._frames[-1]])
                except Exception as exc:
                    print(f"[nova] летопись: {exc!r}")
```

(запись в дневник делает on_seen-хук, повешенный в main; дедуп — в
store.append_seen.)

В main.py — сборка памяти (код из «Производит» выше, целиком):
imports asyncio/datetime, блок в create_app после build_models:

```python
    memory = None
    if not mock and os.environ.get("NOVA_MEMORY", "1") == "1":
        from nova.server.memory.condenser import Condenser
        from nova.server.memory.store import MemoryStore
        from nova.server.memory.sync import MemorySync
        from nova.server.orchestrator import Memory

        mem_root = Path(os.environ.get("NOVA_MEMORY_DIR",
                                       "/workspace/nova-memory"))
        store = MemoryStore(mem_root)
        sync = MemorySync(mem_root)
        sync.ensure_repo()
        brain._context_provider = store.system_context
        if hasattr(llm, "on_seen"):
            llm.on_seen = store.append_seen
        chat = brain.complete
        if os.environ.get("NOVA_MEMORY_LLM") == "gemini" \
                and hasattr(llm, "complete_text"):
            async def chat(s, u, max_tokens=2000):
                return await llm.complete_text(s + "\n\n" + u)
        condenser = Condenser(store, chat=chat, idle_s=float(
            os.environ.get("NOVA_CONDENSE_IDLE_S", "600")))
        memory = Memory(store=store, condenser=condenser, sync=sync,
                        chronicle_s=float(
                            os.environ.get("NOVA_CHRONICLE_S", "10")))
```

(в build_models вернуть также brain: `return asr, llm, tts` →
build_models остаётся как есть, а brain достать так: в build_models
сохранить `llm.inner_brain = brain`? НЕТ — проще: build_models уже
создаёт brain внутри; вынести создание QwenVLM в create_app нельзя без
ломки mock. Решение: build_models возвращает 4-й элемент `brain`
(None в mock): `return MockASR(), MockLLM(...), MockTTS(), None` и
`return asr, llm, tts, brain`; поправить оба call-site и старые тесты
не трогаются — они зовут create_app.)

startup-хук дополняется задачами пушера и цикла конденсера (код цикла —
в «Производит»); ws_endpoint: `memory=memory` в Session,
`memory.store.append_event("клиент подключился")` после HelloAck и
`... («клиент отключился»)` в finally (обе под `if memory:`),
плюс `memory.sync.request_push()` в finally.

- [ ] **Шаг 7.4:** `uv run pytest -q` → все зелёные.

- [ ] **Шаг 7.5:**

```bash
git add nova/server/orchestrator.py nova/server/main.py tests/test_orchestrator.py
git commit -m "feat: wire memory - diary, recall, chronicle pulse, condenser"
```

---

### Задача 8: деплой памяти

**Файлы:** изменить `deploy/onstart.sh`, `deploy/runner.sh`,
`scripts/vast.py`, `NOVA_START.bat`; тест в `tests/test_vast.py`.

- [ ] **Шаг 8.1: тест (падает)** — в `tests/test_vast.py` в
  test_create_env_passes_eyes_and_voice добавить в env вызова
  `"NOVA_MEMORY_TOKEN": "mt", "NOVA_MEMORY_REPO":
  "github.com/MrJayfeather/nova-memory.git"` и ассерты:

```python
    assert sent["NOVA_MEMORY_TOKEN"] == "mt"
    assert "nova-memory" in sent["NOVA_MEMORY_REPO"]
```

- [ ] **Шаг 8.2:** → FAIL. В vast.py env-блок добавить:

```python
            "NOVA_MEMORY_TOKEN": env.get("NOVA_MEMORY_TOKEN", ""),
            "NOVA_MEMORY_REPO": env.get(
                "NOVA_MEMORY_REPO",
                "github.com/MrJayfeather/nova-memory.git"),
```

→ passed.

- [ ] **Шаг 8.3: onstart.sh** — после блока клона NOVA добавить:

```bash
# память NOVA: приватный репо, переживает смерть бокса
if [ -n "$NOVA_MEMORY_TOKEN" ] && [ ! -d /workspace/nova-memory/.git ]; then
  for i in 1 2 3; do
    git clone "https://x-access-token:${NOVA_MEMORY_TOKEN}@${NOVA_MEMORY_REPO}" \
      /workspace/nova-memory && break
    echo "memory clone retry $i"; sleep 5
  done
fi
mkdir -p /workspace/nova-memory
```

- [ ] **Шаг 8.4: runner.sh** — в grep-список env добавить
  `NOVA_MEMORY|NOVA_MEMORY_DIR|NOVA_MEMORY_TOKEN|NOVA_MEMORY_REPO|NOVA_CHRONICLE_S|NOVA_CONDENSE_IDLE_S|NOVA_MEMORY_LLM|NOVA_RECALL_COOLDOWN_D`;
  после gemini_key-строки:

```bash
[ -f /workspace/memory_token ] && export NOVA_MEMORY_TOKEN=$(cat /workspace/memory_token)
# пуш памяти при остановке (вачдог/ручной stop): хук в конце runner не
# сработает — стопаем инстанс снаружи, поэтому страховка перед пуском
cd /workspace/nova-memory 2>/dev/null && git push origin HEAD:master 2>/dev/null; cd /workspace/NOVA
```

  (основной канал — пуш после каждого обмена из процесса сервера;
  строка выше добирает хвост при рестартах.)

- [ ] **Шаг 8.5: NOVA_START.bat** — после существующих строк запуска
  добавить (до старта клиента):

```bat
if exist nova-memory\ (cd nova-memory && git pull --quiet & cd ..) else (
  echo [nova] память: локальной копии нет — появится после первого дня )
```

  (клон на ноут — вручную один раз: `git clone
  https://github.com/MrJayfeather/nova-memory.git` в корень проекта;
  каталог в .gitignore основного репо — добавить строку `nova-memory/`.)

- [ ] **Шаг 8.6:** `uv run pytest -q` → зелёные;

```bash
git add deploy/onstart.sh deploy/runner.sh scripts/vast.py NOVA_START.bat .gitignore
git commit -m "feat: memory deploy - private repo clone, token, laptop pull"
```

---

### Задача 9: merge, деплой, живая приёмка (с Джеем)

- [ ] **Шаг 9.1:** `uv run pytest -q` → всё зелёное;
  `git checkout master && git merge memory3a && git push`.

- [ ] **Шаг 9.2: Джей создаёт** (провести по шагам): приватный репо
  `nova-memory` на GitHub (README не нужен) + fine-grained PAT:
  Settings → Developer settings → Fine-grained tokens → Repository
  access: только nova-memory; Permissions: Contents Read/Write.
  Токен → в .env `NOVA_MEMORY_TOKEN=...`, на бокс:
  `echo <токен> > /workspace/memory_token`.

- [ ] **Шаг 9.3:** на боксе: git pull NOVA, onstart.sh (клонирует память),
  перезапуск сервера двумя ssh-вызовами (kill отдельно от start —
  грабля pkill). Проверить: /workspace/nova-memory/.git есть;
  в nova.log нет ошибок memory.

- [ ] **Шаг 9.4: приёмка по критериям спеки:**
  1. поговорить → diary/сегодня.md наполняется репликами и [видела];
  2. подождать 10+ мин тишины → digest.md/facts.md/index.md появились,
     git log в nova-memory растёт;
  3. перезапустить сервер → она в курсе дня (без «Ого, ты в Rivals?»);
  4. «помнишь, что было сегодня/вчера...» → отвечает из дневника;
  5. реплика во время конденсации → без заметной задержки;
  6. NOVA_START.bat на ноуте → локальная копия nova-memory обновилась.
  Эффект Светки (2а) и повадки (2б) — наблюдаются днями, не в приёмку.

- [ ] **Шаг 9.5:** STATUS.md — раздел «ЭТАП 3А ПАМЯТЬ: В ПРОДЕ» с фактами
  прогона; коммит `docs: stage 3a memory live`.
