import difflib
import re
from datetime import date, timedelta

from nova.server.memory.store import MemoryStore

TRIGGERS = ("помнишь", "вспомни", "тогда", "в прошлый раз", "недавно",
            "на прошлой неделе", "назад", "вчера", "позавчера", "было")

_STOP = {"а", "и", "в", "на", "как", "что", "это", "ты", "я", "мы", "же",
         "не", "ну", "у", "с", "к", "по", "за", "из", "был", "была",
         "было", "были", "помнишь", "вспомни", "тогда", "недавно"}

_NUMS = {"один": 1, "одну": 1, "два": 2, "две": 2, "три": 3,
         "четыре": 4, "пять": 5, "пару": 2}


def wants_recall(text: str) -> bool:
    t = text.lower()
    return any(w in t for w in TRIGGERS)


def _d(s: str) -> date:
    return date.fromisoformat(s)


def _amount(word: str | None) -> int:
    if not word:
        return 1
    return int(word) if word.isdigit() else _NUMS.get(word, 1)


def parse_period(text: str, today: str) -> tuple[str, str] | None:
    """«вчера», «N дней/недель/месяцев назад» -> диапазон дат с запасом:
    люди помнят время неточно, окно лучше широкое (index всё равно
    отфильтрует по словам)."""
    t = text.lower()
    base = _d(today)
    if "позавчера" in t:
        d = base - timedelta(days=2)
        return str(d), str(d)
    if "вчера" in t:
        d = base - timedelta(days=1)
        return str(d), str(d)
    words = "|".join(_NUMS)

    def find(unit: str) -> int | None:
        # число бывает и до, и после: «две недели назад» / «недели две назад»
        m = re.search(rf"(?:(\d+|{words})\s+)?{unit}(?:\s+(\d+|{words}))?\s*назад", t)
        if not m:
            return None
        return _amount(m.group(1) or m.group(2))

    n = find(r"недел\w*")
    if n is not None or "на прошлой неделе" in t:
        n = n or 1
        centre = base - timedelta(weeks=n)
        return str(centre - timedelta(days=7)), str(centre + timedelta(days=7))
    n = find(r"месяц\w*")
    if n is not None:
        centre = base - timedelta(days=30 * n)
        return str(centre - timedelta(days=10)), str(centre + timedelta(days=10))
    n = find(r"(?:дня|дней|день)")
    if n is not None:
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
        in_period = bool(period and period[0] <= day <= period[1])
        hits = sum(1 for k in keys if _match(k, line))
        if period and not in_period and hits == 0:
            continue
        score = hits * 2 + (1.5 if in_period else 0) + (0.5 if "★" in line else 0)
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
