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
    lo, hi = parse_period("неделю назад", TODAY)
    assert lo <= "2026-06-28" <= hi          # центр — минус неделя, окно широкое
    lo, hi = parse_period("недели две назад был анлак", TODAY)
    assert lo <= "2026-06-21" <= hi
    lo, hi = parse_period("месяц назад", TODAY)
    assert lo <= "2026-06-05" <= hi
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
