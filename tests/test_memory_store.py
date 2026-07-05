import time

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
