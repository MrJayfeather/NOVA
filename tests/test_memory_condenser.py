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
    p = build_prompt("старый конспект", "старые факты", "21:14 [Джей] хех",
                     "2026-07-05")
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
