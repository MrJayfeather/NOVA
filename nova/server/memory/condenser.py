import asyncio

from nova.server.memory.store import MemoryStore

PROMPT = """Ты — память NOVA (ИИ-компаньонка Джея). Ниже её текущий
конспект, картотека фактов и НОВЫЕ записи дневника. Обнови все три.

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
