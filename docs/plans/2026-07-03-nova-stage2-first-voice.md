# NOVA Этап 2 «Первый голос» — Implementation Plan

> Выполнять задачи строго по порядку; чекбоксы `- [ ]` — для отметки прогресса.

**Goal:** NOVA говорит настоящим голосом: реальные модели (Qwen3-VL + faster-whisper + XTTS-v2) на арендованном GPU Vast.ai вместо mock-заглушек, с управлением сервером в одну команду и автостопом.

**Architecture:** Серверная часть без изменений архитектуры — реальные модели встают за интерфейсы `ASRModel`/`VisionLLM`/`TTSModel` из этапа 1. LLM работает через vLLM (OpenAI-совместимый API на инстансе), Whisper и XTTS живут в процессе оркестратора. На ноуте — управляющий CLI `scripts/vast.py` (поиск/старт/стоп инстанса) и стриминговое воспроизведение звука. Инстанс сам себя останавливает после 15 минут простоя.

**Tech Stack:** vLLM (образ vllm/vllm-openai), Qwen3-VL-30B-A3B-Instruct-FP8, faster-whisper large-v3-turbo, coqui-tts (XTTS-v2), httpx, Vast.ai REST API.

**Spec:** `docs/specs/2026-07-03-nova-companion-design.md`

## Global Constraints

- **Никаких упоминаний AI-инструментов** в коде, коммитах, доках; трейлеров в сообщениях коммитов нет.
- Python 3.12 (`.python-version`); GPU-зависимости (faster-whisper, coqui-tts) НЕ ставятся на ноут — импорты только внутри конструкторов real-моделей.
- Аудио: клиент шлёт PCM16 mono 16 кГц; XTTS отдаёт 24 кГц — клиент берёт частоту из `SpeakStart.sample_rate`.
- Протокол остаётся v1: поле `token` добавляется как необязательное (обратная совместимость).
- Vast.ai: label `nova`, образ `vllm/vllm-openai:v0.11.0`, диск 80 ГБ, по умолчанию **stop** (диск и кэш моделей сохраняются), destroy — только явным флагом.
- Тесты проходят на ноуте без GPU и сети: сетевые/GPU-обвязки тонкие, логика — чистыми функциями.
- Секреты только в `.env` (в .gitignore): `VAST_API_KEY`, `NOVA_TOKEN`.

## File Structure

```
nova/shared/protocol.py        — Hello.token (изменение)
nova/server/models/base.py     — history в сигнатурах, константа NO_COMMENT
nova/server/models/mock.py     — обновлённые сигнатуры
nova/server/models/qwen_llm.py — QwenVLM: httpx-клиент к vLLM, сборка сообщений (новый)
nova/server/models/whisper_asr.py — WhisperASR (новый)
nova/server/models/xtts_tts.py — XttsTTS со стримингом (новый)
nova/server/orchestrator.py    — история диалога, PASS, устойчивость к ошибкам
nova/server/main.py            — real-режим, синглтоны моделей, /health, токен
nova/client/audio_out.py       — стриминговый Player (переписан)
nova/client/config.py          — token
nova/client/main.py            — token в Hello, новый sink
scripts/vast.py                — CLI: search / up / status / down (новый)
deploy/onstart.sh              — провижининг инстанса (новый)
deploy/idle_watchdog.py        — автостоп по простою (новый)
tests/test_qwen_llm.py, tests/test_vast.py, tests/test_watchdog.py (новые)
tests/test_server_ws.py, test_orchestrator.py, test_mock_models.py,
tests/test_audio_out.py, test_client_config.py (обновления)
```

---

### Task 1: Токен-аутентификация (протокол + сервер + клиент)

Сервер будет торчать в интернет — без токена подключиться сможет кто угодно.

**Files:**
- Modify: `nova/shared/protocol.py`, `nova/server/main.py`, `nova/client/config.py`, `nova/client/main.py`, `client_config.yaml`
- Test: `tests/test_server_ws.py`, `tests/test_client_config.py`

**Interfaces:**
- Produces: `Hello.token: str = ""`; `create_app(..., token: str = "")` — при непустом token сервер закрывает соединение кодом **4002**, если `Hello.token` не совпал; `ClientConfig.token: str = ""`.

- [ ] **Step 1: Добавить падающие тесты**

В `tests/test_server_ws.py` добавить:

```python
def test_wrong_token_closes_4002(tmp_path):
    app = create_app(
        mock=True,
        profiles_root=ROOT / "profiles",
        personas_root=ROOT / "personas",
        feedback_path=tmp_path / "feedback.jsonl",
        token="secret123",
    )
    client = TestClient(app)
    with client.websocket_connect("/ws") as ws:
        ws.send_text(dump_message(Hello(profile="desktop", persona="nova", token="wrong")))
        data = ws.receive()
        assert data["type"] == "websocket.close"
        assert data["code"] == 4002


def test_correct_token_accepted(tmp_path):
    app = create_app(
        mock=True,
        profiles_root=ROOT / "profiles",
        personas_root=ROOT / "personas",
        feedback_path=tmp_path / "feedback.jsonl",
        token="secret123",
    )
    client = TestClient(app)
    with client.websocket_connect("/ws") as ws:
        ws.send_text(dump_message(Hello(profile="desktop", persona="nova", token="secret123")))
        ack = json.loads(ws.receive_text())
        assert ack["type"] == "hello_ack"
```

В `tests/test_client_config.py` добавить в `test_defaults_applied`:

```python
    assert cfg.token == ""
```

- [ ] **Step 2: Убедиться, что тесты падают**

Run: `uv run pytest tests/test_server_ws.py tests/test_client_config.py -q`
Expected: FAIL (`unexpected keyword argument 'token'`)

- [ ] **Step 3: Реализовать**

`nova/shared/protocol.py` — в класс `Hello` добавить поле:

```python
class Hello(BaseModel):
    type: Literal["hello"] = "hello"
    protocol: int = PROTOCOL_VERSION
    profile: str
    persona: str
    token: str = ""
```

`nova/server/main.py` — сигнатура и проверка:

```python
def create_app(
    mock: bool = True,
    profiles_root: Path = Path("profiles"),
    personas_root: Path = Path("personas"),
    feedback_path: Path = Path("data/feedback.jsonl"),
    token: str = "",
) -> FastAPI:
```

и в `ws_endpoint` сразу после проверки версии протокола:

```python
        if token and first.token != token:
            await ws.close(code=4002)
            return
```

в `__main__`-блоке пробросить токен из окружения:

```python
    uvicorn.run(
        create_app(mock=mock, token=os.environ.get("NOVA_TOKEN", "")),
        host="0.0.0.0", port=8000,
    )
```

`nova/client/config.py` — в `ClientConfig` добавить `token: str = ""`.

`nova/client/main.py` — в `amain` при создании Connection:

```python
        hello=Hello(profile=cfg.profile, persona=cfg.persona, token=cfg.token),
```

`client_config.yaml` — добавить строку после `persona: nova`:

```yaml
token: ""
```

- [ ] **Step 4: Прогнать тесты**

Run: `uv run pytest -q`
Expected: все passed

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "feat: token auth for websocket connections"
```

---

### Task 2: История диалога, PASS и устойчивость к ошибкам моделей

**Files:**
- Modify: `nova/server/models/base.py`, `nova/server/models/mock.py`, `nova/server/orchestrator.py`
- Test: `tests/test_orchestrator.py`, `tests/test_mock_models.py`

**Interfaces:**
- Produces:
  - `NO_COMMENT = "PASS"` (в `base.py`) — если `comment_on_event` вернул ровно это, сессия молчит.
  - `VisionLLM.reply_to_user(text: str, history: list[dict]) -> str`
  - `VisionLLM.comment_on_event(event: str, frames: list[bytes], history: list[dict]) -> str`
  - `history` — список `{"role": "user"|"assistant", "content": str}` (формат OpenAI), Session хранит последние 24 записи; текущий ход в history НЕ входит (LLM добавляет его сам).
  - Исключение любой модели не роняет сессию — ошибка печатается, ход пропускается.

- [ ] **Step 1: Обновить и добавить тесты**

В `tests/test_mock_models.py` заменить `test_mock_llm_replies_and_comments`:

```python
async def test_mock_llm_replies_and_comments():
    llm = MockLLM(persona_prompt="Ты — NOVA.")
    reply = await llm.reply_to_user("привет", history=[])
    assert "привет" in reply
    comment = await llm.comment_on_event("scene_change", frames=[b"jpg"], history=[])
    assert "scene_change" in comment
    assert "1" in comment
```

В `tests/test_orchestrator.py` добавить:

```python
from nova.server.models.base import NO_COMMENT, VisionLLM


class RecordingLLM(VisionLLM):
    """Запоминает, с какой историей его вызвали."""

    def __init__(self, reply="ок", comment="вижу"):
        self.calls = []
        self._reply, self._comment = reply, comment

    async def reply_to_user(self, text, history):
        self.calls.append(("reply", text, list(history)))
        return self._reply

    async def comment_on_event(self, event, frames, history):
        self.calls.append(("comment", event, list(history)))
        return self._comment


def make_session_with(llm, tmp_path=None):
    sent = []

    async def send(msg):
        sent.append(msg)

    session = Session(
        send=send,
        engine=ProactiveEngine(cooldown_s=0.0, talkativeness=0.5, dedupe_window_s=0.0),
        asr=MockASR(),
        llm=llm,
        tts=MockTTS(),
        feedback_path=(tmp_path / "feedback.jsonl") if tmp_path else None,
    )
    return session, sent


async def test_history_accumulates_across_turns():
    llm = RecordingLLM(reply="ответ")
    session, _ = make_session_with(llm)
    pcm_b64 = base64.b64encode(b"\x00\x00" * 1600).decode()
    await session.handle(AudioSegment(ts=1.0, pcm_b64=pcm_b64, sample_rate=16000))
    await session.handle(AudioSegment(ts=2.0, pcm_b64=pcm_b64, sample_rate=16000))
    # второй вызов должен видеть первый ход (user + assistant)
    _, _, history = llm.calls[1]
    assert len(history) == 2
    assert history[0]["role"] == "user"
    assert history[1] == {"role": "assistant", "content": "ответ"}


async def test_pass_comment_is_silent():
    llm = RecordingLLM(comment=NO_COMMENT)
    session, sent = make_session_with(llm)
    await session.handle(DetectorEvent(ts=1.0, event="scene_change"))
    assert sent == []


async def test_model_error_does_not_crash_session():
    class BrokenLLM(VisionLLM):
        async def reply_to_user(self, text, history):
            raise RuntimeError("gpu on fire")

        async def comment_on_event(self, event, frames, history):
            raise RuntimeError("gpu on fire")

    session, sent = make_session_with(BrokenLLM())
    await session.handle(DetectorEvent(ts=1.0, event="scene_change"))  # не должно бросить
    assert sent == []
```

- [ ] **Step 2: Убедиться, что тесты падают**

Run: `uv run pytest tests/test_orchestrator.py tests/test_mock_models.py -q`
Expected: FAIL (сигнатуры без `history`)

- [ ] **Step 3: Реализовать**

`nova/server/models/base.py`:

```python
from abc import ABC, abstractmethod
from typing import AsyncIterator

NO_COMMENT = "PASS"


class ASRModel(ABC):
    @abstractmethod
    async def transcribe(self, pcm: bytes, sample_rate: int) -> str: ...


class VisionLLM(ABC):
    @abstractmethod
    async def reply_to_user(self, text: str, history: list[dict]) -> str: ...

    @abstractmethod
    async def comment_on_event(
        self, event: str, frames: list[bytes], history: list[dict]
    ) -> str: ...


class TTSModel(ABC):
    sample_rate: int

    @abstractmethod
    def synthesize(self, text: str) -> AsyncIterator[bytes]: ...
```

`nova/server/models/mock.py` — обновить сигнатуры MockLLM (тела без изменений):

```python
    async def reply_to_user(self, text: str, history: list[dict]) -> str:
        return f"(мок) Ты сказал: «{text}». Отвечаю как положено."

    async def comment_on_event(self, event: str, frames: list[bytes], history: list[dict]) -> str:
        return f"(мок) Заметила событие {event}, кадров получила: {len(frames)}."
```

`nova/server/orchestrator.py` — история и устойчивость. Импорт: `from nova.server.models.base import ASRModel, NO_COMMENT, TTSModel, VisionLLM`. В `__init__` добавить:

```python
        self._history: deque[dict] = deque(maxlen=24)
```

Заменить ветки `AudioSegment` и `DetectorEvent` в `handle`, и `comment_now` в `_handle_hotkey`:

```python
        elif isinstance(msg, AudioSegment):
            try:
                text = await self._asr.transcribe(base64.b64decode(msg.pcm_b64), msg.sample_rate)
                reply = await self._llm.reply_to_user(text, list(self._history))
            except Exception as exc:
                print(f"[nova] ошибка модели (reply): {exc!r}")
                return
            self._history.append({"role": "user", "content": text})
            self._history.append({"role": "assistant", "content": reply})
            await self._speak(reply, reason="reply")
        elif isinstance(msg, DetectorEvent):
            decision = self._engine.on_event(msg.event, now=time.time())
            if decision.speak:
                await self._comment(msg.event, reason="proactive")
```

и добавить метод + обновить `comment_now`:

```python
    async def _comment(self, event: str, reason: str) -> None:
        try:
            comment = await self._llm.comment_on_event(
                event, list(self._frames), list(self._history)
            )
        except Exception as exc:
            print(f"[nova] ошибка модели (comment): {exc!r}")
            return
        if comment.strip() == NO_COMMENT:
            return
        self._history.append({"role": "assistant", "content": comment})
        await self._speak(comment, reason=reason)
```

```python
        if msg.action == "comment_now":
            self._engine.on_event("comment_now", now=time.time(), forced=True)
            await self._comment("user_request", reason="forced")
```

- [ ] **Step 4: Прогнать все тесты**

Run: `uv run pytest -q`
Expected: все passed

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "feat: dialog history, silent PASS comments and model error resilience"
```

---

### Task 3: Стриминговое воспроизведение звука

Реальный TTS отдаёт звук чанками — играть надо с первого чанка, а не после конца фразы.

**Files:**
- Modify: `nova/client/audio_out.py`, `nova/client/main.py`
- Test: `tests/test_audio_out.py`

**Interfaces:**
- Produces:
  - `StreamSink` (ABC): `.start(sample_rate: int)`, `.write(pcm: bytes)`, `.stop()`
  - `SounddeviceStreamSink()` — реальное стрим-воспроизведение
  - `Player(sink: StreamSink)` — `.handle(msg)`, `.muted: bool`, `.drain()` (дождаться опустошения очереди — для тестов). Внутри — рабочий поток: `handle` не блокируется.

- [ ] **Step 1: Переписать тесты**

`tests/test_audio_out.py` целиком:

```python
import base64

from nova.client.audio_out import Player, StreamSink
from nova.shared.protocol import AudioChunk, SpeakEnd, SpeakStart


class FakeStreamSink(StreamSink):
    def __init__(self):
        self.events = []

    def start(self, sample_rate: int) -> None:
        self.events.append(("start", sample_rate))

    def write(self, pcm: bytes) -> None:
        self.events.append(("write", pcm))

    def stop(self) -> None:
        self.events.append(("stop",))


def utterance(uid="u1", parts=(b"aa", b"bb")):
    msgs = [SpeakStart(utterance_id=uid, text="т", reason="reply", sample_rate=24000)]
    for i, p in enumerate(parts):
        msgs.append(AudioChunk(utterance_id=uid, seq=i, pcm_b64=base64.b64encode(p).decode()))
    msgs.append(SpeakEnd(utterance_id=uid))
    return msgs


def test_chunks_streamed_in_order():
    sink = FakeStreamSink()
    player = Player(sink)
    for msg in utterance():
        player.handle(msg)
    player.drain()
    assert sink.events == [
        ("start", 24000), ("write", b"aa"), ("write", b"bb"), ("stop",),
    ]


def test_muted_drops_utterance():
    sink = FakeStreamSink()
    player = Player(sink)
    player.muted = True
    for msg in utterance():
        player.handle(msg)
    player.drain()
    assert sink.events == []


def test_chunk_for_unknown_utterance_ignored():
    sink = FakeStreamSink()
    player = Player(sink)
    player.handle(AudioChunk(utterance_id="ghost", seq=0, pcm_b64="YWE="))
    player.handle(SpeakEnd(utterance_id="ghost"))
    player.drain()
    assert sink.events == []
```

- [ ] **Step 2: Убедиться, что тесты падают**

Run: `uv run pytest tests/test_audio_out.py -q`
Expected: FAIL (`cannot import name 'StreamSink'`)

- [ ] **Step 3: Переписать audio_out.py**

```python
import base64
import queue
import threading
from abc import ABC, abstractmethod

from nova.shared.protocol import AudioChunk, SpeakEnd, SpeakStart


class StreamSink(ABC):
    @abstractmethod
    def start(self, sample_rate: int) -> None: ...

    @abstractmethod
    def write(self, pcm: bytes) -> None: ...

    @abstractmethod
    def stop(self) -> None: ...


class SounddeviceStreamSink(StreamSink):
    def __init__(self):
        self._stream = None

    def start(self, sample_rate: int) -> None:
        import sounddevice as sd

        self.stop()
        self._stream = sd.RawOutputStream(samplerate=sample_rate, channels=1, dtype="int16")
        self._stream.start()

    def write(self, pcm: bytes) -> None:
        if self._stream is not None:
            self._stream.write(pcm)

    def stop(self) -> None:
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None


class Player:
    """Стриминговое воспроизведение: звук начинается с первого чанка.

    handle() не блокируется — запись в устройство идёт в рабочем потоке.
    """

    def __init__(self, sink: StreamSink):
        self._sink = sink
        self.muted = False
        self._uid: str | None = None
        self._q: queue.Queue = queue.Queue()
        threading.Thread(target=self._run, daemon=True).start()

    def handle(self, msg) -> None:
        if isinstance(msg, SpeakStart):
            if self.muted:
                self._uid = None
                return
            self._uid = msg.utterance_id
            self._q.put(("start", msg.sample_rate))
        elif isinstance(msg, AudioChunk):
            if msg.utterance_id == self._uid:
                self._q.put(("write", base64.b64decode(msg.pcm_b64)))
        elif isinstance(msg, SpeakEnd):
            if msg.utterance_id == self._uid:
                self._q.put(("stop", None))
            self._uid = None

    def drain(self) -> None:
        self._q.join()

    def _run(self) -> None:
        while True:
            kind, payload = self._q.get()
            try:
                if kind == "start":
                    self._sink.start(payload)
                elif kind == "write":
                    self._sink.write(payload)
                elif kind == "stop":
                    self._sink.stop()
            except Exception as exc:
                print(f"[nova] ошибка воспроизведения: {exc!r}")
            finally:
                self._q.task_done()
```

`nova/client/main.py` — обновить импорт и создание:

```python
from nova.client.audio_out import Player, SounddeviceStreamSink
```

```python
    player = Player(SounddeviceStreamSink())
```

- [ ] **Step 4: Прогнать тесты**

Run: `uv run pytest -q`
Expected: все passed

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "feat: streaming audio playback starting from first chunk"
```

---

### Task 4: QwenVLM — мозг через vLLM

**Files:**
- Create: `nova/server/models/qwen_llm.py`
- Modify: `pyproject.toml` (httpx из dev-группы в основные зависимости)
- Test: `tests/test_qwen_llm.py`

**Interfaces:**
- Consumes: `VisionLLM`, `NO_COMMENT` (Task 2).
- Produces: `QwenVLM(persona_prompt: str, base_url: str, model: str, timeout: float = 60.0)`; методы `build_reply_messages(text, history) -> list[dict]` и `build_comment_messages(event, frames, history) -> list[dict]` (чистые, тестируются без сети); реализует `VisionLLM`.

- [ ] **Step 1: Написать падающий тест**

```python
# tests/test_qwen_llm.py
from nova.server.models.base import NO_COMMENT
from nova.server.models.qwen_llm import QwenVLM


def make_llm():
    return QwenVLM(persona_prompt="Ты — NOVA.", base_url="http://x/v1", model="test-model")


def test_reply_messages_structure():
    history = [
        {"role": "user", "content": "раньше"},
        {"role": "assistant", "content": "ответ"},
    ]
    msgs = make_llm().build_reply_messages("привет", history)
    assert msgs[0] == {"role": "system", "content": "Ты — NOVA."}
    assert msgs[1:3] == history
    assert msgs[-1] == {"role": "user", "content": "привет"}


def test_comment_messages_have_images_and_pass_instruction():
    frames = [b"jpg1", b"jpg2"]
    msgs = make_llm().build_comment_messages("scene_change", frames, history=[])
    content = msgs[-1]["content"]
    images = [c for c in content if c["type"] == "image_url"]
    assert len(images) == 2
    assert images[0]["image_url"]["url"].startswith("data:image/jpeg;base64,")
    text = [c for c in content if c["type"] == "text"][0]["text"]
    assert "scene_change" in text
    assert NO_COMMENT in text


def test_comment_frames_capped_at_eight():
    frames = [b"x"] * 20
    msgs = make_llm().build_comment_messages("motion_burst", frames, history=[])
    images = [c for c in msgs[-1]["content"] if c["type"] == "image_url"]
    assert len(images) == 8
```

- [ ] **Step 2: Убедиться, что тест падает**

Run: `uv run pytest tests/test_qwen_llm.py -q`
Expected: FAIL — `No module named 'nova.server.models.qwen_llm'`

- [ ] **Step 3: Реализовать**

В `pyproject.toml` перенести `"httpx>=0.27"` из `[dependency-groups] dev` в основной список `dependencies` (в dev-группе строку удалить), затем `uv sync`.

```python
# nova/server/models/qwen_llm.py
import base64

import httpx

from nova.server.models.base import NO_COMMENT, VisionLLM

COMMENT_INSTRUCTION = (
    "Событие на экране: {event}. Посмотри на кадры и, если там есть что-то, "
    "что стоит прокомментировать в твоём стиле, дай короткую живую реплику "
    "(1–2 предложения). Если ничего интересного нет или ты это уже "
    "комментировала — ответь ровно: " + NO_COMMENT
)


def _image_part(jpeg: bytes) -> dict:
    b64 = base64.b64encode(jpeg).decode()
    return {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}


class QwenVLM(VisionLLM):
    def __init__(self, persona_prompt: str, base_url: str, model: str, timeout: float = 60.0):
        self._persona = persona_prompt
        self._model = model
        self._client = httpx.AsyncClient(base_url=base_url, timeout=timeout)

    def build_reply_messages(self, text: str, history: list[dict]) -> list[dict]:
        return [
            {"role": "system", "content": self._persona},
            *history[-24:],
            {"role": "user", "content": text},
        ]

    def build_comment_messages(
        self, event: str, frames: list[bytes], history: list[dict]
    ) -> list[dict]:
        content = [_image_part(f) for f in frames[-8:]]
        content.append({"type": "text", "text": COMMENT_INSTRUCTION.format(event=event)})
        return [
            {"role": "system", "content": self._persona},
            *history[-24:],
            {"role": "user", "content": content},
        ]

    async def _chat(self, messages: list[dict]) -> str:
        r = await self._client.post(
            "/chat/completions",
            json={
                "model": self._model,
                "messages": messages,
                "max_tokens": 200,
                "temperature": 0.8,
            },
        )
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"].strip()

    async def reply_to_user(self, text: str, history: list[dict]) -> str:
        return await self._chat(self.build_reply_messages(text, history))

    async def comment_on_event(
        self, event: str, frames: list[bytes], history: list[dict]
    ) -> str:
        return await self._chat(self.build_comment_messages(event, frames, history))
```

- [ ] **Step 4: Прогнать тесты**

Run: `uv run pytest -q`
Expected: все passed

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "feat: vision llm backed by vllm openai-compatible api"
```

---

### Task 5: WhisperASR и XttsTTS

GPU-обвязки: тонкие, импорты ленивые (на ноуте эти пакеты не установлены).

**Files:**
- Create: `nova/server/models/whisper_asr.py`, `nova/server/models/xtts_tts.py`
- Test: `tests/test_gpu_models.py`

**Interfaces:**
- Consumes: `ASRModel`, `TTSModel` (Task 2).
- Produces: `WhisperASR(model_name: str = "large-v3-turbo", device: str = "cuda")`; `XttsTTS(speaker_wav: Path | None = None, default_speaker: str = "Ana Florence")`, `XttsTTS.sample_rate == 24000`.

- [ ] **Step 1: Написать smoke-тесты (скипаются без GPU-пакетов)**

```python
# tests/test_gpu_models.py
"""Smoke-тесты GPU-моделей: на ноуте пропускаются (пакеты не установлены),
на инстансе гоняются вручную: uv run pytest tests/test_gpu_models.py"""
import pytest


def test_whisper_asr_transcribes_silence():
    pytest.importorskip("faster_whisper")
    from nova.server.models.whisper_asr import WhisperASR

    asr = WhisperASR(model_name="tiny", device="cpu")
    import asyncio

    text = asyncio.run(asr.transcribe(b"\x00\x00" * 16000, 16000))
    assert isinstance(text, str)


def test_xtts_yields_pcm_chunks():
    pytest.importorskip("TTS")
    import asyncio

    from nova.server.models.xtts_tts import XttsTTS

    tts = XttsTTS()

    async def run():
        return [c async for c in tts.synthesize("привет")]

    chunks = asyncio.run(run())
    assert chunks and all(isinstance(c, bytes) for c in chunks)
```

- [ ] **Step 2: Убедиться, что тесты скипаются локально**

Run: `uv run pytest tests/test_gpu_models.py -v`
Expected: 2 **skipped** (нет faster_whisper/TTS) — это норма

- [ ] **Step 3: Реализовать**

```python
# nova/server/models/whisper_asr.py
import asyncio

from nova.server.models.base import ASRModel


class WhisperASR(ASRModel):
    def __init__(self, model_name: str = "large-v3-turbo", device: str = "cuda"):
        from faster_whisper import WhisperModel

        compute = "int8_float16" if device == "cuda" else "int8"
        self._model = WhisperModel(model_name, device=device, compute_type=compute)

    def _transcribe_sync(self, pcm: bytes, sample_rate: int) -> str:
        import numpy as np

        audio = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
        segments, _ = self._model.transcribe(audio, language="ru", beam_size=1)
        return " ".join(s.text.strip() for s in segments).strip()

    async def transcribe(self, pcm: bytes, sample_rate: int) -> str:
        return await asyncio.to_thread(self._transcribe_sync, pcm, sample_rate)
```

```python
# nova/server/models/xtts_tts.py
import asyncio
import threading
from pathlib import Path
from typing import AsyncIterator

from nova.server.models.base import TTSModel


class XttsTTS(TTSModel):
    sample_rate = 24000

    def __init__(self, speaker_wav: Path | None = None, default_speaker: str = "Ana Florence"):
        from TTS.tts.configs.xtts_config import XttsConfig
        from TTS.tts.models.xtts import Xtts
        from TTS.utils.manage import ModelManager

        model_dir, _, _ = ModelManager().download_model(
            "tts_models/multilingual/multi-dataset/xtts_v2"
        )
        config = XttsConfig()
        config.load_json(str(Path(model_dir) / "config.json"))
        self._model = Xtts.init_from_config(config)
        self._model.load_checkpoint(config, checkpoint_dir=str(model_dir), eval=True)
        try:
            self._model.cuda()
        except Exception:
            pass  # cpu-режим для smoke-теста
        if speaker_wav is not None and Path(speaker_wav).exists():
            self._latent, self._embedding = self._model.get_conditioning_latents(
                audio_path=[str(speaker_wav)]
            )
            print(f"[nova] голос клонирован из {speaker_wav}")
        else:
            spk = self._model.speaker_manager.speakers[default_speaker]
            self._latent = spk["gpt_cond_latent"]
            self._embedding = spk["speaker_embedding"]
            print(f"[nova] голос по умолчанию: {default_speaker}")

    async def synthesize(self, text: str) -> AsyncIterator[bytes]:
        out: asyncio.Queue = asyncio.Queue()
        loop = asyncio.get_running_loop()

        def produce():
            import numpy as np

            try:
                stream = self._model.inference_stream(
                    text, "ru", self._latent, self._embedding
                )
                for chunk in stream:
                    pcm = (
                        (chunk.squeeze().clamp(-1, 1).cpu().numpy() * 32767)
                        .astype(np.int16)
                        .tobytes()
                    )
                    loop.call_soon_threadsafe(out.put_nowait, pcm)
            except Exception as exc:
                print(f"[nova] ошибка TTS: {exc!r}")
            finally:
                loop.call_soon_threadsafe(out.put_nowait, None)

        threading.Thread(target=produce, daemon=True).start()
        while True:
            item = await out.get()
            if item is None:
                break
            yield item
```

- [ ] **Step 4: Прогнать тесты**

Run: `uv run pytest -q`
Expected: passed + 2 skipped

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "feat: whisper asr and xtts streaming tts wrappers"
```

---

### Task 6: Сервер — real-режим, синглтоны моделей, /health

**Files:**
- Modify: `nova/server/main.py`
- Test: `tests/test_server_ws.py`

**Interfaces:**
- Consumes: все модели (Task 2, 4, 5).
- Produces:
  - `build_models(mock: bool, persona_prompt: str) -> tuple[ASRModel, VisionLLM, TTSModel]` — модели создаются ОДИН раз на приложение (реальные тяжёлые, шарятся между подключениями; Session остаётся пер-подключение).
  - `GET /health` → `{"clients": int, "idle_s": float}` (для автостопа).
  - env: `NOVA_MOCK` (деф. "1"), `NOVA_TOKEN`, `NOVA_VLLM_URL` (деф. `http://127.0.0.1:5000/v1`), `NOVA_MODEL` (деф. `Qwen/Qwen3-VL-30B-A3B-Instruct-FP8`), `NOVA_WHISPER` (деф. `large-v3-turbo`), `NOVA_PERSONA` (деф. `nova`).
  - Персона загружается один раз при старте (из `NOVA_PERSONA`), `Hello.persona` игнорируется.

- [ ] **Step 1: Добавить падающий тест**

В `tests/test_server_ws.py`:

```python
import time


def test_health_reports_clients_and_idle(tmp_path):
    client = make_client(tmp_path)
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["clients"] == 0
    assert body["idle_s"] >= 0

    with client.websocket_connect("/ws") as ws:
        ws.send_text(dump_message(Hello(profile="desktop", persona="nova")))
        ws.receive_text()
        assert client.get("/health").json()["clients"] == 1
```

- [ ] **Step 2: Убедиться, что тест падает**

Run: `uv run pytest tests/test_server_ws.py -q`
Expected: FAIL (404 на /health)

- [ ] **Step 3: Переписать nova/server/main.py**

```python
import os
import time
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from pydantic import ValidationError

from nova.server.models.mock import MockASR, MockLLM, MockTTS
from nova.server.orchestrator import Session
from nova.server.proactive import ProactiveEngine
from nova.shared.profiles import load_persona_prompt, load_profile
from nova.shared.protocol import (
    PROTOCOL_VERSION, Hello, HelloAck, dump_message, parse_client_message,
)


def build_models(mock: bool, persona_prompt: str):
    if mock:
        return MockASR(), MockLLM(persona_prompt=persona_prompt), MockTTS()
    from nova.server.models.qwen_llm import QwenVLM
    from nova.server.models.whisper_asr import WhisperASR
    from nova.server.models.xtts_tts import XttsTTS

    asr = WhisperASR(model_name=os.environ.get("NOVA_WHISPER", "large-v3-turbo"))
    llm = QwenVLM(
        persona_prompt=persona_prompt,
        base_url=os.environ.get("NOVA_VLLM_URL", "http://127.0.0.1:5000/v1"),
        model=os.environ.get("NOVA_MODEL", "Qwen/Qwen3-VL-30B-A3B-Instruct-FP8"),
    )
    persona = os.environ.get("NOVA_PERSONA", "nova")
    tts = XttsTTS(speaker_wav=Path("personas") / persona / "voice_sample.wav")
    return asr, llm, tts


def create_app(
    mock: bool = True,
    profiles_root: Path = Path("profiles"),
    personas_root: Path = Path("personas"),
    feedback_path: Path = Path("data/feedback.jsonl"),
    token: str = "",
) -> FastAPI:
    app = FastAPI(title="NOVA server")
    persona = os.environ.get("NOVA_PERSONA", "nova")
    persona_prompt = load_persona_prompt(persona, personas_root)
    asr, llm, tts = build_models(mock, persona_prompt)
    app.state.clients = 0
    app.state.last_activity = time.time()

    @app.get("/health")
    def health():
        return {
            "clients": app.state.clients,
            "idle_s": round(time.time() - app.state.last_activity, 1),
        }

    @app.websocket("/ws")
    async def ws_endpoint(ws: WebSocket):
        await ws.accept()
        try:
            first = parse_client_message(await ws.receive_text())
        except ValidationError:
            await ws.close(code=4000)
            return
        if not isinstance(first, Hello):
            await ws.close(code=4000)
            return
        if first.protocol != PROTOCOL_VERSION:
            await ws.close(code=4001)
            return
        if token and first.token != token:
            await ws.close(code=4002)
            return

        profile = load_profile(first.profile, profiles_root)
        engine = ProactiveEngine(
            cooldown_s=profile.proactive.cooldown_s,
            talkativeness=profile.proactive.talkativeness,
            dedupe_window_s=profile.proactive.dedupe_window_s,
        )

        async def send(msg):
            await ws.send_text(dump_message(msg))

        session = Session(
            send=send, engine=engine, asr=asr, llm=llm, tts=tts,
            feedback_path=feedback_path,
        )
        await send(HelloAck(mock=mock))
        app.state.clients += 1
        app.state.last_activity = time.time()
        try:
            while True:
                msg = parse_client_message(await ws.receive_text())
                app.state.last_activity = time.time()
                await session.handle(msg)
        except WebSocketDisconnect:
            pass
        finally:
            app.state.clients -= 1
            app.state.last_activity = time.time()

    return app


if __name__ == "__main__":
    import uvicorn

    mock = os.environ.get("NOVA_MOCK", "1") == "1"
    uvicorn.run(
        create_app(mock=mock, token=os.environ.get("NOVA_TOKEN", "")),
        host="0.0.0.0", port=8000,
    )
```

Примечание: `HelloAck(mock=mock)` — в mock-режиме True, в real — False; старая жёсткая `NotImplementedError` удалена.

- [ ] **Step 4: Прогнать все тесты**

Run: `uv run pytest -q`
Expected: все passed (+2 skipped)

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "feat: real model wiring, shared model singletons and health endpoint"
```

---

### Task 7: scripts/vast.py — управление инстансом

**Files:**
- Create: `scripts/vast.py`
- Test: `tests/test_vast.py`

**Interfaces:**
- Produces (чистые функции — тестируются): `load_env(path: Path) -> dict`, `pick_offer(offers: list[dict], min_disk: float = 80.0) -> dict | None`, `ws_url(instance: dict) -> str | None`.
- CLI: `python scripts/vast.py search|up|status|down [--destroy] [--write-config]`.
- Требования к офферу: `gpu_ram >= 47000` (МБ, = 48 ГБ), `reliability2 >= 0.98`, `inet_down >= 200`, `disk_space >= 80`, минимальный `dph_total`.

- [ ] **Step 1: Написать падающий тест**

```python
# tests/test_vast.py
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from vast import load_env, pick_offer, ws_url


def offer(**kw):
    base = dict(id=1, gpu_ram=49152, reliability2=0.99, inet_down=500,
                disk_space=100, dph_total=0.40, gpu_name="A40")
    base.update(kw)
    return base


def test_pick_offer_cheapest_valid():
    offers = [offer(id=1, dph_total=0.50), offer(id=2, dph_total=0.30),
              offer(id=3, dph_total=0.20, gpu_ram=24000)]  # мало VRAM
    assert pick_offer(offers)["id"] == 2


def test_pick_offer_rejects_unreliable_and_slow():
    offers = [offer(reliability2=0.90), offer(inet_down=50), offer(disk_space=40)]
    assert pick_offer(offers) is None


def test_ws_url_from_instance():
    inst = {"public_ipaddr": "1.2.3.4 ", "ports": {"8000/tcp": [{"HostPort": "41234"}]}}
    assert ws_url(inst) == "ws://1.2.3.4:41234/ws"
    assert ws_url({"public_ipaddr": "", "ports": {}}) is None


def test_load_env(tmp_path):
    p = tmp_path / ".env"
    p.write_text("# comment\nVAST_API_KEY=abc\nNOVA_TOKEN = xyz\n", encoding="utf-8")
    env = load_env(p)
    assert env["VAST_API_KEY"] == "abc"
    assert env["NOVA_TOKEN"] == "xyz"
```

- [ ] **Step 2: Убедиться, что тест падает**

Run: `uv run pytest tests/test_vast.py -q`
Expected: FAIL — `No module named 'vast'`

- [ ] **Step 3: Реализовать scripts/vast.py**

```python
"""Управление GPU-инстансом NOVA на Vast.ai.

  python scripts/vast.py search           — топ дешёвых подходящих карт
  python scripts/vast.py up [--write-config]  — старт (существующий или новый)
  python scripts/vast.py status           — состояние и баланс
  python scripts/vast.py down [--destroy] — стоп (диск сохраняется) / полное удаление
"""
import argparse
import secrets
import sys
import time
from pathlib import Path

import httpx

API = "https://console.vast.ai/api/v0"
LABEL = "nova"
IMAGE = "vllm/vllm-openai:v0.11.0"
DISK_GB = 80
ROOT = Path(__file__).parent.parent


def load_env(path: Path) -> dict:
    env = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if "=" in line and not line.lstrip().startswith("#"):
                k, v = line.split("=", 1)
                env[k.strip()] = v.strip()
    return env


def pick_offer(offers: list[dict], min_disk: float = 80.0) -> dict | None:
    ok = [
        o for o in offers
        if o.get("gpu_ram", 0) >= 47000
        and o.get("reliability2", 0) >= 0.98
        and o.get("inet_down", 0) >= 200
        and o.get("disk_space", 0) >= min_disk
    ]
    return min(ok, key=lambda o: o.get("dph_total", 9e9)) if ok else None


def ws_url(instance: dict) -> str | None:
    ports = (instance.get("ports") or {}).get("8000/tcp") or []
    ip = (instance.get("public_ipaddr") or "").strip()
    if ip and ports:
        return f"ws://{ip}:{ports[0]['HostPort']}/ws"
    return None


# ---- REST-обвязка (не тестируется юнитами) ----

def _headers(key: str) -> dict:
    return {"Authorization": f"Bearer {key}"}


def search_offers(key: str) -> list[dict]:
    q = {
        "verified": {"eq": True}, "rentable": {"eq": True},
        "num_gpus": {"eq": 1}, "gpu_ram": {"gte": 47000},
        "order": [["dph_total", "asc"]], "type": "on-demand", "limit": 40,
    }
    r = httpx.post(f"{API}/bundles/", headers=_headers(key), json=q, timeout=60)
    r.raise_for_status()
    return r.json().get("offers", [])


def my_instances(key: str) -> list[dict]:
    r = httpx.get(f"{API}/instances/", headers=_headers(key),
                  params={"owner": "me"}, timeout=60)
    r.raise_for_status()
    return [i for i in r.json().get("instances", []) if i.get("label") == LABEL]


def create_instance(key: str, offer_id: int, token: str) -> None:
    onstart = (ROOT / "deploy" / "onstart.sh").read_text(encoding="utf-8")
    body = {
        "client_id": "me",
        "image": IMAGE,
        "disk": DISK_GB,
        "label": LABEL,
        "onstart": onstart,
        "runtype": "ssh",
        "env": {
            "-p 8000:8000": "1",
            "NOVA_MOCK": "0",
            "NOVA_TOKEN": token,
            "VAST_API_KEY": key,
            "HF_HOME": "/workspace/hf",
            "COQUI_TOS_AGREED": "1",
        },
    }
    r = httpx.put(f"{API}/asks/{offer_id}/", headers=_headers(key), json=body, timeout=60)
    r.raise_for_status()


def set_state(key: str, instance_id: int, state: str) -> None:
    r = httpx.put(f"{API}/instances/{instance_id}/", headers=_headers(key),
                  json={"state": state}, timeout=60)
    r.raise_for_status()


def destroy(key: str, instance_id: int) -> None:
    r = httpx.delete(f"{API}/instances/{instance_id}/", headers=_headers(key), timeout=60)
    r.raise_for_status()


def credit(key: str) -> float:
    r = httpx.get(f"{API}/users/current/", headers=_headers(key), timeout=60)
    r.raise_for_status()
    return round(r.json().get("credit", 0.0), 2)


# ---- команды ----

def ensure_token(env_path: Path, env: dict) -> str:
    token = env.get("NOVA_TOKEN", "")
    if not token:
        token = secrets.token_hex(16)
        with env_path.open("a", encoding="utf-8") as f:
            f.write(f"NOVA_TOKEN={token}\n")
        print("[vast] сгенерирован NOVA_TOKEN и добавлен в .env")
    return token


def write_client_config(url: str, token: str) -> None:
    import yaml

    path = ROOT / "client_config.yaml"
    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    cfg["server_url"] = url
    cfg["token"] = token
    path.write_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False),
                    encoding="utf-8")
    print(f"[vast] client_config.yaml обновлён: {url}")


def cmd_search(key: str) -> None:
    for o in sorted(search_offers(key), key=lambda o: o["dph_total"])[:10]:
        print(f"  {o['gpu_name']:<14} {o['gpu_ram'] / 1024:.0f}ГБ  "
              f"${o['dph_total']:.3f}/ч  надёжн.{o.get('reliability2', 0):.2f}  "
              f"↓{o.get('inet_down', 0):.0f}Мбит  id={o['id']}")


def cmd_up(key: str, env_path: Path, env: dict, write_config: bool) -> None:
    token = ensure_token(env_path, env)
    existing = my_instances(key)
    if existing:
        inst = existing[0]
        if inst.get("actual_status") != "running":
            print(f"[vast] запускаю существующий инстанс {inst['id']}...")
            set_state(key, inst["id"], "running")
    else:
        offer = pick_offer(search_offers(key), min_disk=DISK_GB)
        if offer is None:
            print("[vast] нет подходящих карт — попробуй позже"); sys.exit(1)
        print(f"[vast] арендую {offer['gpu_name']} за ${offer['dph_total']:.3f}/ч...")
        create_instance(key, offer["id"], token)

    print("[vast] жду запуска (первый старт с загрузкой моделей — 15–25 минут)...")
    while True:
        time.sleep(15)
        insts = my_instances(key)
        if not insts:
            continue
        inst = insts[0]
        url = ws_url(inst)
        status = inst.get("actual_status")
        print(f"  статус: {status}")
        if status == "running" and url:
            print(f"[vast] инстанс работает: {url}")
            print(f"[vast] цена: ${inst.get('dph_total', 0):.3f}/ч | баланс: ${credit(key)}")
            if write_config:
                write_client_config(url, token)
            print("[vast] дождись, пока прогреются модели (см. status), затем запускай клиент")
            return


def cmd_status(key: str) -> None:
    insts = my_instances(key)
    if not insts:
        print("[vast] инстансов нет")
    for i in insts:
        print(f"  id={i['id']}  {i.get('gpu_name')}  {i.get('actual_status')}  "
              f"${i.get('dph_total', 0):.3f}/ч  {ws_url(i) or 'ещё нет адреса'}")
        url = ws_url(i)
        if url:
            health = url.replace("ws://", "http://").replace("/ws", "/health")
            try:
                h = httpx.get(health, timeout=5).json()
                print(f"    NOVA готова: клиентов {h['clients']}, простой {h['idle_s']}с")
            except Exception:
                print("    NOVA ещё грузит модели (или порт не открылся)")
    print(f"  баланс: ${credit(key)}")


def cmd_down(key: str, destroy_it: bool) -> None:
    for i in my_instances(key):
        if destroy_it:
            destroy(key, i["id"])
            print(f"[vast] инстанс {i['id']} УДАЛЁН (диск и кэш моделей стёрты)")
        else:
            set_state(key, i["id"], "stopped")
            print(f"[vast] инстанс {i['id']} остановлен (диск сохранён, ~$0.1-0.2/день за хранение)")
    print(f"  баланс: ${credit(key)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["search", "up", "status", "down"])
    parser.add_argument("--destroy", action="store_true")
    parser.add_argument("--write-config", action="store_true")
    args = parser.parse_args()

    env_path = ROOT / ".env"
    env = load_env(env_path)
    key = env.get("VAST_API_KEY", "")
    if not key:
        print("Нет VAST_API_KEY в .env"); sys.exit(1)

    if args.command == "search":
        cmd_search(key)
    elif args.command == "up":
        cmd_up(key, env_path, env, args.write_config)
    elif args.command == "status":
        cmd_status(key)
    elif args.command == "down":
        cmd_down(key, args.destroy)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Прогнать тесты**

Run: `uv run pytest -q`
Expected: все passed (+2 skipped)

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "feat: vast.ai instance management cli"
```

---

### Task 8: Провижининг инстанса и автостоп

**Files:**
- Create: `deploy/onstart.sh`, `deploy/idle_watchdog.py`
- Test: `tests/test_watchdog.py`

**Interfaces:**
- Consumes: `/health` (Task 6), env `VAST_API_KEY`, `VAST_CONTAINERLABEL` (Vast ставит сам, вид `C.12345`).
- Produces: `should_stop(clients: int, idle_s: float, limit_s: float = 900) -> bool`; `instance_id() -> str | None`.

- [ ] **Step 1: Написать падающий тест**

```python
# tests/test_watchdog.py
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "deploy"))

from idle_watchdog import instance_id, should_stop


def test_should_stop_only_when_idle_and_empty():
    assert should_stop(clients=0, idle_s=901)
    assert not should_stop(clients=1, idle_s=901)
    assert not should_stop(clients=0, idle_s=100)


def test_instance_id_parsed_from_label(monkeypatch):
    monkeypatch.setenv("VAST_CONTAINERLABEL", "C.12345")
    assert instance_id() == "12345"
    monkeypatch.delenv("VAST_CONTAINERLABEL")
    assert instance_id() is None
```

- [ ] **Step 2: Убедиться, что тест падает**

Run: `uv run pytest tests/test_watchdog.py -q`
Expected: FAIL — `No module named 'idle_watchdog'`

- [ ] **Step 3: Реализовать**

```python
# deploy/idle_watchdog.py
"""Автостоп: если к NOVA 15 минут никто не подключён — остановить инстанс,
чтобы не жечь деньги. Работает на самом инстансе."""
import os
import re
import time

import httpx

API = "https://console.vast.ai/api/v0"
IDLE_LIMIT_S = 900.0


def should_stop(clients: int, idle_s: float, limit_s: float = IDLE_LIMIT_S) -> bool:
    return clients == 0 and idle_s >= limit_s


def instance_id() -> str | None:
    m = re.search(r"(\d+)", os.environ.get("VAST_CONTAINERLABEL", ""))
    return m.group(1) if m else None


def main() -> None:
    key = os.environ.get("VAST_API_KEY", "")
    iid = instance_id()
    if not key or not iid:
        print("[watchdog] нет VAST_API_KEY или id инстанса — автостоп выключен")
        return
    print(f"[watchdog] слежу за простоем инстанса {iid} (лимит {IDLE_LIMIT_S:.0f}с)")
    while True:
        time.sleep(60)
        try:
            h = httpx.get("http://127.0.0.1:8000/health", timeout=5).json()
        except Exception:
            continue  # оркестратор ещё грузится
        if should_stop(h.get("clients", 0), h.get("idle_s", 0.0)):
            print("[watchdog] 15 минут простоя — останавливаю инстанс")
            httpx.put(
                f"{API}/instances/{iid}/",
                headers={"Authorization": f"Bearer {key}"},
                json={"state": "stopped"},
                timeout=30,
            )
            return


if __name__ == "__main__":
    main()
```

```bash
# deploy/onstart.sh
#!/bin/bash
# Провижининг инстанса Vast.ai (образ vllm/vllm-openai).
# Все тяжёлые данные — на /workspace (переживает stop/start).
set -x
export HF_HOME=/workspace/hf
export COQUI_TOS_AGREED=1

cd /workspace
if [ ! -d NOVA ]; then
  git clone https://github.com/MrJayfeather/NOVA.git
fi
cd NOVA && git pull

pip install -e . faster-whisper coqui-tts "nvidia-cudnn-cu12>=9" \
  > /workspace/pip.log 2>&1

# ctranslate2 (whisper) ищет cudnn в pip-пакете
export LD_LIBRARY_PATH="$(python3 -c 'import nvidia.cudnn, os; print(os.path.join(os.path.dirname(nvidia.cudnn.__file__), "lib"))'):$LD_LIBRARY_PATH"

# 1) vLLM с мозгом (Qwen3-VL). Первый старт качает ~31 ГБ весов.
nohup vllm serve Qwen/Qwen3-VL-30B-A3B-Instruct-FP8 \
  --host 127.0.0.1 --port 5000 \
  --max-model-len 16384 \
  --gpu-memory-utilization 0.75 \
  --limit-mm-per-prompt '{"image":12}' \
  > /workspace/vllm.log 2>&1 &

# ждём готовности vLLM, потом поднимаем оркестратор
until curl -s http://127.0.0.1:5000/v1/models > /dev/null; do sleep 10; done

cd /workspace/NOVA
nohup python3 -m nova.server.main > /workspace/nova.log 2>&1 &
nohup python3 deploy/idle_watchdog.py > /workspace/watchdog.log 2>&1 &
```

- [ ] **Step 4: Прогнать тесты**

Run: `uv run pytest -q`
Expected: все passed (+2 skipped)

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "feat: instance provisioning script and idle auto-stop watchdog"
```

---

### Task 9: README этапа 2

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Добавить раздел после «Тесты»**

```markdown
## Запуск с реальными моделями (GPU в облаке)

Один раз: положи `VAST_API_KEY` в `.env` (файл в .gitignore).

    # найти и запустить сервер (первый запуск 15–25 минут: качаются модели)
    uv run python scripts/vast.py up --write-config

    # проверить готовность
    uv run python scripts/vast.py status

    # запустить клиент (client_config.yaml уже настроен командой up)
    uv run python -m nova.client.main

    # закончил — остановить (иначе остановится сам через 15 минут простоя)
    uv run python scripts/vast.py down

`down` останавливает инстанс с сохранением диска (хранение ~$0.1–0.2/день,
зато повторный старт за ~2–3 минуты без скачивания моделей).
Полное удаление: `uv run python scripts/vast.py down --destroy`.

Голос NOVA: положи 10–15 с чистой записи голоса в
`personas/nova/voice_sample.wav` — клонируется автоматически при старте
сервера. Без файла используется встроенный женский голос.
```

- [ ] **Step 2: Commit**

```bash
git add README.md
git commit -m "docs: cloud gpu usage instructions"
```

---

### Task 10: Живой прогон (ручной, с реальными деньгами)

Чеклист первого настоящего запуска. Аренда ~$0.3–0.5/час — следить за `status`.

- [ ] **Step 1:** `uv run python scripts/vast.py search` — убедиться, что есть карты дешевле $0.5/ч.
- [ ] **Step 2:** `uv run python scripts/vast.py up --write-config` — дождаться URL.
- [ ] **Step 3:** Каждые ~5 минут `uv run python scripts/vast.py status` — пока не появится «NOVA готова» (первый раз 15–25 мин: vLLM качает веса).
- [ ] **Step 4:** `uv run python -m nova.client.main` — клиент подключается к облаку.
- [ ] **Step 5:** Сказать в микрофон «Привет, NOVA, ты меня слышишь?» → она отвечает **голосом**, по-русски, в характере. Замерить задержку по `data/metrics.jsonl` (цель ≤ 2 с до первого звука; тюнинг — этап 3).
- [ ] **Step 6:** Открыть видео/аниме на весь экран → в течение минуты она сама комментирует смену сцен. Проверить, что PASS-фильтр работает (не тараторит на статичном экране).
- [ ] **Step 7:** `Ctrl+Alt+C` — комментарий по запросу; `Ctrl+Alt+M` — mute; 👍/👎 пишутся в feedback.jsonl на сервере.
- [ ] **Step 8:** `uv run python scripts/vast.py down` — инстанс остановлен, баланс напечатан.
- [ ] **Step 9:** Проверить автостоп: `up`, не подключать клиента 16 минут, `status` → инстанс сам в `stopped`.
- [ ] **Step 10: Commit заметок** — если в ходе прогона менялись параметры (промпт, температура, кулдауны):

```bash
git add -A
git commit -m "tune: first live session adjustments"
```

---

## Вне скоупа этапа 2 (по спеке — дальше)

- Этап 3: RAG/память сессий, web-самообучение, игровые профили, голосовая регулировка болтливости, зоны внимания.
- Этап 4: Discord-бот. Этап 5: GUI настроек. Этап 6: LoRA-дообучение.
- Перебивание (включаемое), «шепнуть NOVA», спенд-предупреждения сверх показа баланса.
