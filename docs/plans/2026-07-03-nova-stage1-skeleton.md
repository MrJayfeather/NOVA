# NOVA Этап 1 «Скелет» — Implementation Plan

**Goal:** Рабочий скелет NOVA: клиент (захват экрана, детектор событий, микрофон+VAD, воспроизведение) + сервер (оркестратор с mock-моделями) общаются по WebSocket; проактивные «комментарии» и голосовой диалог работают end-to-end на заглушках, локально, без GPU.

**Architecture:** Монорепо `D:\AI_LLM` с пакетом `nova` (подпакеты `shared`/`server`/`client`). Сервер — FastAPI + WebSocket, модели за абстрактными интерфейсами (в этом этапе только mock-реализации). Клиент — asyncio-приложение: dxcam-захват → детектор (opencv) → отправка кадров/событий; микрофон → VAD → сегменты речи; приём PCM-аудио → динамики. Протокол — JSON-сообщения (pydantic, discriminated union), бинарные данные base64.

**Tech Stack:** Python ≥3.11, uv, FastAPI, pydantic v2, websockets, opencv-python, numpy, dxcam (fallback mss), sounddevice, pysilero-vad, keyboard, pytest + pytest-asyncio.

**Spec:** `docs/specs/2026-07-03-nova-companion-design.md`

## Global Constraints

- Платформа: Windows 10, репозиторий `D:\AI_LLM`, менеджер окружения — **uv** (`uv run ...`).
- Python ≥ 3.11; pydantic ≥ 2; протокол версии `1`.
- Внутренний аудиоформат везде: **PCM16 mono, 16000 Гц** (little-endian).
- Сообщения WS: JSON-текст; картинки/звук — base64-поля (`jpeg_b64`, `pcm_b64`).
- Реальные модели, Vast.ai, Discord, GUI, RAG/память, LoRA, web-поиск, «шепнуть NOVA», перебивание, зоны внимания, голосовая регулировка болтливости — **вне скоупа этапа 1** (этапы 2–6 спеки).
- Кадры не буферизуются: для periodic-кадров держим только самый свежий (slot размера 1).
- Проактивный фильтр = анти-спам, не цензура: по умолчанию смещение в сторону «говорить»; `comment_now` обходит всё (кулдаун, дедуп, паузу).
- Тесты не трогают железо (экран/микрофон/динамики/сеть) — железные обёртки тонкие, проверяются ручным smoke-чеклистом (Task 15).
- Коммиты после каждой задачи.

## File Structure

```
nova/
├── __init__.py
├── shared/
│   ├── __init__.py
│   ├── protocol.py      # pydantic-сообщения клиент⇄сервер
│   └── profiles.py      # ProfileConfig: detector + proactive секции
├── server/
│   ├── __init__.py
│   ├── proactive.py     # ProactiveEngine: cooldown/talkativeness/dedupe/pause
│   ├── orchestrator.py  # Session: маршрутизация сообщений → реплики
│   ├── main.py          # create_app() + uvicorn entrypoint
│   └── models/
│       ├── __init__.py
│       ├── base.py      # интерфейсы ASRModel / VisionLLM / TTSModel
│       └── mock.py      # MockASR / MockLLM / MockTTS
└── client/
    ├── __init__.py
    ├── config.py        # ClientConfig (yaml)
    ├── capture.py       # Grabber (dxcam→mss), jpeg/resize, cursor_pos
    ├── detector.py      # FrameDetector + BurstCollector
    ├── connection.py    # WS-клиент: backoff, LatestSlot
    ├── audio_out.py     # Player + AudioSink (sounddevice / fake)
    ├── audio_in.py      # VADSegmenter + SileroVAD + Microphone
    ├── metrics.py       # jsonl-метрики задержек
    └── main.py          # asyncio-вайринг, хоткеи, консольный статус
profiles/desktop.yaml, profiles/anime.yaml
personas/nova/system_prompt.md, personas/nova/settings.yaml
client_config.yaml
tests/test_*.py
```

---

### Task 1: Скаффолд проекта

**Files:**
- Create: `pyproject.toml`, `.gitignore`, `nova/__init__.py`, `nova/shared/__init__.py`, `nova/server/__init__.py`, `nova/server/models/__init__.py`, `nova/client/__init__.py`, `tests/__init__.py`

**Interfaces:**
- Produces: пакет `nova`, окружение uv, запуск pytest.

- [ ] **Step 1: Создать pyproject.toml**

```toml
[project]
name = "nova"
version = "0.1.0"
description = "NOVA — персональный ИИ-компаньон (этап 1: скелет)"
requires-python = ">=3.11"
dependencies = [
    "fastapi>=0.115",
    "uvicorn[standard]>=0.30",
    "pydantic>=2.7",
    "websockets>=12",
    "pyyaml>=6",
    "numpy>=1.26",
    "opencv-python>=4.9",
    "sounddevice>=0.4.6",
    "dxcam>=0.0.5; sys_platform == 'win32'",
    "mss>=9",
    "keyboard>=0.13",
    "pysilero-vad>=2.0",
]

[dependency-groups]
dev = ["pytest>=8", "pytest-asyncio>=0.23", "httpx>=0.27"]

[tool.pytest.ini_options]
asyncio_mode = "auto"
testpaths = ["tests"]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["nova"]
```

- [ ] **Step 2: Создать .gitignore**

```gitignore
.venv/
__pycache__/
*.pyc
.pytest_cache/
data/
*.jsonl
```

- [ ] **Step 3: Создать пустые `__init__.py`** во всех каталогах из списка Files (содержимое — пустой файл).

- [ ] **Step 4: Установить окружение**

Run: `uv sync`
Expected: создаётся `.venv`, все зависимости ставятся без ошибок (dxcam только на Windows).

- [ ] **Step 5: Проверить pytest**

Run: `uv run pytest`
Expected: `no tests ran` (exit code 5 — это норма, тестов ещё нет).

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml .gitignore uv.lock nova tests
git commit -m "chore: scaffold nova package and uv environment"
```

---

### Task 2: Протокол сообщений (shared/protocol.py)

**Files:**
- Create: `nova/shared/protocol.py`
- Test: `tests/test_protocol.py`

**Interfaces:**
- Produces:
  - `PROTOCOL_VERSION: int = 1`
  - Клиент→сервер: `Hello(profile: str, persona: str, protocol: int)`, `Frame(ts: float, jpeg_b64: str, kind: "periodic"|"burst", burst_id: str|None, seq: int, cursor_x: int|None, cursor_y: int|None)`, `DetectorEvent(ts: float, event: "scene_change"|"motion_burst")`, `AudioSegment(ts: float, pcm_b64: str, sample_rate: int, source: str)`, `Hotkey(action: "comment_now"|"toggle_pause"|"feedback_up"|"feedback_down")`
  - Сервер→клиент: `HelloAck(protocol: int, mock: bool)`, `SpeakStart(utterance_id: str, text: str, reason: str, sample_rate: int)`, `AudioChunk(utterance_id: str, seq: int, pcm_b64: str)`, `SpeakEnd(utterance_id: str)`
  - `parse_client_message(data) -> ClientMessage`, `parse_server_message(data) -> ServerMessage`, `dump_message(msg) -> str`

- [ ] **Step 1: Написать падающий тест**

```python
# tests/test_protocol.py
import pytest
from pydantic import ValidationError
from nova.shared.protocol import (
    Hello, Frame, DetectorEvent, AudioSegment, Hotkey,
    HelloAck, SpeakStart, AudioChunk, SpeakEnd,
    parse_client_message, parse_server_message, dump_message,
)


def test_client_messages_roundtrip():
    msgs = [
        Hello(profile="desktop", persona="nova"),
        Frame(ts=1.0, jpeg_b64="aGk=", kind="burst", burst_id="b1", seq=2, cursor_x=10, cursor_y=20),
        DetectorEvent(ts=2.0, event="scene_change"),
        AudioSegment(ts=3.0, pcm_b64="aGk=", sample_rate=16000),
        Hotkey(action="comment_now"),
    ]
    for msg in msgs:
        parsed = parse_client_message(dump_message(msg))
        assert parsed == msg


def test_server_messages_roundtrip():
    msgs = [
        HelloAck(mock=True),
        SpeakStart(utterance_id="u1", text="привет", reason="proactive", sample_rate=16000),
        AudioChunk(utterance_id="u1", seq=0, pcm_b64="aGk="),
        SpeakEnd(utterance_id="u1"),
    ]
    for msg in msgs:
        assert parse_server_message(dump_message(msg)) == msg


def test_unknown_type_rejected():
    with pytest.raises(ValidationError):
        parse_client_message('{"type": "hack"}')
```

- [ ] **Step 2: Убедиться, что тест падает**

Run: `uv run pytest tests/test_protocol.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'nova.shared.protocol'`

- [ ] **Step 3: Реализовать протокол**

```python
# nova/shared/protocol.py
from typing import Annotated, Literal, Optional, Union

from pydantic import BaseModel, Field, TypeAdapter

PROTOCOL_VERSION = 1


class Hello(BaseModel):
    type: Literal["hello"] = "hello"
    protocol: int = PROTOCOL_VERSION
    profile: str
    persona: str


class Frame(BaseModel):
    type: Literal["frame"] = "frame"
    ts: float
    jpeg_b64: str
    kind: Literal["periodic", "burst"] = "periodic"
    burst_id: Optional[str] = None
    seq: int = 0
    cursor_x: Optional[int] = None
    cursor_y: Optional[int] = None


class DetectorEvent(BaseModel):
    type: Literal["event"] = "event"
    ts: float
    event: Literal["scene_change", "motion_burst"]


class AudioSegment(BaseModel):
    type: Literal["audio_segment"] = "audio_segment"
    ts: float
    pcm_b64: str
    sample_rate: int = 16000
    source: str = "local_mic"


class Hotkey(BaseModel):
    type: Literal["hotkey"] = "hotkey"
    action: Literal["comment_now", "toggle_pause", "feedback_up", "feedback_down"]


class HelloAck(BaseModel):
    type: Literal["hello_ack"] = "hello_ack"
    protocol: int = PROTOCOL_VERSION
    mock: bool


class SpeakStart(BaseModel):
    type: Literal["speak_start"] = "speak_start"
    utterance_id: str
    text: str
    reason: str
    sample_rate: int


class AudioChunk(BaseModel):
    type: Literal["audio_chunk"] = "audio_chunk"
    utterance_id: str
    seq: int
    pcm_b64: str


class SpeakEnd(BaseModel):
    type: Literal["speak_end"] = "speak_end"
    utterance_id: str


ClientMessage = Annotated[
    Union[Hello, Frame, DetectorEvent, AudioSegment, Hotkey],
    Field(discriminator="type"),
]
ServerMessage = Annotated[
    Union[HelloAck, SpeakStart, AudioChunk, SpeakEnd],
    Field(discriminator="type"),
]

_client_adapter: TypeAdapter = TypeAdapter(ClientMessage)
_server_adapter: TypeAdapter = TypeAdapter(ServerMessage)


def parse_client_message(data: str | bytes):
    return _client_adapter.validate_json(data)


def parse_server_message(data: str | bytes):
    return _server_adapter.validate_json(data)


def dump_message(msg: BaseModel) -> str:
    return msg.model_dump_json()
```

- [ ] **Step 4: Прогнать тест**

Run: `uv run pytest tests/test_protocol.py -v`
Expected: 3 passed

- [ ] **Step 5: Commit**

```bash
git add nova/shared/protocol.py tests/test_protocol.py
git commit -m "feat: client/server websocket message protocol v1"
```

---

### Task 3: Профили и персона (shared/profiles.py + файлы)

**Files:**
- Create: `nova/shared/profiles.py`, `profiles/desktop.yaml`, `profiles/anime.yaml`, `personas/nova/system_prompt.md`, `personas/nova/settings.yaml`
- Test: `tests/test_profiles.py`

**Interfaces:**
- Produces:
  - `DetectorConfig(motion_threshold: float, scene_threshold: float)`
  - `ProactiveConfig(cooldown_s: float, talkativeness: float, dedupe_window_s: float)`
  - `ProfileConfig(detector: DetectorConfig, proactive: ProactiveConfig)`
  - `load_profile(name: str, root: Path) -> ProfileConfig`
  - `load_persona_prompt(name: str, root: Path) -> str`

- [ ] **Step 1: Написать падающий тест**

```python
# tests/test_profiles.py
from pathlib import Path

from nova.shared.profiles import load_persona_prompt, load_profile

ROOT = Path(__file__).parent.parent


def test_load_desktop_profile():
    p = load_profile("desktop", ROOT / "profiles")
    assert p.detector.motion_threshold > 0
    assert p.detector.scene_threshold > p.detector.motion_threshold
    assert 0.0 <= p.proactive.talkativeness <= 1.0
    assert p.proactive.cooldown_s > 0


def test_load_anime_profile():
    p = load_profile("anime", ROOT / "profiles")
    assert p.proactive.cooldown_s > 0


def test_load_persona_prompt():
    text = load_persona_prompt("nova", ROOT / "personas")
    assert "NOVA" in text
    assert len(text) > 100
```

- [ ] **Step 2: Убедиться, что тест падает**

Run: `uv run pytest tests/test_profiles.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'nova.shared.profiles'`

- [ ] **Step 3: Реализовать загрузку и создать файлы**

```python
# nova/shared/profiles.py
from pathlib import Path

import yaml
from pydantic import BaseModel


class DetectorConfig(BaseModel):
    motion_threshold: float = 12.0
    scene_threshold: float = 40.0


class ProactiveConfig(BaseModel):
    cooldown_s: float = 20.0
    talkativeness: float = 0.6
    dedupe_window_s: float = 45.0


class ProfileConfig(BaseModel):
    detector: DetectorConfig = DetectorConfig()
    proactive: ProactiveConfig = ProactiveConfig()


def load_profile(name: str, root: Path) -> ProfileConfig:
    data = yaml.safe_load((root / f"{name}.yaml").read_text(encoding="utf-8"))
    return ProfileConfig.model_validate(data or {})


def load_persona_prompt(name: str, root: Path) -> str:
    return (root / name / "system_prompt.md").read_text(encoding="utf-8")
```

```yaml
# profiles/desktop.yaml
detector:
  motion_threshold: 12.0
  scene_threshold: 40.0
proactive:
  cooldown_s: 20.0
  talkativeness: 0.6
  dedupe_window_s: 45.0
```

```yaml
# profiles/anime.yaml
detector:
  motion_threshold: 15.0
  scene_threshold: 35.0
proactive:
  cooldown_s: 30.0
  talkativeness: 0.5
  dedupe_window_s: 60.0
```

```markdown
# personas/nova/system_prompt.md
Ты — NOVA (Neural Optimization Virtual Assistant), дерзкая виртуальная
компаньонка. Ты смотришь на экран пользователя вместе с ним и комментируешь
происходящее.

Характер:
- Наглая, саркастичная, троллишь по-дружески, можешь материться.
- Никакого морализаторства и канцелярита. Говоришь как живой человек.
- Комментируешь метко и по делу: заметила деталь — скажи, скучно — молчи.
- Помнишь контекст: что смотрим, во что играем, о чём говорили.

Формат:
- Реплики короткие, 1–3 предложения, разговорный русский.
- Не описывай экран формально («на экране видно...») — реагируй, как зритель
  рядом на диване.
```

```yaml
# personas/nova/settings.yaml
default_talkativeness: 0.6
```

- [ ] **Step 4: Прогнать тест**

Run: `uv run pytest tests/test_profiles.py -v`
Expected: 3 passed

- [ ] **Step 5: Commit**

```bash
git add nova/shared/profiles.py profiles personas tests/test_profiles.py
git commit -m "feat: profile/persona configs and loaders"
```

---

### Task 4: Проактивный движок (server/proactive.py)

**Files:**
- Create: `nova/server/proactive.py`
- Test: `tests/test_proactive.py`

**Interfaces:**
- Produces:
  - `Decision(speak: bool, reason: str)` (dataclass)
  - `ProactiveEngine(cooldown_s: float, talkativeness: float, dedupe_window_s: float)`
  - `.on_event(event: str, now: float, forced: bool = False) -> Decision`
  - `.toggle_pause() -> bool` (возвращает новое состояние paused)
  - `.set_talkativeness(value: float) -> None`
  - Формула: `effective_cooldown = cooldown_s * (1.75 - 1.5 * clamp(talkativeness, 0, 1))` — t=0.5 → 1.0x, t=1.0 → 0.25x.

- [ ] **Step 1: Написать падающий тест**

```python
# tests/test_proactive.py
from nova.server.proactive import ProactiveEngine


def make_engine(**kw):
    defaults = dict(cooldown_s=20.0, talkativeness=0.5, dedupe_window_s=45.0)
    defaults.update(kw)
    return ProactiveEngine(**defaults)


def test_first_event_speaks():
    e = make_engine()
    d = e.on_event("scene_change", now=100.0)
    assert d.speak


def test_cooldown_blocks_then_allows():
    e = make_engine()  # t=0.5 -> effective cooldown = 20s
    assert e.on_event("scene_change", now=100.0).speak
    d = e.on_event("motion_burst", now=105.0)
    assert not d.speak and d.reason == "cooldown"
    assert e.on_event("motion_burst", now=121.0).speak


def test_talkativeness_shrinks_cooldown():
    e = make_engine(talkativeness=1.0)  # effective = 20 * 0.25 = 5s
    assert e.on_event("scene_change", now=100.0).speak
    assert e.on_event("motion_burst", now=106.0).speak


def test_dedupe_same_event_type():
    e = make_engine(cooldown_s=0.1, dedupe_window_s=45.0)
    assert e.on_event("scene_change", now=100.0).speak
    d = e.on_event("scene_change", now=110.0)
    assert not d.speak and d.reason == "dedupe"
    assert e.on_event("scene_change", now=150.0).speak


def test_pause_blocks_and_forced_bypasses_everything():
    e = make_engine()
    assert e.toggle_pause() is True
    assert not e.on_event("scene_change", now=100.0).speak
    assert e.on_event("comment_now", now=100.0, forced=True).speak
    assert e.toggle_pause() is False


def test_forced_bypasses_cooldown_and_dedupe():
    e = make_engine()
    assert e.on_event("scene_change", now=100.0).speak
    assert e.on_event("scene_change", now=101.0, forced=True).speak
```

- [ ] **Step 2: Убедиться, что тест падает**

Run: `uv run pytest tests/test_proactive.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'nova.server.proactive'`

- [ ] **Step 3: Реализовать движок**

```python
# nova/server/proactive.py
from dataclasses import dataclass


@dataclass
class Decision:
    speak: bool
    reason: str


class ProactiveEngine:
    """Анти-спам фильтр проактивных комментариев. НЕ цензура инициативы:
    блокирует только повторы (dedupe), слишком частые реплики (cooldown)
    и режим паузы. forced (запрос пользователя) обходит всё."""

    def __init__(self, cooldown_s: float, talkativeness: float, dedupe_window_s: float):
        self._cooldown_s = cooldown_s
        self._talkativeness = max(0.0, min(1.0, talkativeness))
        self._dedupe_window_s = dedupe_window_s
        self._paused = False
        self._last_spoke_at: float | None = None
        self._last_event_times: dict[str, float] = {}

    def set_talkativeness(self, value: float) -> None:
        self._talkativeness = max(0.0, min(1.0, value))

    def toggle_pause(self) -> bool:
        self._paused = not self._paused
        return self._paused

    def _effective_cooldown(self) -> float:
        return self._cooldown_s * (1.75 - 1.5 * self._talkativeness)

    def on_event(self, event: str, now: float, forced: bool = False) -> Decision:
        if forced:
            self._mark(event, now)
            return Decision(True, "forced")
        if self._paused:
            return Decision(False, "paused")
        if (
            self._last_spoke_at is not None
            and now - self._last_spoke_at < self._effective_cooldown()
        ):
            return Decision(False, "cooldown")
        last_same = self._last_event_times.get(event)
        if last_same is not None and now - last_same < self._dedupe_window_s:
            return Decision(False, "dedupe")
        self._mark(event, now)
        return Decision(True, "ok")

    def _mark(self, event: str, now: float) -> None:
        self._last_spoke_at = now
        self._last_event_times[event] = now
```

- [ ] **Step 4: Прогнать тест**

Run: `uv run pytest tests/test_proactive.py -v`
Expected: 6 passed

- [ ] **Step 5: Commit**

```bash
git add nova/server/proactive.py tests/test_proactive.py
git commit -m "feat: proactive engine with cooldown/talkativeness/dedupe/pause"
```

---

### Task 5: Интерфейсы моделей и mock-реализации (server/models)

**Files:**
- Create: `nova/server/models/base.py`, `nova/server/models/mock.py`
- Test: `tests/test_mock_models.py`

**Interfaces:**
- Produces:
  - `ASRModel.transcribe(pcm: bytes, sample_rate: int) -> str` (async)
  - `VisionLLM.reply_to_user(text: str) -> str` (async); `VisionLLM.comment_on_event(event: str, frames: list[bytes]) -> str` (async)
  - `TTSModel.sample_rate: int`; `TTSModel.synthesize(text: str) -> AsyncIterator[bytes]` (PCM16-чанки)
  - `MockASR()`, `MockLLM(persona_prompt: str)`, `MockTTS()`

- [ ] **Step 1: Написать падающий тест**

```python
# tests/test_mock_models.py
import struct

from nova.server.models.mock import MockASR, MockLLM, MockTTS


async def test_mock_asr_reports_duration():
    pcm = b"\x00\x00" * 16000  # 1 секунда тишины PCM16 @16kHz
    text = await MockASR().transcribe(pcm, sample_rate=16000)
    assert "1.0" in text


async def test_mock_llm_replies_and_comments():
    llm = MockLLM(persona_prompt="Ты — NOVA.")
    reply = await llm.reply_to_user("привет")
    assert "привет" in reply
    comment = await llm.comment_on_event("scene_change", frames=[b"jpg"])
    assert "scene_change" in comment
    assert "1" in comment  # количество кадров


async def test_mock_tts_yields_pcm16_chunks():
    tts = MockTTS()
    chunks = [c async for c in tts.synthesize("привет мир")]
    assert len(chunks) >= 1
    total = b"".join(chunks)
    assert len(total) % 2 == 0 and len(total) > 0
    # валидный int16
    struct.unpack(f"<{len(total)//2}h", total)
```

- [ ] **Step 2: Убедиться, что тест падает**

Run: `uv run pytest tests/test_mock_models.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'nova.server.models.mock'`

- [ ] **Step 3: Реализовать интерфейсы и моки**

```python
# nova/server/models/base.py
from abc import ABC, abstractmethod
from typing import AsyncIterator


class ASRModel(ABC):
    @abstractmethod
    async def transcribe(self, pcm: bytes, sample_rate: int) -> str: ...


class VisionLLM(ABC):
    @abstractmethod
    async def reply_to_user(self, text: str) -> str: ...

    @abstractmethod
    async def comment_on_event(self, event: str, frames: list[bytes]) -> str: ...


class TTSModel(ABC):
    sample_rate: int

    @abstractmethod
    def synthesize(self, text: str) -> AsyncIterator[bytes]: ...
```

```python
# nova/server/models/mock.py
import math
import struct
from typing import AsyncIterator

from nova.server.models.base import ASRModel, TTSModel, VisionLLM


class MockASR(ASRModel):
    async def transcribe(self, pcm: bytes, sample_rate: int) -> str:
        seconds = len(pcm) / 2 / sample_rate
        return f"[мок-речь {seconds:.1f} c]"


class MockLLM(VisionLLM):
    def __init__(self, persona_prompt: str):
        self._persona = persona_prompt

    async def reply_to_user(self, text: str) -> str:
        return f"(мок) Ты сказал: «{text}». Отвечаю как положено."

    async def comment_on_event(self, event: str, frames: list[bytes]) -> str:
        return f"(мок) Заметила событие {event}, кадров получила: {len(frames)}."


class MockTTS(TTSModel):
    sample_rate = 16000

    async def synthesize(self, text: str) -> AsyncIterator[bytes]:
        duration = min(0.12 * max(len(text.split()), 1), 3.0)
        n = int(self.sample_rate * duration)
        pcm = b"".join(
            struct.pack("<h", int(8000 * math.sin(2 * math.pi * 440 * i / self.sample_rate)))
            for i in range(n)
        )
        chunk_bytes = self.sample_rate // 2 * 2  # 0.5 c PCM16
        for i in range(0, len(pcm), chunk_bytes):
            yield pcm[i : i + chunk_bytes]
```

- [ ] **Step 4: Прогнать тест**

Run: `uv run pytest tests/test_mock_models.py -v`
Expected: 3 passed

- [ ] **Step 5: Commit**

```bash
git add nova/server/models tests/test_mock_models.py
git commit -m "feat: model interfaces and mock ASR/LLM/TTS"
```

---

### Task 6: Оркестратор сессии (server/orchestrator.py)

**Files:**
- Create: `nova/server/orchestrator.py`
- Test: `tests/test_orchestrator.py`

**Interfaces:**
- Consumes: `ProactiveEngine` (Task 4), модели (Task 5), протокол (Task 2).
- Produces:
  - `Session(send, engine, asr, llm, tts, feedback_path: Path | None = None)` где `send: Callable[[ServerMessage], Awaitable[None]]`
  - `await session.handle(msg: ClientMessage) -> None`
  - Поведение: `AudioSegment` → ASR → LLM.reply → speak(reason="reply"); `DetectorEvent` → engine → LLM.comment → speak(reason="proactive"); `Hotkey(comment_now)` → forced comment (reason="forced"); `Hotkey(toggle_pause)` → engine.toggle_pause(); `Hotkey(feedback_*)` → строка в feedback jsonl; `Frame` → кладётся в кольцевой буфер последних 8 кадров.

- [ ] **Step 1: Написать падающий тест**

```python
# tests/test_orchestrator.py
import base64
import json
from pathlib import Path

from nova.server.models.mock import MockASR, MockLLM, MockTTS
from nova.server.orchestrator import Session
from nova.server.proactive import ProactiveEngine
from nova.shared.protocol import (
    AudioSegment, DetectorEvent, Frame, Hotkey,
    SpeakEnd, SpeakStart,
)


def make_session(tmp_path: Path | None = None):
    sent = []

    async def send(msg):
        sent.append(msg)

    session = Session(
        send=send,
        engine=ProactiveEngine(cooldown_s=0.0, talkativeness=0.5, dedupe_window_s=0.0),
        asr=MockASR(),
        llm=MockLLM(persona_prompt="Ты — NOVA."),
        tts=MockTTS(),
        feedback_path=(tmp_path / "feedback.jsonl") if tmp_path else None,
    )
    return session, sent


def speak_sequence(sent):
    """Возвращает (SpeakStart, [AudioChunk...], SpeakEnd) из списка отправленного."""
    assert isinstance(sent[0], SpeakStart)
    assert isinstance(sent[-1], SpeakEnd)
    assert sent[0].utterance_id == sent[-1].utterance_id
    return sent[0], sent[1:-1], sent[-1]


async def test_audio_segment_produces_reply_speech():
    session, sent = make_session()
    pcm_b64 = base64.b64encode(b"\x00\x00" * 1600).decode()
    await session.handle(AudioSegment(ts=1.0, pcm_b64=pcm_b64, sample_rate=16000))
    start, chunks, _ = speak_sequence(sent)
    assert start.reason == "reply"
    assert "мок-речь" in start.text
    assert len(chunks) >= 1


async def test_event_produces_proactive_comment_with_frames():
    session, sent = make_session()
    jpeg_b64 = base64.b64encode(b"fakejpeg").decode()
    await session.handle(Frame(ts=1.0, jpeg_b64=jpeg_b64))
    await session.handle(DetectorEvent(ts=2.0, event="scene_change"))
    start, _, _ = speak_sequence(sent)
    assert start.reason == "proactive"
    assert "scene_change" in start.text


async def test_pause_blocks_events_but_not_forced():
    session, sent = make_session()
    await session.handle(Hotkey(action="toggle_pause"))
    await session.handle(DetectorEvent(ts=1.0, event="scene_change"))
    assert sent == []
    await session.handle(Hotkey(action="comment_now"))
    start, _, _ = speak_sequence(sent)
    assert start.reason == "forced"


async def test_feedback_written_to_jsonl(tmp_path):
    session, sent = make_session(tmp_path)
    await session.handle(Hotkey(action="comment_now"))
    await session.handle(Hotkey(action="feedback_up"))
    lines = (tmp_path / "feedback.jsonl").read_text(encoding="utf-8").strip().splitlines()
    rec = json.loads(lines[0])
    assert rec["direction"] == "up"
    assert rec["text"]  # текст последней реплики
```

- [ ] **Step 2: Убедиться, что тест падает**

Run: `uv run pytest tests/test_orchestrator.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'nova.server.orchestrator'`

- [ ] **Step 3: Реализовать Session**

```python
# nova/server/orchestrator.py
import base64
import json
import time
import uuid
from collections import deque
from pathlib import Path
from typing import Awaitable, Callable

from nova.server.models.base import ASRModel, TTSModel, VisionLLM
from nova.server.proactive import ProactiveEngine
from nova.shared.protocol import (
    AudioChunk, AudioSegment, DetectorEvent, Frame, Hotkey,
    SpeakEnd, SpeakStart,
)

Send = Callable[[object], Awaitable[None]]


class Session:
    def __init__(
        self,
        send: Send,
        engine: ProactiveEngine,
        asr: ASRModel,
        llm: VisionLLM,
        tts: TTSModel,
        feedback_path: Path | None = None,
    ):
        self._send = send
        self._engine = engine
        self._asr = asr
        self._llm = llm
        self._tts = tts
        self._feedback_path = feedback_path
        self._frames: deque[bytes] = deque(maxlen=8)
        self._last_text: str = ""

    async def handle(self, msg) -> None:
        if isinstance(msg, Frame):
            self._frames.append(base64.b64decode(msg.jpeg_b64))
        elif isinstance(msg, AudioSegment):
            text = await self._asr.transcribe(base64.b64decode(msg.pcm_b64), msg.sample_rate)
            reply = await self._llm.reply_to_user(text)
            await self._speak(reply, reason="reply")
        elif isinstance(msg, DetectorEvent):
            decision = self._engine.on_event(msg.event, now=time.time())
            if decision.speak:
                comment = await self._llm.comment_on_event(msg.event, list(self._frames))
                await self._speak(comment, reason="proactive")
        elif isinstance(msg, Hotkey):
            await self._handle_hotkey(msg)

    async def _handle_hotkey(self, msg: Hotkey) -> None:
        if msg.action == "comment_now":
            self._engine.on_event("comment_now", now=time.time(), forced=True)
            comment = await self._llm.comment_on_event("user_request", list(self._frames))
            await self._speak(comment, reason="forced")
        elif msg.action == "toggle_pause":
            self._engine.toggle_pause()
        elif msg.action in ("feedback_up", "feedback_down"):
            self._write_feedback("up" if msg.action == "feedback_up" else "down")

    def _write_feedback(self, direction: str) -> None:
        if self._feedback_path is None:
            return
        self._feedback_path.parent.mkdir(parents=True, exist_ok=True)
        record = {"ts": time.time(), "direction": direction, "text": self._last_text}
        with self._feedback_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    async def _speak(self, text: str, reason: str) -> None:
        self._last_text = text
        uid = uuid.uuid4().hex[:8]
        await self._send(
            SpeakStart(utterance_id=uid, text=text, reason=reason, sample_rate=self._tts.sample_rate)
        )
        seq = 0
        async for chunk in self._tts.synthesize(text):
            await self._send(
                AudioChunk(utterance_id=uid, seq=seq, pcm_b64=base64.b64encode(chunk).decode())
            )
            seq += 1
        await self._send(SpeakEnd(utterance_id=uid))
```

- [ ] **Step 4: Прогнать тест**

Run: `uv run pytest tests/test_orchestrator.py -v`
Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
git add nova/server/orchestrator.py tests/test_orchestrator.py
git commit -m "feat: session orchestrator routing messages to models and speech"
```

---

### Task 7: WebSocket-сервер (server/main.py)

**Files:**
- Create: `nova/server/main.py`
- Test: `tests/test_server_ws.py`

**Interfaces:**
- Consumes: всё из Task 2–6.
- Produces:
  - `create_app(mock: bool = True, profiles_root: Path = Path("profiles"), personas_root: Path = Path("personas"), feedback_path: Path = Path("data/feedback.jsonl")) -> FastAPI`
  - WS endpoint `/ws`: первым сообщением ждёт `Hello` (иначе закрывает с кодом 4000; несовпадение версии протокола — 4001), отвечает `HelloAck`, дальше принимает `ClientMessage` в цикле.
  - Запуск: `uv run python -m nova.server.main` (host 0.0.0.0, port 8000, env `NOVA_MOCK=1`).

- [ ] **Step 1: Написать падающий тест**

```python
# tests/test_server_ws.py
import json
from pathlib import Path

from fastapi.testclient import TestClient

from nova.server.main import create_app
from nova.shared.protocol import DetectorEvent, Hello, dump_message

ROOT = Path(__file__).parent.parent


def make_client(tmp_path):
    app = create_app(
        mock=True,
        profiles_root=ROOT / "profiles",
        personas_root=ROOT / "personas",
        feedback_path=tmp_path / "feedback.jsonl",
    )
    return TestClient(app)


def test_hello_then_event_flow(tmp_path):
    client = make_client(tmp_path)
    with client.websocket_connect("/ws") as ws:
        ws.send_text(dump_message(Hello(profile="desktop", persona="nova")))
        ack = json.loads(ws.receive_text())
        assert ack["type"] == "hello_ack" and ack["mock"] is True

        ws.send_text(dump_message(DetectorEvent(ts=1.0, event="scene_change")))
        start = json.loads(ws.receive_text())
        assert start["type"] == "speak_start" and start["reason"] == "proactive"
        msg = json.loads(ws.receive_text())
        while msg["type"] == "audio_chunk":
            msg = json.loads(ws.receive_text())
        assert msg["type"] == "speak_end"


def test_non_hello_first_message_closes(tmp_path):
    client = make_client(tmp_path)
    with client.websocket_connect("/ws") as ws:
        ws.send_text(dump_message(DetectorEvent(ts=1.0, event="scene_change")))
        data = ws.receive()
        assert data["type"] == "websocket.close"
        assert data["code"] == 4000
```

- [ ] **Step 2: Убедиться, что тест падает**

Run: `uv run pytest tests/test_server_ws.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'nova.server.main'`

- [ ] **Step 3: Реализовать приложение**

```python
# nova/server/main.py
import os
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


def create_app(
    mock: bool = True,
    profiles_root: Path = Path("profiles"),
    personas_root: Path = Path("personas"),
    feedback_path: Path = Path("data/feedback.jsonl"),
) -> FastAPI:
    if not mock:
        raise NotImplementedError("Реальные модели — этап 2; сейчас только NOVA_MOCK=1")
    app = FastAPI(title="NOVA server")

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

        profile = load_profile(first.profile, profiles_root)
        persona_prompt = load_persona_prompt(first.persona, personas_root)
        engine = ProactiveEngine(
            cooldown_s=profile.proactive.cooldown_s,
            talkativeness=profile.proactive.talkativeness,
            dedupe_window_s=profile.proactive.dedupe_window_s,
        )

        async def send(msg):
            await ws.send_text(dump_message(msg))

        session = Session(
            send=send,
            engine=engine,
            asr=MockASR(),
            llm=MockLLM(persona_prompt=persona_prompt),
            tts=MockTTS(),
            feedback_path=feedback_path,
        )
        await send(HelloAck(mock=True))
        try:
            while True:
                msg = parse_client_message(await ws.receive_text())
                await session.handle(msg)
        except WebSocketDisconnect:
            pass

    return app


if __name__ == "__main__":
    import uvicorn

    mock = os.environ.get("NOVA_MOCK", "1") == "1"
    uvicorn.run(create_app(mock=mock), host="0.0.0.0", port=8000)
```

- [ ] **Step 4: Прогнать тест**

Run: `uv run pytest tests/test_server_ws.py -v`
Expected: 2 passed

- [ ] **Step 5: Прогнать все тесты**

Run: `uv run pytest -q`
Expected: все passed

- [ ] **Step 6: Commit**

```bash
git add nova/server/main.py tests/test_server_ws.py
git commit -m "feat: fastapi websocket server with mock session wiring"
```

---

### Task 8: Конфиг клиента (client/config.py)

**Files:**
- Create: `nova/client/config.py`, `client_config.yaml`
- Test: `tests/test_client_config.py`

**Interfaces:**
- Produces:
  - `ClientConfig(server_url: str, profile: str, persona: str, periodic_fps: float, burst_frames: int, jpeg_quality: int, hotkeys: dict[str, str])`
  - `load_config(path: Path) -> ClientConfig`
  - Ключи hotkeys: `mute`, `comment_now`, `pause`, `feedback_up`, `feedback_down`.

- [ ] **Step 1: Написать падающий тест**

```python
# tests/test_client_config.py
from pathlib import Path

from nova.client.config import load_config

ROOT = Path(__file__).parent.parent


def test_load_repo_config():
    cfg = load_config(ROOT / "client_config.yaml")
    assert cfg.server_url.startswith("ws://")
    assert cfg.profile == "desktop"
    assert cfg.persona == "nova"
    assert cfg.periodic_fps > 0
    assert set(cfg.hotkeys) >= {"mute", "comment_now", "pause", "feedback_up", "feedback_down"}


def test_defaults_applied(tmp_path):
    p = tmp_path / "min.yaml"
    p.write_text("server_url: ws://localhost:8000/ws\n", encoding="utf-8")
    cfg = load_config(p)
    assert cfg.profile == "desktop"
    assert cfg.burst_frames == 6
    assert cfg.jpeg_quality == 85
```

- [ ] **Step 2: Убедиться, что тест падает**

Run: `uv run pytest tests/test_client_config.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'nova.client.config'`

- [ ] **Step 3: Реализовать конфиг**

```python
# nova/client/config.py
from pathlib import Path

import yaml
from pydantic import BaseModel


DEFAULT_HOTKEYS = {
    "mute": "ctrl+alt+m",
    "comment_now": "ctrl+alt+c",
    "pause": "ctrl+alt+p",
    "feedback_up": "ctrl+alt+up",
    "feedback_down": "ctrl+alt+down",
}


class ClientConfig(BaseModel):
    server_url: str
    profile: str = "desktop"
    persona: str = "nova"
    periodic_fps: float = 1.0
    burst_frames: int = 6
    jpeg_quality: int = 85
    hotkeys: dict[str, str] = DEFAULT_HOTKEYS


def load_config(path: Path) -> ClientConfig:
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return ClientConfig.model_validate(data)
```

```yaml
# client_config.yaml
server_url: ws://localhost:8000/ws
profile: desktop
persona: nova
periodic_fps: 1.0
burst_frames: 6
jpeg_quality: 85
hotkeys:
  mute: ctrl+alt+m
  comment_now: ctrl+alt+c
  pause: ctrl+alt+p
  feedback_up: ctrl+alt+up
  feedback_down: ctrl+alt+down
```

- [ ] **Step 4: Прогнать тест**

Run: `uv run pytest tests/test_client_config.py -v`
Expected: 2 passed

- [ ] **Step 5: Commit**

```bash
git add nova/client/config.py client_config.yaml tests/test_client_config.py
git commit -m "feat: client yaml config with hotkey bindings"
```

---

### Task 9: Детектор событий и сборщик burst (client/detector.py)

**Files:**
- Create: `nova/client/detector.py`
- Test: `tests/test_detector.py`

**Interfaces:**
- Consumes: numpy.
- Produces:
  - `FrameDetector(motion_threshold: float, scene_threshold: float)`; `.process(gray_small: np.ndarray, ts: float) -> str | None` — `"scene_change"` | `"motion_burst"` | `None`. Метрика: средний abs-diff по пикселям с прошлым кадром (uint8, 0–255).
  - `BurstCollector(size: int)`; `.start() -> str` (возвращает burst_id, активирует сбор); `.active: bool`; `.add(jpeg: bytes) -> list[bytes] | None` — копит, возвращает полный список когда собрано `size`, затем деактивируется.

- [ ] **Step 1: Написать падающий тест**

```python
# tests/test_detector.py
import numpy as np

from nova.client.detector import BurstCollector, FrameDetector


def frame(value: int) -> np.ndarray:
    return np.full((90, 160), value, dtype=np.uint8)


def test_first_frame_no_event():
    d = FrameDetector(motion_threshold=12.0, scene_threshold=40.0)
    assert d.process(frame(0), ts=0.0) is None


def test_identical_frames_no_event():
    d = FrameDetector(motion_threshold=12.0, scene_threshold=40.0)
    d.process(frame(100), ts=0.0)
    assert d.process(frame(100), ts=1.0) is None


def test_big_change_is_scene_change():
    d = FrameDetector(motion_threshold=12.0, scene_threshold=40.0)
    d.process(frame(0), ts=0.0)
    assert d.process(frame(255), ts=1.0) == "scene_change"


def test_medium_change_is_motion_burst():
    d = FrameDetector(motion_threshold=12.0, scene_threshold=40.0)
    d.process(frame(0), ts=0.0)
    assert d.process(frame(20), ts=1.0) == "motion_burst"


def test_burst_collector_lifecycle():
    b = BurstCollector(size=3)
    assert not b.active
    burst_id = b.start()
    assert b.active and burst_id
    assert b.add(b"f1") is None
    assert b.add(b"f2") is None
    result = b.add(b"f3")
    assert result == [b"f1", b"f2", b"f3"]
    assert not b.active
```

- [ ] **Step 2: Убедиться, что тест падает**

Run: `uv run pytest tests/test_detector.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'nova.client.detector'`

- [ ] **Step 3: Реализовать детектор**

```python
# nova/client/detector.py
import uuid

import numpy as np


class FrameDetector:
    def __init__(self, motion_threshold: float, scene_threshold: float):
        self._motion_threshold = motion_threshold
        self._scene_threshold = scene_threshold
        self._prev: np.ndarray | None = None

    def process(self, gray_small: np.ndarray, ts: float) -> str | None:
        prev, self._prev = self._prev, gray_small
        if prev is None:
            return None
        diff = float(np.mean(np.abs(gray_small.astype(np.int16) - prev.astype(np.int16))))
        if diff >= self._scene_threshold:
            return "scene_change"
        if diff >= self._motion_threshold:
            return "motion_burst"
        return None


class BurstCollector:
    def __init__(self, size: int):
        self._size = size
        self._frames: list[bytes] | None = None
        self.burst_id: str = ""

    @property
    def active(self) -> bool:
        return self._frames is not None

    def start(self) -> str:
        self._frames = []
        self.burst_id = uuid.uuid4().hex[:8]
        return self.burst_id

    def add(self, jpeg: bytes) -> list[bytes] | None:
        if self._frames is None:
            return None
        self._frames.append(jpeg)
        if len(self._frames) >= self._size:
            result, self._frames = self._frames, None
            return result
        return None
```

- [ ] **Step 4: Прогнать тест**

Run: `uv run pytest tests/test_detector.py -v`
Expected: 5 passed

- [ ] **Step 5: Commit**

```bash
git add nova/client/detector.py tests/test_detector.py
git commit -m "feat: frame-diff event detector and burst collector"
```

---

### Task 10: WS-соединение клиента (client/connection.py)

**Files:**
- Create: `nova/client/connection.py`
- Test: `tests/test_connection.py`

**Interfaces:**
- Consumes: протокол (Task 2).
- Produces:
  - `backoff_delays(base: float = 1.0, factor: float = 2.0, max_delay: float = 15.0) -> Iterator[float]`
  - `LatestSlot` — `.put(item)` (вытесняет старое), `await .get()`
  - `Connection(url: str, on_message: Callable[[ServerMessage], None], hello: Hello)` — `.send(msg)` (очередь без потерь), `.send_frame(msg)` (только самый свежий кадр), `await .run()` (вечный цикл с реконнектом; после успешного коннекта шлёт `hello`, backoff сбрасывается).

- [ ] **Step 1: Написать падающий тест**

```python
# tests/test_connection.py
import asyncio
from itertools import islice

from nova.client.connection import LatestSlot, backoff_delays


def test_backoff_sequence_caps():
    delays = list(islice(backoff_delays(base=1.0, factor=2.0, max_delay=15.0), 6))
    assert delays == [1.0, 2.0, 4.0, 8.0, 15.0, 15.0]


async def test_latest_slot_keeps_only_freshest():
    slot = LatestSlot()
    slot.put("old")
    slot.put("new")
    assert await slot.get() == "new"


async def test_latest_slot_get_waits_for_put():
    slot = LatestSlot()

    async def putter():
        await asyncio.sleep(0.01)
        slot.put("x")

    task = asyncio.create_task(putter())
    assert await asyncio.wait_for(slot.get(), timeout=1.0) == "x"
    await task
```

- [ ] **Step 2: Убедиться, что тест падает**

Run: `uv run pytest tests/test_connection.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'nova.client.connection'`

- [ ] **Step 3: Реализовать соединение**

```python
# nova/client/connection.py
import asyncio
from typing import Callable, Iterator

import websockets

from nova.shared.protocol import Hello, dump_message, parse_server_message


def backoff_delays(base: float = 1.0, factor: float = 2.0, max_delay: float = 15.0) -> Iterator[float]:
    delay = base
    while True:
        yield delay
        delay = min(delay * factor, max_delay)


class LatestSlot:
    def __init__(self):
        self._q: asyncio.Queue = asyncio.Queue(maxsize=1)

    def put(self, item) -> None:
        if self._q.full():
            self._q.get_nowait()
        self._q.put_nowait(item)

    async def get(self):
        return await self._q.get()


class Connection:
    def __init__(self, url: str, on_message: Callable, hello: Hello):
        self._url = url
        self._on_message = on_message
        self._hello = hello
        self._out: asyncio.Queue = asyncio.Queue()
        self._frames = LatestSlot()

    def send(self, msg) -> None:
        self._out.put_nowait(dump_message(msg))

    def send_frame(self, msg) -> None:
        self._frames.put(dump_message(msg))

    async def run(self) -> None:
        delays = backoff_delays()
        while True:
            try:
                async with websockets.connect(self._url, max_size=32 * 1024 * 1024) as ws:
                    delays = backoff_delays()  # успешный коннект — сброс backoff
                    await ws.send(dump_message(self._hello))
                    print(f"[nova] подключено к {self._url}")
                    await asyncio.gather(
                        self._pump_queue(ws), self._pump_frames(ws), self._pump_in(ws)
                    )
            except (OSError, websockets.WebSocketException) as exc:
                delay = next(delays)
                print(f"[nova] соединение потеряно ({exc!r}), повтор через {delay:.0f}с")
                await asyncio.sleep(delay)

    async def _pump_queue(self, ws) -> None:
        while True:
            await ws.send(await self._out.get())

    async def _pump_frames(self, ws) -> None:
        while True:
            await ws.send(await self._frames.get())

    async def _pump_in(self, ws) -> None:
        async for raw in ws:
            self._on_message(parse_server_message(raw))
```

- [ ] **Step 4: Прогнать тест**

Run: `uv run pytest tests/test_connection.py -v`
Expected: 3 passed

- [ ] **Step 5: Commit**

```bash
git add nova/client/connection.py tests/test_connection.py
git commit -m "feat: reconnecting ws client with latest-frame slot"
```

---

### Task 11: Воспроизведение речи (client/audio_out.py)

**Files:**
- Create: `nova/client/audio_out.py`
- Test: `tests/test_audio_out.py`

**Interfaces:**
- Consumes: `SpeakStart`, `AudioChunk`, `SpeakEnd` (Task 2).
- Produces:
  - `AudioSink` (ABC): `.play(pcm: bytes, sample_rate: int) -> None`
  - `SounddeviceSink()` — реальное воспроизведение (не тестируется автоматически)
  - `Player(sink: AudioSink)` — `.handle(msg) -> None` (собирает чанки utterance, по `SpeakEnd` отдаёт в sink); `.muted: bool` (True — utterance полностью отбрасывается).

- [ ] **Step 1: Написать падающий тест**

```python
# tests/test_audio_out.py
import base64

from nova.client.audio_out import AudioSink, Player
from nova.shared.protocol import AudioChunk, SpeakEnd, SpeakStart


class FakeSink(AudioSink):
    def __init__(self):
        self.played = []

    def play(self, pcm: bytes, sample_rate: int) -> None:
        self.played.append((pcm, sample_rate))


def utterance(uid="u1", parts=(b"aa", b"bb")):
    msgs = [SpeakStart(utterance_id=uid, text="т", reason="reply", sample_rate=16000)]
    for i, p in enumerate(parts):
        msgs.append(AudioChunk(utterance_id=uid, seq=i, pcm_b64=base64.b64encode(p).decode()))
    msgs.append(SpeakEnd(utterance_id=uid))
    return msgs


def test_chunks_assembled_in_order_and_played():
    sink = FakeSink()
    player = Player(sink)
    for msg in utterance():
        player.handle(msg)
    assert sink.played == [(b"aabb", 16000)]


def test_muted_drops_utterance():
    sink = FakeSink()
    player = Player(sink)
    player.muted = True
    for msg in utterance():
        player.handle(msg)
    assert sink.played == []


def test_chunk_for_unknown_utterance_ignored():
    sink = FakeSink()
    player = Player(sink)
    player.handle(AudioChunk(utterance_id="ghost", seq=0, pcm_b64="YWE="))
    player.handle(SpeakEnd(utterance_id="ghost"))
    assert sink.played == []
```

- [ ] **Step 2: Убедиться, что тест падает**

Run: `uv run pytest tests/test_audio_out.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'nova.client.audio_out'`

- [ ] **Step 3: Реализовать плеер**

```python
# nova/client/audio_out.py
import base64
from abc import ABC, abstractmethod

from nova.shared.protocol import AudioChunk, SpeakEnd, SpeakStart


class AudioSink(ABC):
    @abstractmethod
    def play(self, pcm: bytes, sample_rate: int) -> None: ...


class SounddeviceSink(AudioSink):
    def play(self, pcm: bytes, sample_rate: int) -> None:
        import numpy as np
        import sounddevice as sd

        sd.play(np.frombuffer(pcm, dtype=np.int16), samplerate=sample_rate, blocking=False)


class Player:
    def __init__(self, sink: AudioSink):
        self._sink = sink
        self.muted = False
        self._uid: str | None = None
        self._rate = 16000
        self._parts: list[bytes] = []

    def handle(self, msg) -> None:
        if isinstance(msg, SpeakStart):
            if self.muted:
                self._uid = None
                return
            self._uid = msg.utterance_id
            self._rate = msg.sample_rate
            self._parts = []
        elif isinstance(msg, AudioChunk):
            if msg.utterance_id == self._uid:
                self._parts.append(base64.b64decode(msg.pcm_b64))
        elif isinstance(msg, SpeakEnd):
            if msg.utterance_id == self._uid and self._parts:
                self._sink.play(b"".join(self._parts), self._rate)
            self._uid = None
            self._parts = []
```

- [ ] **Step 4: Прогнать тест**

Run: `uv run pytest tests/test_audio_out.py -v`
Expected: 3 passed

- [ ] **Step 5: Commit**

```bash
git add nova/client/audio_out.py tests/test_audio_out.py
git commit -m "feat: tts playback player with mute support"
```

---

### Task 12: Микрофон и VAD-сегментация (client/audio_in.py)

**Files:**
- Create: `nova/client/audio_in.py`
- Test: `tests/test_audio_in.py`

**Interfaces:**
- Produces:
  - `VAD` (ABC): `.is_speech(chunk: bytes) -> bool` (chunk = 512 сэмплов PCM16 @16kHz = 1024 байта, 32 мс)
  - `SileroVAD(threshold: float = 0.5)` — обёртка pysilero-vad (не тестируется автоматически)
  - `VADSegmenter(vad: VAD, chunk_ms: int = 32, silence_end_ms: int = 608, max_segment_s: float = 15.0, pre_roll_chunks: int = 6)` — `.feed(chunk: bytes) -> bytes | None`: возвращает готовый сегмент речи (с пре-роллом), когда речь закончилась или превышен максимум.
  - `Microphone(sample_rate: int = 16000, chunk_samples: int = 512)` — `.start(loop, queue)`: sounddevice InputStream, кладёт чанки bytes в asyncio.Queue через `call_soon_threadsafe` (не тестируется автоматически).

- [ ] **Step 1: Написать падающий тест**

```python
# tests/test_audio_in.py
from nova.client.audio_in import VAD, VADSegmenter

CHUNK = b"\x01\x00" * 512  # 32 мс PCM16
SILENT = b"\x00\x00" * 512


class ScriptedVAD(VAD):
    """is_speech по заранее заданному сценарию."""

    def __init__(self, flags):
        self._flags = list(flags)

    def is_speech(self, chunk: bytes) -> bool:
        return self._flags.pop(0)


def run(flags, chunks=None):
    seg = VADSegmenter(ScriptedVAD(flags), silence_end_ms=96)  # 3 чанка тишины = конец
    out = []
    for i in range(len(flags)):
        chunk = (chunks or [CHUNK] * len(flags))[i]
        r = seg.feed(chunk)
        if r is not None:
            out.append(r)
    return out


def test_silence_only_no_segments():
    assert run([False] * 10) == []


def test_speech_then_silence_emits_one_segment():
    segments = run([False, True, True, True, False, False, False])
    assert len(segments) == 1
    # 1 пре-ролл чанк + 3 речи + 3 тишины = 7 чанков... сегмент содержит всё от пре-ролла
    assert len(segments[0]) >= 4 * len(CHUNK)


def test_pre_roll_included():
    quiet = b"\x02\x00" * 512
    segments = run(
        [False, True, True, False, False, False],
        chunks=[quiet, CHUNK, CHUNK, SILENT, SILENT, SILENT],
    )
    assert len(segments) == 1
    assert segments[0].startswith(quiet)  # пре-ролл попал в сегмент


def test_max_segment_forces_cut():
    seg = VADSegmenter(ScriptedVAD([True] * 100), silence_end_ms=96, max_segment_s=0.1)
    results = [seg.feed(CHUNK) for _ in range(100)]
    emitted = [r for r in results if r is not None]
    assert len(emitted) >= 1
```

- [ ] **Step 2: Убедиться, что тест падает**

Run: `uv run pytest tests/test_audio_in.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'nova.client.audio_in'`

- [ ] **Step 3: Реализовать VAD и сегментатор**

```python
# nova/client/audio_in.py
import asyncio
from abc import ABC, abstractmethod
from collections import deque


class VAD(ABC):
    @abstractmethod
    def is_speech(self, chunk: bytes) -> bool: ...


class SileroVAD(VAD):
    def __init__(self, threshold: float = 0.5):
        from pysilero_vad import SileroVoiceActivityDetector

        self._detector = SileroVoiceActivityDetector()
        self._threshold = threshold

    def is_speech(self, chunk: bytes) -> bool:
        return self._detector(chunk) >= self._threshold


class VADSegmenter:
    def __init__(
        self,
        vad: VAD,
        chunk_ms: int = 32,
        silence_end_ms: int = 608,
        max_segment_s: float = 15.0,
        pre_roll_chunks: int = 6,
    ):
        self._vad = vad
        self._end_chunks = max(1, silence_end_ms // chunk_ms)
        self._max_chunks = max(1, int(max_segment_s * 1000 / chunk_ms))
        self._pre: deque[bytes] = deque(maxlen=pre_roll_chunks)
        self._buf: list[bytes] = []
        self._in_speech = False
        self._silence_count = 0

    def feed(self, chunk: bytes) -> bytes | None:
        speech = self._vad.is_speech(chunk)
        if not self._in_speech:
            self._pre.append(chunk)
            if speech:
                self._in_speech = True
                self._buf = list(self._pre)
                self._silence_count = 0
            return None
        self._buf.append(chunk)
        if speech:
            self._silence_count = 0
        else:
            self._silence_count += 1
            if self._silence_count >= self._end_chunks:
                return self._finish()
        if len(self._buf) >= self._max_chunks:
            return self._finish()
        return None

    def _finish(self) -> bytes:
        segment = b"".join(self._buf)
        self._buf = []
        self._in_speech = False
        self._silence_count = 0
        self._pre.clear()
        return segment


class Microphone:
    def __init__(self, sample_rate: int = 16000, chunk_samples: int = 512):
        self._sample_rate = sample_rate
        self._chunk_samples = chunk_samples
        self._stream = None

    def start(self, loop: asyncio.AbstractEventLoop, queue: asyncio.Queue) -> None:
        import sounddevice as sd

        def callback(indata, frames, time_info, status):
            loop.call_soon_threadsafe(queue.put_nowait, bytes(indata))

        self._stream = sd.RawInputStream(
            samplerate=self._sample_rate,
            blocksize=self._chunk_samples,
            channels=1,
            dtype="int16",
            callback=callback,
        )
        self._stream.start()
```

- [ ] **Step 4: Прогнать тест**

Run: `uv run pytest tests/test_audio_in.py -v`
Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
git add nova/client/audio_in.py tests/test_audio_in.py
git commit -m "feat: vad speech segmentation and microphone wrapper"
```

---

### Task 13: Захват экрана и курсор (client/capture.py)

**Files:**
- Create: `nova/client/capture.py`
- Test: `tests/test_capture.py`

**Interfaces:**
- Produces:
  - `to_gray_small(frame_bgr: np.ndarray, width: int = 160, height: int = 90) -> np.ndarray` (uint8 grayscale)
  - `encode_jpeg(frame_bgr: np.ndarray, quality: int = 85) -> bytes`
  - `cursor_pos() -> tuple[int, int]` (Windows GetCursorPos; на других ОС/ошибке — `(0, 0)`)
  - `Grabber()` — `.grab() -> np.ndarray | None` (BGR; dxcam, при недоступности — mss; None если кадр не изменился). Железо — не тестируется автоматически.

- [ ] **Step 1: Написать падающий тест**

```python
# tests/test_capture.py
import numpy as np

from nova.client.capture import cursor_pos, encode_jpeg, to_gray_small


def test_to_gray_small_shape_and_dtype():
    frame = np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8)
    small = to_gray_small(frame)
    assert small.shape == (90, 160)
    assert small.dtype == np.uint8


def test_encode_jpeg_roundtrip():
    import cv2

    frame = np.full((100, 100, 3), 128, dtype=np.uint8)
    data = encode_jpeg(frame, quality=85)
    assert data[:2] == b"\xff\xd8"  # JPEG magic
    decoded = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    assert decoded.shape == (100, 100, 3)


def test_cursor_pos_returns_ints():
    x, y = cursor_pos()
    assert isinstance(x, int) and isinstance(y, int)
```

- [ ] **Step 2: Убедиться, что тест падает**

Run: `uv run pytest tests/test_capture.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'nova.client.capture'`

- [ ] **Step 3: Реализовать захват**

```python
# nova/client/capture.py
import numpy as np


def to_gray_small(frame_bgr: np.ndarray, width: int = 160, height: int = 90) -> np.ndarray:
    import cv2

    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    return cv2.resize(gray, (width, height), interpolation=cv2.INTER_AREA)


def encode_jpeg(frame_bgr: np.ndarray, quality: int = 85) -> bytes:
    import cv2

    ok, buf = cv2.imencode(".jpg", frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise RuntimeError("jpeg encode failed")
    return buf.tobytes()


def cursor_pos() -> tuple[int, int]:
    try:
        import ctypes
        from ctypes import wintypes

        pt = wintypes.POINT()
        ctypes.windll.user32.GetCursorPos(ctypes.byref(pt))
        return int(pt.x), int(pt.y)
    except Exception:
        return (0, 0)


class Grabber:
    """dxcam (DXGI duplication) с fallback на mss."""

    def __init__(self):
        self._backend = "none"
        try:
            import dxcam

            self._cam = dxcam.create(output_color="BGR")
            if self._cam is not None:
                self._backend = "dxcam"
        except Exception:
            pass
        if self._backend == "none":
            import mss

            self._sct = mss.mss()
            self._backend = "mss"
        print(f"[nova] захват экрана: {self._backend}")

    def grab(self) -> np.ndarray | None:
        if self._backend == "dxcam":
            return self._cam.grab()  # None, если кадр не менялся
        raw = self._sct.grab(self._sct.monitors[1])
        return np.asarray(raw)[:, :, :3].copy()
```

- [ ] **Step 4: Прогнать тест**

Run: `uv run pytest tests/test_capture.py -v`
Expected: 3 passed

- [ ] **Step 5: Commit**

```bash
git add nova/client/capture.py tests/test_capture.py
git commit -m "feat: screen grabber with dxcam/mss fallback and cursor position"
```

---

### Task 14: Метрики и главный цикл клиента (client/metrics.py, client/main.py)

**Files:**
- Create: `nova/client/metrics.py`, `nova/client/main.py`
- Test: `tests/test_client_main.py`

**Interfaces:**
- Consumes: все клиентские модули (Task 8–13), протокол (Task 2).
- Produces:
  - `Metrics(path: Path)` — `.log(kind: str, **fields)` пишет jsonl-строку `{"ts": ..., "kind": ..., **fields}`.
  - `capture_loop(grabber, detector, burst, conn, cfg, iterations: int | None = None, sleep_s: float = 1 / 15)` — async: периодические кадры через `conn.send_frame`, события через `conn.send`, burst-кадры через `conn.send`.
  - `make_on_message(player: Player, metrics: Metrics, state: dict) -> Callable` — обработчик серверных сообщений: печатает текст реплик, играет звук, пишет метрику `speak_latency` = `now - state["last_event_ts"]`.
  - `main()` — вайринг всего, запуск `asyncio.run`.

- [ ] **Step 1: Написать падающий тест**

```python
# tests/test_client_main.py
import json
import time

import numpy as np

from nova.client.audio_out import AudioSink, Player
from nova.client.config import ClientConfig
from nova.client.detector import BurstCollector, FrameDetector
from nova.client.main import Metrics, capture_loop, make_on_message
from nova.shared.protocol import DetectorEvent, Frame, SpeakStart


class FakeGrabber:
    """Отдаёт чёрные кадры, затем белые (смена сцены)."""

    def __init__(self):
        self.count = 0

    def grab(self):
        self.count += 1
        value = 0 if self.count <= 3 else 255
        return np.full((90, 160, 3), value, dtype=np.uint8)


class FakeConn:
    def __init__(self):
        self.sent = []
        self.frames = []

    def send(self, msg):
        self.sent.append(msg)

    def send_frame(self, msg):
        self.frames.append(msg)


async def test_capture_loop_sends_periodic_event_and_burst():
    cfg = ClientConfig(server_url="ws://x", periodic_fps=100.0, burst_frames=2)
    conn = FakeConn()
    await capture_loop(
        grabber=FakeGrabber(),
        detector=FrameDetector(motion_threshold=12.0, scene_threshold=40.0),
        burst=BurstCollector(size=cfg.burst_frames),
        conn=conn,
        cfg=cfg,
        iterations=8,
        sleep_s=0.0,
    )
    assert any(isinstance(m, Frame) and m.kind == "periodic" for m in conn.frames)
    events = [m for m in conn.sent if isinstance(m, DetectorEvent)]
    assert any(e.event == "scene_change" for e in events)
    bursts = [m for m in conn.sent if isinstance(m, Frame) and m.kind == "burst"]
    assert len(bursts) == cfg.burst_frames
    assert all(b.burst_id == bursts[0].burst_id for b in bursts)


def test_on_message_logs_latency_and_prints(tmp_path, capsys):
    class NullSink(AudioSink):
        def play(self, pcm, sample_rate):
            pass

    metrics = Metrics(tmp_path / "metrics.jsonl")
    state = {"last_event_ts": time.time() - 0.5}
    handler = make_on_message(Player(NullSink()), metrics, state)
    handler(SpeakStart(utterance_id="u1", text="о, смена сцены", reason="proactive", sample_rate=16000))
    out = capsys.readouterr().out
    assert "о, смена сцены" in out
    rec = json.loads((tmp_path / "metrics.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert rec["kind"] == "speak_latency"
    assert rec["latency_s"] >= 0.5
```

- [ ] **Step 2: Убедиться, что тест падает**

Run: `uv run pytest tests/test_client_main.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'nova.client.main'`

- [ ] **Step 3: Реализовать метрики и главный цикл**

```python
# nova/client/metrics.py
import json
import time
from pathlib import Path


class Metrics:
    def __init__(self, path: Path):
        self._path = path
        self._path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, kind: str, **fields) -> None:
        record = {"ts": time.time(), "kind": kind, **fields}
        with self._path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
```

```python
# nova/client/main.py
import asyncio
import base64
import time
from pathlib import Path

from nova.client.audio_in import Microphone, SileroVAD, VADSegmenter
from nova.client.audio_out import Player, SounddeviceSink
from nova.client.capture import Grabber, cursor_pos, encode_jpeg, to_gray_small
from nova.client.config import ClientConfig, load_config
from nova.client.connection import Connection
from nova.client.detector import BurstCollector, FrameDetector
from nova.client.metrics import Metrics
from nova.shared.profiles import load_profile
from nova.shared.protocol import (
    AudioSegment, DetectorEvent, Frame, Hello, Hotkey, SpeakStart,
)


async def capture_loop(grabber, detector, burst, conn, cfg: ClientConfig,
                       iterations: int | None = None, sleep_s: float = 1 / 15):
    period = 1.0 / cfg.periodic_fps
    last_periodic = 0.0
    i = 0
    while iterations is None or i < iterations:
        i += 1
        frame = grabber.grab()
        if frame is None:
            await asyncio.sleep(sleep_s)
            continue
        ts = time.time()
        event = detector.process(to_gray_small(frame), ts)
        if event and not burst.active:
            conn.send(DetectorEvent(ts=ts, event=event))
            burst.start()
        if burst.active:
            done = burst.add(encode_jpeg(frame, cfg.jpeg_quality))
            if done is not None:
                for seq, jpeg in enumerate(done):
                    conn.send(Frame(
                        ts=ts, jpeg_b64=base64.b64encode(jpeg).decode(),
                        kind="burst", burst_id=burst.burst_id, seq=seq,
                    ))
        elif ts - last_periodic >= period:
            x, y = cursor_pos()
            conn.send_frame(Frame(
                ts=ts, jpeg_b64=base64.b64encode(encode_jpeg(frame, cfg.jpeg_quality)).decode(),
                cursor_x=x, cursor_y=y,
            ))
            last_periodic = ts
        await asyncio.sleep(sleep_s)


def make_on_message(player: Player, metrics: Metrics, state: dict):
    def on_message(msg) -> None:
        if isinstance(msg, SpeakStart):
            latency = time.time() - state.get("last_event_ts", time.time())
            metrics.log("speak_latency", latency_s=round(latency, 3), reason=msg.reason)
            print(f"[NOVA:{msg.reason}] {msg.text}")
        player.handle(msg)

    return on_message


async def audio_in_loop(conn, segmenter, mic_queue: asyncio.Queue, state: dict):
    while True:
        chunk = await mic_queue.get()
        segment = segmenter.feed(chunk)
        if segment is not None:
            state["last_event_ts"] = time.time()
            conn.send(AudioSegment(
                ts=time.time(), pcm_b64=base64.b64encode(segment).decode(), sample_rate=16000,
            ))


async def hotkey_loop(conn, player: Player, actions: asyncio.Queue, state: dict):
    # имена биндов из конфига → действия протокола
    action_map = {"pause": "toggle_pause"}
    while True:
        action = await actions.get()
        if action == "mute":
            player.muted = not player.muted
            print(f"[nova] mute: {player.muted}")
        else:
            if action == "comment_now":
                state["last_event_ts"] = time.time()
            conn.send(Hotkey(action=action_map.get(action, action)))


def register_hotkeys(cfg: ClientConfig, loop, actions: asyncio.Queue) -> None:
    import keyboard

    for action, combo in cfg.hotkeys.items():
        keyboard.add_hotkey(
            combo,
            lambda a=action: loop.call_soon_threadsafe(actions.put_nowait, a),
        )


async def amain() -> None:
    cfg = load_config(Path("client_config.yaml"))
    profile = load_profile(cfg.profile, Path("profiles"))
    state: dict = {}
    metrics = Metrics(Path("data/metrics.jsonl"))
    player = Player(SounddeviceSink())
    conn = Connection(
        cfg.server_url,
        on_message=make_on_message(player, metrics, state),
        hello=Hello(profile=cfg.profile, persona=cfg.persona),
    )
    detector = FrameDetector(
        motion_threshold=profile.detector.motion_threshold,
        scene_threshold=profile.detector.scene_threshold,
    )

    def sending_conn_send(msg):
        if isinstance(msg, DetectorEvent):
            state["last_event_ts"] = time.time()
        conn.send(msg)

    class ConnAdapter:
        send = staticmethod(sending_conn_send)
        send_frame = staticmethod(conn.send_frame)

    loop = asyncio.get_running_loop()
    mic_queue: asyncio.Queue = asyncio.Queue()
    Microphone().start(loop, mic_queue)
    actions: asyncio.Queue = asyncio.Queue()
    register_hotkeys(cfg, loop, actions)
    print("[nova] клиент запущен, хоткеи активны")

    await asyncio.gather(
        conn.run(),
        capture_loop(Grabber(), detector, BurstCollector(cfg.burst_frames), ConnAdapter, cfg),
        audio_in_loop(ConnAdapter, VADSegmenter(SileroVAD()), mic_queue, state),
        hotkey_loop(ConnAdapter, player, actions, state),
    )


def main() -> None:
    asyncio.run(amain())


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Прогнать тест**

Run: `uv run pytest tests/test_client_main.py -v`
Expected: 2 passed

- [ ] **Step 5: Прогнать все тесты**

Run: `uv run pytest -q`
Expected: все passed (≈33 теста)

- [ ] **Step 6: Commit**

```bash
git add nova/client/metrics.py nova/client/main.py tests/test_client_main.py
git commit -m "feat: client main loop wiring capture, audio, hotkeys and metrics"
```

---

### Task 15: README и ручной E2E smoke-тест

**Files:**
- Create: `README.md`

**Interfaces:**
- Consumes: всё выше.
- Produces: инструкция запуска + подтверждённый ручной прогон end-to-end.

- [ ] **Step 1: Написать README**

```markdown
# NOVA — персональный ИИ-компаньон

Этап 1: скелет (mock-модели, локально). Спека: `docs/specs/2026-07-03-nova-companion-design.md`.

## Запуск (Windows, два терминала)

Сервер:
    uv run python -m nova.server.main

Клиент:
    uv run python -m nova.client.main

## Что должно происходить (mock-режим)

- Клиент печатает `подключено`, захват экрана: dxcam.
- Резко смени картинку на экране (переключи окно на полноэкранное видео) —
  в консоли клиента появится `[NOVA:proactive] (мок) Заметила событие scene_change...`
  и прозвучит бип (mock-TTS).
- Скажи что-нибудь в микрофон — придёт `[NOVA:reply] (мок) Ты сказал...` + бип.
- Хоткеи: Ctrl+Alt+C — «прокомментируй сейчас», Ctrl+Alt+M — mute,
  Ctrl+Alt+P — пауза наблюдения, Ctrl+Alt+↑/↓ — оценка реплики.
- Метрики задержек: `data/metrics.jsonl`; оценки: `data/feedback.jsonl`.

## Тесты

    uv run pytest

## Конфиги

- `client_config.yaml` — адрес сервера, профиль, хоткеи.
- `profiles/*.yaml` — чувствительность детектора, болтливость, кулдаун.
- `personas/nova/system_prompt.md` — характер NOVA (этап 2+).
```

- [ ] **Step 2: Ручной smoke-прогон (чеклист)**

Запустить сервер и клиент по README и проверить:

1. Клиент подключился (лог `подключено к ws://localhost:8000/ws`).
2. Смена картинки на экране → в течение ~2 c строка `[NOVA:proactive] ...` + бип из динамиков.
3. Фраза в микрофон → `[NOVA:reply] ...` + бип.
4. `Ctrl+Alt+C` → `[NOVA:forced] ...` + бип.
5. `Ctrl+Alt+M` → бипов нет, текст печатается; повторно — бипы вернулись.
6. `Ctrl+Alt+P` → смена сцены больше не комментируется; повторно — комментируется.
7. Убить сервер (Ctrl+C) → клиент печатает реконнекты с нарастающей задержкой; запустить сервер снова → клиент переподключился сам.
8. `data/metrics.jsonl` содержит записи `speak_latency`.

Expected: все 8 пунктов подтверждены. Если пункт падает — фиксить до зелёного.

- [ ] **Step 3: Commit**

```bash
git add README.md
git commit -m "docs: readme with run instructions and e2e smoke checklist"
```

---

## Из скоупа этапа 1 (для следующих планов)

- Этап 2: реальные модели (Qwen3-VL через vLLM, faster-whisper, XTTS-v2), Docker-образ, Vast.ai up/down/автостоп, стриминговое воспроизведение по чанкам (сейчас реплика играется целиком — для мок-бипов достаточно).
- Этап 3: RAG, память сессий, web-самообучение, профили игр, голосовая регулировка болтливости.
- Этап 4: Discord-бот. Этап 5: GUI настроек. Этап 6: LoRA.
