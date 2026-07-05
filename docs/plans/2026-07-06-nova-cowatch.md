# План 3В: Со-просмотр (глаза-видео)

> Спека: docs/specs/2026-07-06-nova-cowatch-design.md. Ветка cowatch3v
> от master. Каждая задача — тест-цикл и коммит.

**Цель:** движуху NOVA видит клипами (15с, звук системы, таймштампы),
статику — кадрами; кино-режим хоткеем-тогглом и голосом («смотрим фильм»).

**Архитектура:** клиент — MotionGate (STILL/MOTION по событиям детектора)
и ClipRecorder (буфер кадров -> ffmpeg mp4 + WASAPI-звук, троттлинг
выгрузки); протокол — Clip (клиент->сервер) и CinemaMode (сервер->клиент);
сервер — GeminiEyes.describe_clip, оркестратор пишет сводки в [видела],
голосовые команды взгляда, клип-событие в проактивный цикл.

**Стек:** ffmpeg (CLI, есть у Джея), pyaudiowpatch (WASAPI loopback,
клиентская зависимость, мягкий фолбэк), pydantic-протокол как есть.

## Глобальные ограничения

- Без упоминаний ассистентов в коде/коммитах; ветка cowatch3v; русские
  комментарии в стиле проекта.
- `uv run pytest -q` до старта: 165 passed, 2 skipped; после каждой
  задачи — прирост без красных.
- Крутилки (все env, дефолты): NOVA_COWATCH=1, NOVA_CLIP_S=15,
  NOVA_CLIP_FPS=8, NOVA_CLIP_AUDIO=1, NOVA_CLIP_KBPS=1500,
  NOVA_MOTION_ON=3 (событий за 30с), NOVA_MOTION_OFF=60 (тишины, с).
- Одиночное «смотри,» — НЕ голосовой триггер кино.
- Троттлинг: клип уходит кусками <= NOVA_CLIP_KBPS, бэклога нет
  (очередь «только свежий»).

## Карта файлов

| Файл | Роль |
|---|---|
| nova/shared/protocol.py | + Clip (client->server), + CinemaMode (server->client), + hotkey action "cinema" |
| nova/client/motion.py (новый) | MotionGate: STILL/MOTION + кино-форс |
| nova/client/clip.py (новый) | ClipRecorder: буфер кадров, ffmpeg-энкод, WASAPI-звук, троттл-отправка |
| nova/client/main.py | интеграция: gate в capture_loop, recorder-цикл, хоткей cinema, CinemaMode в on_message |
| nova/client/config.py | hotkey cinema: ctrl+alt+v |
| nova/server/models/gemini_vision.py | describe_clip(mp4, hint) |
| nova/server/orchestrator.py | Clip-обработка, last_clip, голосовые команды взгляда, клип-событие |
| pyproject.toml | pyaudiowpatch в клиентские зависимости |
| tests/test_protocol.py, test_motion.py, test_clip.py, test_client_main.py, test_gemini_vision.py, test_orchestrator.py | тесты |

---

### Задача 1: протокол — Clip, CinemaMode, hotkey cinema

**Файлы:** изменить `nova/shared/protocol.py`, `tests/test_protocol.py`.

**Производит:** `Clip(type="clip", ts: float, mp4_b64: str, dur_s: float,
audio: bool)` в ClientMessage; `CinemaMode(type="cinema_mode", on: bool)`
в ServerMessage; Hotkey.action += "cinema".

- [ ] **Шаг 1.1: тесты (падают)** — в `tests/test_protocol.py` добавить:

```python
def test_clip_roundtrip():
    from nova.shared.protocol import Clip, dump_message, parse_client_message

    msg = Clip(ts=1.0, mp4_b64="QUJD", dur_s=15.0, audio=True)
    back = parse_client_message(dump_message(msg))
    assert isinstance(back, Clip)
    assert back.dur_s == 15.0 and back.audio is True


def test_cinema_mode_roundtrip():
    from nova.shared.protocol import (
        CinemaMode, dump_message, parse_server_message,
    )

    back = parse_server_message(dump_message(CinemaMode(on=True)))
    assert isinstance(back, CinemaMode) and back.on is True


def test_hotkey_cinema_allowed():
    from nova.shared.protocol import Hotkey

    assert Hotkey(action="cinema").action == "cinema"
```

- [ ] **Шаг 1.2:** `uv run pytest tests/test_protocol.py -q` → FAIL.

- [ ] **Шаг 1.3: реализация** — в protocol.py:

```python
class Clip(BaseModel):
    """Видео-взгляд: клип экрана при движухе (кино-режим/экшн)."""
    type: Literal["clip"] = "clip"
    ts: float
    mp4_b64: str
    dur_s: float
    audio: bool = False


class CinemaMode(BaseModel):
    """Сервер -> клиент: включить/выключить принудительный кино-режим
    (голосовая команда «смотрим фильм» ловится на сервере после ASR)."""
    type: Literal["cinema_mode"] = "cinema_mode"
    on: bool
```

Hotkey.action: `Literal["comment_now", "toggle_pause", "feedback_up",
"feedback_down", "cinema"]`. Clip — в Union ClientMessage, CinemaMode —
в Union ServerMessage.

- [ ] **Шаг 1.4:** тест → passed. **Шаг 1.5:**

```bash
git add nova/shared/protocol.py tests/test_protocol.py
git commit -m "feat: clip and cinema-mode protocol messages"
```

---

### Задача 2: MotionGate — переключатель STILL/MOTION

**Файлы:** создать `nova/client/motion.py`, `tests/test_motion.py`.

**Производит:** `MotionGate(on_events: int = 3, on_window_s: float = 30.0,
off_silence_s: float = 60.0)`: `note_event(ts)`, `is_motion(ts) -> bool`,
`set_cinema(on: bool)`, `cinema: bool` (проперти). Кино-форс перекрывает
автомат; автомат: >=on_events событий за on_window_s -> MOTION до тех
пор, пока не пройдёт off_silence_s без событий.

- [ ] **Шаг 2.1: тесты (падают)** — `tests/test_motion.py`:

```python
from nova.client.motion import MotionGate


def test_still_by_default():
    g = MotionGate()
    assert not g.is_motion(ts=100.0)


def test_motion_after_burst_of_events():
    g = MotionGate(on_events=3, on_window_s=30.0, off_silence_s=60.0)
    for t in (100.0, 105.0, 110.0):
        g.note_event(t)
    assert g.is_motion(ts=111.0)


def test_sparse_events_stay_still():
    g = MotionGate(on_events=3, on_window_s=30.0)
    for t in (100.0, 140.0, 180.0):   # реже окна
        g.note_event(t)
    assert not g.is_motion(ts=181.0)


def test_motion_decays_after_silence():
    g = MotionGate(on_events=3, on_window_s=30.0, off_silence_s=60.0)
    for t in (100.0, 101.0, 102.0):
        g.note_event(t)
    assert g.is_motion(ts=110.0)
    assert g.is_motion(ts=161.0)          # 59с тишины — ещё смотрим
    assert not g.is_motion(ts=163.0)      # 61с — расслабилась


def test_cinema_forces_motion_regardless():
    g = MotionGate()
    g.set_cinema(True)
    assert g.is_motion(ts=100.0)
    assert g.cinema
    g.set_cinema(False)
    assert not g.is_motion(ts=100.0)
```

- [ ] **Шаг 2.2:** → FAIL. **Шаг 2.3: реализация** — `nova/client/motion.py`:

```python
class MotionGate:
    """Автомат взгляда: статика -> кадры (STILL), движуха -> клипы
    (MOTION). Кино-режим (хоткей/голос) — принудительный MOTION."""

    def __init__(self, on_events: int = 3, on_window_s: float = 30.0,
                 off_silence_s: float = 60.0):
        self._on_events = on_events
        self._on_window_s = on_window_s
        self._off_silence_s = off_silence_s
        self._events: list[float] = []
        self._motion_since: float | None = None
        self._last_event = 0.0
        self.cinema = False

    def set_cinema(self, on: bool) -> None:
        self.cinema = on

    def note_event(self, ts: float) -> None:
        self._last_event = ts
        self._events = [t for t in self._events
                        if ts - t <= self._on_window_s]
        self._events.append(ts)
        if len(self._events) >= self._on_events:
            self._motion_since = ts

    def is_motion(self, ts: float) -> bool:
        if self.cinema:
            return True
        if self._motion_since is None:
            return False
        if ts - self._last_event > self._off_silence_s:
            self._motion_since = None
            self._events.clear()
            return False
        return True
```

- [ ] **Шаг 2.4:** → passed. **Шаг 2.5:**

```bash
git add nova/client/motion.py tests/test_motion.py
git commit -m "feat: motion gate - still/motion switch with cinema force"
```

---

### Задача 3: ClipRecorder — кадры -> mp4 (+звук), троттл-отправка

**Файлы:** создать `nova/client/clip.py`, `tests/test_clip.py`;
изменить `pyproject.toml` (pyaudiowpatch в зависимости клиента).

**Производит:**
`encode_clip(frames: list[np.ndarray], fps: int, wav_path: str | None,
out_path: str) -> bool` (ffmpeg; кадры BGR; при wav_path муксит звук);
`class LoopbackRecorder` — `start()`, `stop() -> str | None` (путь wav
или None при недоступном WASAPI; все ошибки — print + None);
`class ClipSender(conn, kbps: int)` — `send(clip_msg)`: base64-строка
режется на куски и уходит через conn.send_frame-подобный слот
«только свежий» — НО клип неделим, поэтому: собственный asyncio-цикл,
который держит ТОЛЬКО последний несданный клип и отправляет его
conn.send(...) целиком, зато по расписанию не чаще чем длина_байт/kbps
секунд на клип (сон до конца окна = естественный троттлинг);
`class ClipPipeline(grabber_state, gate, conn, cfg)` — `tick(ts, frame)`:
копит кадры при MOTION с шагом 1/fps, каждые clip_s секунд собирает mp4
(в temp), шлёт через ClipSender, чистит буфер; при STILL буфер пуст.

- [ ] **Шаг 3.1: тесты (падают)** — `tests/test_clip.py`:

```python
import shutil

import numpy as np
import pytest

from nova.client.clip import encode_clip

needs_ffmpeg = pytest.mark.skipif(shutil.which("ffmpeg") is None,
                                  reason="нет ffmpeg")


@needs_ffmpeg
def test_encode_clip_makes_playable_mp4(tmp_path):
    frames = [np.full((90, 160, 3), i * 16, dtype=np.uint8)
              for i in range(16)]
    out = tmp_path / "clip.mp4"
    assert encode_clip(frames, fps=8, wav_path=None, out_path=str(out))
    data = out.read_bytes()
    assert len(data) > 500
    assert b"ftyp" in data[:64]           # валидная mp4-шапка


@needs_ffmpeg
def test_encode_clip_with_audio(tmp_path):
    import wave

    wav = tmp_path / "a.wav"
    with wave.open(str(wav), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes(b"\x00\x01" * 16000)
    frames = [np.zeros((90, 160, 3), dtype=np.uint8) for _ in range(8)]
    out = tmp_path / "clip.mp4"
    assert encode_clip(frames, fps=8, wav_path=str(wav), out_path=str(out))
    assert out.stat().st_size > 500


def test_encode_clip_no_frames_returns_false(tmp_path):
    assert not encode_clip([], fps=8, wav_path=None,
                           out_path=str(tmp_path / "x.mp4"))


async def test_clip_sender_keeps_only_latest():
    from nova.client.clip import ClipSender

    sent = []

    class FakeConn:
        def send(self, msg):
            sent.append(msg)

    s = ClipSender(FakeConn(), kbps=10 ** 9)   # без троттла в тесте
    s.offer("клип1")
    s.offer("клип2")                            # первый ещё не ушёл — вытеснен
    await s.pump_once()
    assert sent == ["клип2"]
    await s.pump_once()                         # пусто — ничего не шлёт
    assert sent == ["клип2"]
```

- [ ] **Шаг 3.2:** → FAIL. **Шаг 3.3: реализация** — `nova/client/clip.py`:

```python
import asyncio
import subprocess
import tempfile
import time
from pathlib import Path

import numpy as np


def encode_clip(frames: list, fps: int, wav_path: str | None,
                out_path: str) -> bool:
    """Кадры BGR -> h264 mp4 (720p максимум); wav_path муксится в звук.
    ffmpeg обязателен; любая ошибка -> False и печать."""
    if not frames:
        return False
    h, w = frames[0].shape[:2]
    # даунскейл к 720p по высоте, чётные размеры для h264
    scale = min(1.0, 720 / h)
    out_w, out_h = int(w * scale) // 2 * 2, int(h * scale) // 2 * 2
    cmd = ["ffmpeg", "-y", "-v", "error",
           "-f", "rawvideo", "-pix_fmt", "bgr24",
           "-s", f"{w}x{h}", "-r", str(fps), "-i", "-"]
    if wav_path:
        cmd += ["-i", wav_path, "-c:a", "aac", "-shortest"]
    cmd += ["-vf", f"scale={out_w}:{out_h}",
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "28",
            "-movflags", "+faststart", out_path]
    try:
        p = subprocess.Popen(cmd, stdin=subprocess.PIPE,
                             stderr=subprocess.PIPE)
        for f in frames:
            p.stdin.write(np.ascontiguousarray(f).tobytes())
        p.stdin.close()
        p.wait(timeout=60)
        return p.returncode == 0 and Path(out_path).exists()
    except Exception as exc:
        print(f"[nova] клип: ffmpeg не собрал ({exc!r})")
        return False


class LoopbackRecorder:
    """Системный звук (WASAPI loopback) для кино-режима. Нет
    pyaudiowpatch/устройства — работаем без звука, не падаем."""

    def __init__(self):
        self._stream = None
        self._frames: list[bytes] = []
        self._rate = 48000

    def start(self) -> None:
        try:
            import pyaudiowpatch as pa

            self._pa = pa.PyAudio()
            wasapi = self._pa.get_host_api_info_by_type(pa.paWASAPI)
            out = self._pa.get_device_info_by_index(
                wasapi["defaultOutputDevice"])
            # loopback-двойник дефолтного вывода
            for i in range(self._pa.get_device_count()):
                dev = self._pa.get_device_info_by_index(i)
                if dev.get("isLoopbackDevice") and \
                        out["name"] in dev["name"]:
                    self._rate = int(dev["defaultSampleRate"])
                    self._frames = []
                    self._stream = self._pa.open(
                        format=pa.paInt16, channels=1, rate=self._rate,
                        input=True, input_device_index=i,
                        frames_per_buffer=4096,
                        stream_callback=self._cb)
                    return
            print("[nova] клип: loopback-устройство не найдено — без звука")
        except Exception as exc:
            print(f"[nova] клип: звук недоступен ({exc!r}) — без звука")
            self._stream = None

    def _cb(self, in_data, *_):
        self._frames.append(in_data)
        import pyaudiowpatch as pa

        return (None, pa.paContinue)

    def stop(self) -> str | None:
        if self._stream is None:
            return None
        import wave

        try:
            self._stream.stop_stream()
            self._stream.close()
        except Exception:
            pass
        self._stream = None
        if not self._frames:
            return None
        path = tempfile.mktemp(suffix=".wav")
        with wave.open(path, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(self._rate)
            w.writeframes(b"".join(self._frames))
        return path


class ClipSender:
    """Очередь «только свежий» + троттлинг: клип неделим, но следующий
    не начнёт уходить раньше, чем длина_текущего/kbps секунд."""

    def __init__(self, conn, kbps: int = 1500):
        self._conn = conn
        self._kbps = max(1, kbps)
        self._latest = None

    def offer(self, msg) -> None:
        self._latest = msg   # старый несданный — вытесняется

    async def pump_once(self) -> None:
        msg = self._latest
        if msg is None:
            return
        self._latest = None
        self._conn.send(msg)
        size_kbit = len(getattr(msg, "mp4_b64", "") or str(msg)) * 8 / 1000
        await asyncio.sleep(size_kbit / self._kbps)

    async def pump_loop(self) -> None:
        while True:
            await self.pump_once()
            await asyncio.sleep(0.5)
```

(ClipPipeline — в задаче 4 вместе с интеграцией: он тонкий и живёт
рядом с capture_loop.) pyproject.toml: в зависимости добавить
`"pyaudiowpatch>=0.2; sys_platform == 'win32'"`.

- [ ] **Шаг 3.4:** → passed (ffmpeg-тесты живые). **Шаг 3.5:**

```bash
git add nova/client/clip.py tests/test_clip.py pyproject.toml
git commit -m "feat: clip recorder - ffmpeg encode, wasapi sound, throttle"
```

---

### Задача 4: интеграция клиента

**Файлы:** изменить `nova/client/main.py`, `nova/client/config.py`,
`tests/test_client_main.py`.

**Потребляет:** MotionGate, encode_clip, LoopbackRecorder, ClipSender,
Clip, CinemaMode.

Поведение:
- config: DEFAULT_HOTKEYS += {"cinema": "ctrl+alt+v"}; поля
  cowatch: bool = True, clip_s: float = 15.0, clip_fps: int = 8,
  clip_audio: bool = True, clip_kbps: int = 1500, motion_on: int = 3,
  motion_off: float = 60.0.
- capture_loop: gate.note_event(ts) при каждом событии детектора;
  при gate.is_motion(ts) кадры уходят в клип-буфер (state["clip_frames"],
  с прореживанием до clip_fps), периодика кадров-пульса при MOTION
  отключается (клипы её заменяют).
- клип-цикл (новая корутина clip_loop): каждые clip_s секунд, если
  буфер непуст — encode_clip во временный файл (+wav из LoopbackRecorder
  при cinema и clip_audio), Clip(mp4_b64=..., audio=...) -> ClipSender.
  offer; при STILL — буфер чистится, рекордер звука остановлен.
- hotkey cinema: gate.set_cinema(not gate.cinema) + печать
  «[nova] кино-режим: ВКЛ/ВЫКЛ» + при ВКЛ LoopbackRecorder.start().
- on_message: CinemaMode(on) — то же самое, что хоткей (печать
  «(голосом)»).
- РУБИЛЬНИК: cfg.cowatch=False (env NOVA_COWATCH=0 у start_all или
  поле client_config.yaml) -> gate=None, clip_loop и pump_loop не
  запускаются, capture_loop работает ровно как в 3А.

- [ ] **Шаг 4.1: тесты (падают)** — в test_client_main.py:

```python
async def test_capture_feeds_clip_buffer_in_motion():
    from nova.client.main import capture_loop
    from nova.client.motion import MotionGate

    cfg = ClientConfig(server_url="ws://x", periodic_fps=100.0,
                       burst_frames=2, clip_fps=100)
    gate = MotionGate(on_events=1, on_window_s=30.0)
    state = {}
    await capture_loop(
        source=FakeSource(),
        detector=FrameDetector(motion_threshold=12.0, scene_threshold=40.0),
        burst=BurstCollector(size=2),
        conn=FakeConn(), cfg=cfg, iterations=8, state=state, gate=gate,
    )
    # сцена сменилась -> событие -> MOTION -> кадры пошли в клип-буфер
    assert state.get("clip_frames")


def test_cinema_hotkey_toggles(capsys):
    from nova.client.main import apply_cinema
    from nova.client.motion import MotionGate

    gate = MotionGate()
    rec = type("R", (), {"started": False,
                         "start": lambda s: setattr(s, "started", True),
                         "stop": lambda s: None})()
    apply_cinema(gate, rec, on=True, via="хоткей", audio=True)
    assert gate.cinema and rec.started
    out = capsys.readouterr().out
    assert "кино-режим: ВКЛ" in out
```

- [ ] **Шаг 4.2:** → FAIL. **Шаг 4.3: реализация** — main.py: функция

```python
def apply_cinema(gate, recorder, on: bool, via: str, audio: bool) -> None:
    gate.set_cinema(on)
    print(f"[nova] кино-режим: {'ВКЛ' if on else 'ВЫКЛ'} ({via})")
    if on and audio:
        recorder.start()
    elif not on:
        recorder.stop()
```

capture_loop(+gate=None): при event -> gate.note_event(ts); при
gate.is_motion(ts): прореживание по clip_fps в state["clip_frames"]
(list на append), periodic-отправка пропускается. clip_loop-корутина
в amain + ClipSender.pump_loop-таск + обработка CinemaMode в
make_on_message (вызов apply_cinema(via="голосом")). Хоткей cinema в
hotkey_loop -> apply_cinema(via="хоткей"). Полные диффы — по образцу
уже существующих циклов main.py (fresh-frame, cooldown), исполнитель —
эта же сессия, файл в контексте.

- [ ] **Шаг 4.4:** `uv run pytest -q` → зелёные. **Шаг 4.5:**

```bash
git add nova/client/main.py nova/client/config.py tests/test_client_main.py
git commit -m "feat: client cowatch - motion clips, cinema toggle, voice mode"
```

---

### Задача 5: сервер — describe_clip и оркестратор

**Файлы:** изменить `nova/server/models/gemini_vision.py`,
`nova/server/orchestrator.py`; тесты test_gemini_vision.py,
test_orchestrator.py.

**Производит:**
`GeminiEyes.describe_clip(mp4: bytes, hint: str = "") -> str` —
inline_data video/mp4 + CLIP_PROMPT («разбери с таймштампами, что
происходило и что говорили; по-русски, фактами, 3-6 строк; {hint}»);
результат уходит и в on_seen (дневник).
Оркестратор: обработка Clip в handle() — describe_clip -> self._last_clip
(строка) -> DetectorEvent-подобный путь: engine.on_event("clip") и при
decision.speak -> _comment("клип: " + сводка, reason="proactive");
голосовые команды взгляда в AudioSegment-ветке ДО мозга:
`cinema_command(text) -> bool | None` (None — не команда; True/False —
вкл/выкл) — триггеры из спеки; при команде: send(CinemaMode(on)) и
короткий ответ голосом без мозга («Смотрю!» / «Ладно, расслабляюсь»)
через _speak; вопрос про экран: last_clip добавляется в текст мозгу
(«[последний клип: ...]»).

- [ ] **Шаг 5.1: тесты (падают).** test_gemini_vision.py:

```python
async def test_describe_clip_sends_video_and_logs_seen():
    eyes = make_eyes(FakeInner())
    parts_seen = {}

    async def fake_call(frames, prompt, video=None):
        parts_seen["video"] = video
        parts_seen["prompt"] = prompt
        return "0:03 Джефф съел троих"

    eyes._call_gemini = fake_call
    seen = []
    eyes.on_seen = seen.append
    out = await eyes.describe_clip(b"MP4DATA", hint="матч Rivals")
    assert "Джефф" in out
    assert parts_seen["video"] == b"MP4DATA"
    assert "Rivals" in parts_seen["prompt"]
    assert seen == ["0:03 Джефф съел троих"]
```

(подпись _call_gemini расширяется опциональным video: bytes | None —
существующие вызовы не меняются.)

test_orchestrator.py:

```python
async def test_cinema_voice_command_toggles(tmp_path):
    from nova.shared.protocol import CinemaMode

    sent = []

    async def send(msg):
        sent.append(msg)

    session = Session(
        send=send, engine=ProactiveEngine(cooldown_s=0.0, talkativeness=0.0,
                                          dedupe_window_s=0.0),
        asr=FixedASR("Нова, смотрим фильм"), llm=RecordingLLM(),
        tts=MockTTS(),
    )
    llm = session._llm
    await session.handle(_audio_msg())
    modes = [m for m in sent if isinstance(m, CinemaMode)]
    assert modes and modes[0].on is True
    assert llm.calls == []      # мозг НЕ вызывался — команда до него
    starts = [m for m in sent if isinstance(m, SpeakStart)]
    assert starts               # подтвердила голосом


async def test_clip_message_becomes_seen_and_comment_material(tmp_path):
    import base64 as b64

    from nova.shared.protocol import Clip

    class ClipEyes(RecordingLLM):
        async def describe_clip(self, mp4, hint=""):
            return "0:05 замес, Джефф съел троих"

    llm = ClipEyes(comment="Ого, вот это замес!")
    sent = []

    async def send(msg):
        sent.append(msg)

    session = Session(
        send=send, engine=ProactiveEngine(cooldown_s=0.0, talkativeness=1.0,
                                          dedupe_window_s=0.0),
        asr=MockASR(), llm=llm, tts=MockTTS(),
    )
    await session.handle(Clip(ts=1.0, mp4_b64=b64.b64encode(b"MP4").decode(),
                              dur_s=15.0, audio=True))
    assert session._last_clip.startswith("0:05")
    starts = [m for m in sent if isinstance(m, SpeakStart)]
    assert starts and starts[0].reason == "proactive"


def test_cinema_command_detection():
    from nova.server.orchestrator import cinema_command

    assert cinema_command("давай смотрим фильм") is True
    assert cinema_command("смотри внимательно") is True
    assert cinema_command("хватит смотреть") is False
    assert cinema_command("смотри, какой анлак!") is None
    assert cinema_command("как дела?") is None
```

- [ ] **Шаг 5.2:** → FAIL. **Шаг 5.3: реализация.**
gemini_vision.py: _call_gemini(frames, prompt, video: bytes | None = None)
— video добавляет part {"inline_data": {"mime_type": "video/mp4",
"data": b64}}; describe_clip:

```python
CLIP_PROMPT = (
    "Это видеоклип с экрана пользователя (движуха/кино-режим). Разбери "
    "по таймштампам: кто что сделал, что произошло, ЧТО СКАЗАЛИ (если "
    "есть звук — реплики важны), счёт/итоги. По-русски, фактами, 3-6 "
    "строк. {hint}"
)

    async def describe_clip(self, mp4: bytes, hint: str = "") -> str:
        try:
            out = await self._call_gemini(
                [], CLIP_PROMPT.format(hint=hint), video=mp4)
        except Exception as exc:
            print(f"[nova] глаза-видео недоступны: {exc!r}")
            return ""
        if self.on_seen and out:
            self.on_seen(out)
        return out
```

orchestrator.py: `_CINEMA_ON = ("смотрим фильм", "смотрим кино",
"смотрим видос", "давай смотреть", "смотри внимательно",
"следи за экраном")`, `_CINEMA_OFF = ("хватит смотреть", "не смотри",
"можешь расслабиться", "можешь не смотреть")`;

```python
def cinema_command(text: str) -> bool | None:
    t = text.lower()
    if any(w in t for w in _CINEMA_OFF):
        return False
    if any(w in t for w in _CINEMA_ON):
        return True
    return None
```

Session: поле `self._last_clip = ""`; в AudioSegment-ветке сразу после
asr_garbage-проверки:

```python
            cmd = cinema_command(text)
            if cmd is not None:
                await self._send(CinemaMode(on=cmd))
                if self._memory:
                    self._memory.store.append_event(
                        f"кино-режим {'вкл' if cmd else 'выкл'} голосом")
                await self._speak("Смотрю во все глаза!" if cmd
                                  else "Ладно, расслабляюсь.",
                                  reason="reply", heard=text)
                return
```

вопрос про экран: в wants_screen-ветке text_llm дополняется
`f"[последний клип: {self._last_clip}]\n"` при непустом _last_clip;
обработка Clip в handle():

```python
        elif isinstance(msg, Clip):
            describe = getattr(self._llm, "describe_clip", None)
            if describe is None:
                return
            summary = await describe(base64.b64decode(msg.mp4_b64))
            if not summary:
                return
            self._last_clip = summary
            decision = self._engine.on_event("clip", now=time.time())
            if decision.speak:
                await self._comment(f"клип: {summary}", reason="proactive")
```

(engine.on_event("clip") — ProactiveEngine принимает произвольные имена
событий? проверить сигнатуру; если Literal — расширить.)
CinemaMode импорт в orchestrator; протокольный Clip — тоже.

- [ ] **Шаг 5.4:** `uv run pytest -q` → зелёные. **Шаг 5.5:**

```bash
git add nova/server/models/gemini_vision.py nova/server/orchestrator.py \
  tests/test_gemini_vision.py tests/test_orchestrator.py
git commit -m "feat: server cowatch - clip describe, voice cinema, comments"
```

---

### Задача 6: живая приёмка (с Джеем)

- [ ] merge cowatch3v -> master, push; на боксе git pull + рестарт
  сервера (kill/start раздельно!); на ноуте uv sync (pyaudiowpatch)
  + перезапуск клиента.
- [ ] Приёмка по спеке: (1) замес в Rivals -> коммент с ходом событий;
  (2) кино-режим: фильм со звуком -> пересказ реплик героев;
  (2а) «смотрим фильм»/«хватит смотреть» голосом; «смотри, какой
  анлак!» — не включает; ctrl+alt+v — тоггл в консоли, обе раскладки
  (пробник); (3) статика — расход как в 3А; (4) пинг в катке живой;
  (5) назавтра «помнишь замес?» — вспоминает из дневника;
  (6) NOVA_COWATCH=0 — поведение 3А.
- [ ] STATUS.md: «ЭТАП 3В СО-ПРОСМОТР: В ПРОДЕ» с замерами.
