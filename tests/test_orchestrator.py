import base64
import json
from pathlib import Path

from nova.server.models.base import ASRModel, NO_COMMENT, VisionLLM
from nova.server.models.mock import MockASR, MockLLM, MockTTS
from nova.server.orchestrator import Session, wants_screen
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


class RecordingLLM(VisionLLM):
    """Запоминает, с какой историей его вызвали."""

    def __init__(self, reply="ок", comment="вижу"):
        self.calls = []
        self.reply_frames = []
        self._reply, self._comment = reply, comment

    async def reply_to_user(self, text, frames, history):
        self.calls.append(("reply", text, list(history)))
        self.reply_frames.append(list(frames))
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
        async def reply_to_user(self, text, frames, history):
            raise RuntimeError("gpu on fire")

        async def comment_on_event(self, event, frames, history):
            raise RuntimeError("gpu on fire")

    session, sent = make_session_with(BrokenLLM())
    await session.handle(DetectorEvent(ts=1.0, event="scene_change"))  # не должно бросить
    assert sent == []


async def test_hanging_tts_does_not_block_session():
    import asyncio

    from nova.server.models.base import TTSModel

    class HangingTTS(TTSModel):
        sample_rate = 16000

        async def synthesize(self, text):
            yield b"aa"
            await asyncio.sleep(999)

    sent = []

    async def send(msg):
        sent.append(msg)

    session = Session(
        send=send,
        engine=ProactiveEngine(cooldown_s=0.0, talkativeness=0.5, dedupe_window_s=0.0),
        asr=MockASR(), llm=MockLLM(persona_prompt="x"), tts=HangingTTS(),
        tts_timeout_s=0.2,
    )
    await session.handle(Hotkey(action="comment_now"))
    # несмотря на зависший TTS, реплика закрыта и сессия жива
    assert isinstance(sent[-1], SpeakEnd)


def test_wants_screen_detects_screen_questions():
    assert wants_screen("Нова, что ты сейчас видишь?")
    assert wants_screen("что это за приложение?")
    assert wants_screen("Посмотри на экран!")
    assert wants_screen("что тут происходит")
    # системные «про сейчас»: без кадра мозг сочинял дату из головы
    assert wants_screen("какая дата у меня сейчас на компе?")
    assert wants_screen("какая раскладка клавиатуры включена?")
    assert wants_screen("сколько времени?")
    assert wants_screen("глянь в трей")
    assert not wants_screen("Нова, как у тебя дела?")
    assert not wants_screen("расскажи что-нибудь интересное")


async def test_frames_attached_only_for_screen_questions():
    class FixedASR(ASRModel):
        def __init__(self, text):
            self._text = text

        async def transcribe(self, pcm, sample_rate):
            return self._text

    pcm_b64 = base64.b64encode(b"\x00\x00" * 1600).decode()
    jpeg_b64 = base64.b64encode(b"fakejpeg").decode()
    for phrase, n_frames in [("что видишь на экране?", 1), ("как дела?", 0)]:
        llm = RecordingLLM(reply="ок")
        sent = []

        async def send(msg):
            sent.append(msg)

        session = Session(
            send=send,
            engine=ProactiveEngine(cooldown_s=0.0, talkativeness=0.5, dedupe_window_s=0.0),
            asr=FixedASR(phrase), llm=llm, tts=MockTTS(),
        )
        await session.handle(Frame(ts=1.0, jpeg_b64=jpeg_b64))
        await session.handle(AudioSegment(ts=2.0, pcm_b64=pcm_b64, sample_rate=16000))
        assert len(llm.reply_frames[0]) == n_frames, phrase


async def test_feedback_written_to_jsonl(tmp_path):
    session, sent = make_session(tmp_path)
    await session.handle(Hotkey(action="comment_now"))
    await session.handle(Hotkey(action="feedback_up"))
    lines = (tmp_path / "feedback.jsonl").read_text(encoding="utf-8").strip().splitlines()
    rec = json.loads(lines[0])
    assert rec["direction"] == "up"
    assert rec["text"]  # текст последней реплики


# ---- память (этап 3А) ----

class FixedASR(ASRModel):
    def __init__(self, text):
        self._text = text

    async def transcribe(self, pcm, sample_rate):
        return self._text


def _audio_msg():
    return AudioSegment(ts=1.0, pcm_b64=base64.b64encode(b"\x00\x00" * 160).decode(),
                        sample_rate=16000)


def _frame_msg():
    return Frame(ts=1.0, jpeg_b64=base64.b64encode(b"fakejpeg").decode())


async def test_memory_writes_dialog_and_recall(tmp_path):
    import time as _t

    from nova.server.memory.store import MemoryStore
    from nova.server.orchestrator import Memory

    st = MemoryStore(tmp_path)
    # старый день с Джеффом — для вспоминания
    ts = _t.mktime((2026, 6, 20, 21, 0, 0, 0, 0, -1))
    st.append_seen("Джефф съел троих на турнире", ts=ts)
    st.set_index_line("2026-06-20", "2026-06-20 ★: Джефф | сущности: Джефф")

    llm = RecordingLLM(reply="Помню, конечно!")
    sent = []

    async def send(msg):
        sent.append(msg)

    session = Session(
        send=send,
        engine=ProactiveEngine(cooldown_s=0.0, talkativeness=0.0, dedupe_window_s=0.0),
        asr=FixedASR("помнишь как Джефф съел троих?"),
        llm=llm, tts=MockTTS(),
        memory=Memory(store=st),
    )
    await session.handle(_audio_msg())
    day = st.read_day(_t.strftime("%Y-%m-%d"))
    assert "[Джей] помнишь как Джефф съел троих?" in day
    assert "[NOVA] Помню, конечно!" in day
    # recall вшит в текст для мозга
    assert "[из дневника за 2026-06-20" in llm.calls[0][1]
    # а в историю ушёл чистый вопрос, без вставки
    assert session._history[0]["content"] == "помнишь как Джефф съел троих?"


async def test_chronicle_pulse_throttles(tmp_path):
    from nova.server.memory.store import MemoryStore
    from nova.server.orchestrator import Memory

    calls = []

    class EyesLLM(RecordingLLM):
        async def describe(self, frames):
            calls.append(1)
            return "экран"

    st = MemoryStore(tmp_path)
    sent = []

    async def send(msg):
        sent.append(msg)

    session = Session(
        send=send,
        engine=ProactiveEngine(cooldown_s=999.0, talkativeness=0.0, dedupe_window_s=0.0),
        asr=MockASR(), llm=EyesLLM(), tts=MockTTS(),
        memory=Memory(store=st, chronicle_s=10.0),
    )
    await session.handle(_frame_msg())          # первый кадр — описали
    await session.handle(_frame_msg())          # сразу второй — пульс молчит
    assert calls == [1]
    session._last_chronicle -= 11               # «прошло» 11 секунд
    await session.handle(_frame_msg())
    assert calls == [1, 1]


async def test_memory_condenser_interrupted_by_reply(tmp_path):
    from nova.server.memory.store import MemoryStore
    from nova.server.orchestrator import Memory

    interrupted = []

    class FakeCondenser:
        def interrupt(self):
            interrupted.append(1)

    st = MemoryStore(tmp_path)
    sent = []

    async def send(msg):
        sent.append(msg)

    session = Session(
        send=send,
        engine=ProactiveEngine(cooldown_s=0.0, talkativeness=0.0, dedupe_window_s=0.0),
        asr=FixedASR("привет"), llm=RecordingLLM(), tts=MockTTS(),
        memory=Memory(store=st, condenser=FakeCondenser()),
    )
    await session.handle(_audio_msg())
    assert interrupted == [1]                   # реплика прервала сжатие


# ---- со-просмотр (этап 3В) ----

async def test_cinema_voice_command_toggles():
    from nova.shared.protocol import CinemaMode

    sent = []

    async def send(msg):
        sent.append(msg)

    llm = RecordingLLM()
    session = Session(
        send=send,
        engine=ProactiveEngine(cooldown_s=0.0, talkativeness=0.0,
                               dedupe_window_s=0.0),
        asr=FixedASR("Нова, смотрим фильм"), llm=llm, tts=MockTTS(),
    )
    await session.handle(_audio_msg())
    modes = [m for m in sent if isinstance(m, CinemaMode)]
    assert modes and modes[0].on is True
    assert llm.calls == []      # мозг НЕ вызывался — команда до него
    starts = [m for m in sent if isinstance(m, SpeakStart)]
    assert starts               # подтвердила голосом


async def test_clip_message_becomes_seen_and_comment_material():
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
        send=send,
        engine=ProactiveEngine(cooldown_s=0.0, talkativeness=1.0,
                               dedupe_window_s=0.0),
        asr=MockASR(), llm=llm, tts=MockTTS(),
    )
    import time as _t

    await session.handle(Clip(ts=_t.time(),
                              mp4_b64=b64.b64encode(b"MP4").decode(),
                              dur_s=15.0, audio=True))
    assert session._last_clip.startswith("0:05")
    starts = [m for m in sent if isinstance(m, SpeakStart)]
    assert starts and starts[0].reason == "proactive"
    # протухший клип — в дневник, но не в эфир
    sent.clear()
    await session.handle(Clip(ts=_t.time() - 60,
                              mp4_b64=b64.b64encode(b"MP4").decode(),
                              dur_s=15.0, audio=True))
    assert [m for m in sent if isinstance(m, SpeakStart)] == []


def test_cinema_command_detection():
    from nova.server.orchestrator import cinema_command

    assert cinema_command("давай смотрим фильм") is True
    assert cinema_command("смотри внимательно") is True
    assert cinema_command("хватит смотреть") is False
    assert cinema_command("смотри, какой анлак!") is None
    assert cinema_command("как дела?") is None


def test_game_hint_picked_for_rivals():
    from nova.server.game_hints import pick_hint

    hint = pick_hint("турнир Marvel Rivals, Эмма Фрост на точке")
    assert "ПАТРОНЫ" in hint and "ЗАБАНЕННЫЕ" in hint
    assert pick_hint("читает доку по питону") == ""


async def test_proactive_quiet_hint_after_user_reply():
    import time as _t

    llm = RecordingLLM(comment="комментирую!")
    sent = []

    async def send(msg):
        sent.append(msg)

    session = Session(
        send=send,
        engine=ProactiveEngine(cooldown_s=0.0, talkativeness=1.0,
                               dedupe_window_s=0.0),
        asr=FixedASR("привет"), llm=llm, tts=MockTTS(),
    )
    await session.handle(_audio_msg())          # Джей только что говорил
    await session.handle(DetectorEvent(ts=1.0, event="scene_change"))
    # мягкая тишина: событие ушло мозгу С НАКАЗОМ «только если стоит»
    comment_events = [c[1] for c in llm.calls if c[0] == "comment"]
    assert any("реально того стоит" in e for e in comment_events)
    session._last_user_ts = _t.time() - 60      # «прошла» минута
    await session.handle(DetectorEvent(ts=2.0, event="scene_change"))
    assert "реально того стоит" not in llm.calls[-1][1]  # наказ снят
