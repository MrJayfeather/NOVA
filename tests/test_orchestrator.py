import base64
import json
from pathlib import Path

from nova.server.models.base import NO_COMMENT, VisionLLM
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


class RecordingLLM(VisionLLM):
    """Запоминает, с какой историей его вызвали."""

    def __init__(self, reply="ок", comment="вижу"):
        self.calls = []
        self._reply, self._comment = reply, comment

    async def reply_to_user(self, text, frames, history):
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
        async def reply_to_user(self, text, frames, history):
            raise RuntimeError("gpu on fire")

        async def comment_on_event(self, event, frames, history):
            raise RuntimeError("gpu on fire")

    session, sent = make_session_with(BrokenLLM())
    await session.handle(DetectorEvent(ts=1.0, event="scene_change"))  # не должно бросить
    assert sent == []


async def test_feedback_written_to_jsonl(tmp_path):
    session, sent = make_session(tmp_path)
    await session.handle(Hotkey(action="comment_now"))
    await session.handle(Hotkey(action="feedback_up"))
    lines = (tmp_path / "feedback.jsonl").read_text(encoding="utf-8").strip().splitlines()
    rec = json.loads(lines[0])
    assert rec["direction"] == "up"
    assert rec["text"]  # текст последней реплики
