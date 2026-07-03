import asyncio
import base64
import json
import time
import uuid
from collections import deque
from pathlib import Path
from typing import Awaitable, Callable

from nova.server.models.base import ASRModel, NO_COMMENT, TTSModel, VisionLLM
from nova.server.proactive import ProactiveEngine
from nova.server.tts_text import normalize_for_tts
from nova.shared.protocol import (
    AudioChunk, AudioSegment, DetectorEvent, Frame, Hotkey,
    SpeakEnd, SpeakStart,
)

Send = Callable[[object], Awaitable[None]]

# Кадры прикладываются к ответу только когда вопрос реально про экран:
# с постоянными кадрами модель пересказывает экран вместо ответа на вопрос.
_SCREEN_WORDS = (
    "экран", "монитор", "видишь", "видно", "посмотр", "смотри", "покажи",
    "что это", "что тут", "что здесь", "что происходит", "окн", "вкладк",
    "приложени", "программ", "сайт", "страниц", "игр", "видео", "ролик",
)


def wants_screen(text: str) -> bool:
    t = text.lower()
    return any(w in t for w in _SCREEN_WORDS)


class Session:
    def __init__(
        self,
        send: Send,
        engine: ProactiveEngine,
        asr: ASRModel,
        llm: VisionLLM,
        tts: TTSModel,
        feedback_path: Path | None = None,
        tts_timeout_s: float = 90.0,
    ):
        self._tts_timeout_s = tts_timeout_s
        self._send = send
        self._engine = engine
        self._asr = asr
        self._llm = llm
        self._tts = tts
        self._feedback_path = feedback_path
        self._frames: deque[bytes] = deque(maxlen=8)
        # история текстовая (кадры в неё не пишутся), поэтому дешёвая:
        # 100 реплик ~ 4к токенов из 16к окна
        self._history: deque[dict] = deque(maxlen=100)
        self._last_text: str = ""

    async def handle(self, msg) -> None:
        if isinstance(msg, Frame):
            self._frames.append(base64.b64decode(msg.jpeg_b64))
        elif isinstance(msg, AudioSegment):
            try:
                text = await self._asr.transcribe(base64.b64decode(msg.pcm_b64), msg.sample_rate)
                frames = list(self._frames) if wants_screen(text) else []
                reply = await self._llm.reply_to_user(text, frames, list(self._history))
            except Exception as exc:
                print(f"[nova] ошибка модели (reply): {exc!r}")
                return
            self._history.append({"role": "user", "content": text})
            self._history.append({"role": "assistant", "content": reply})
            await self._speak(reply, reason="reply", heard=text)
        elif isinstance(msg, DetectorEvent):
            decision = self._engine.on_event(msg.event, now=time.time())
            if decision.speak:
                await self._comment(msg.event, reason="proactive")
        elif isinstance(msg, Hotkey):
            await self._handle_hotkey(msg)

    async def _handle_hotkey(self, msg: Hotkey) -> None:
        if msg.action == "comment_now":
            self._engine.on_event("comment_now", now=time.time(), forced=True)
            await self._comment("user_request", reason="forced")
        elif msg.action == "toggle_pause":
            self._engine.toggle_pause()
        elif msg.action in ("feedback_up", "feedback_down"):
            self._write_feedback("up" if msg.action == "feedback_up" else "down")

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

    def _write_feedback(self, direction: str) -> None:
        if self._feedback_path is None:
            return
        self._feedback_path.parent.mkdir(parents=True, exist_ok=True)
        record = {"ts": time.time(), "direction": direction, "text": self._last_text}
        with self._feedback_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    async def _speak(self, text: str, reason: str, heard: str = "") -> None:
        self._last_text = text
        uid = uuid.uuid4().hex[:8]
        await self._send(
            SpeakStart(utterance_id=uid, text=text, reason=reason,
                       sample_rate=self._tts.sample_rate, heard=heard)
        )
        seq = 0
        try:
            async with asyncio.timeout(self._tts_timeout_s):
                # в синтез — произносимый текст; пользователю показан исходный
                async for chunk in self._tts.synthesize(normalize_for_tts(text)):
                    await self._send(
                        AudioChunk(utterance_id=uid, seq=seq,
                                   pcm_b64=base64.b64encode(chunk).decode())
                    )
                    seq += 1
        except TimeoutError:
            print("[nova] TTS завис — реплика оборвана, сессия жива")
        await self._send(SpeakEnd(utterance_id=uid))
