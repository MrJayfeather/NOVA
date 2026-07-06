import asyncio
import base64
import json
import os
import time
import uuid
from collections import deque
from pathlib import Path
from typing import Awaitable, Callable

from nova.server.models.base import ASRModel, NO_COMMENT, TTSModel, VisionLLM
from nova.server.proactive import ProactiveEngine
from nova.server.tts_text import (
    asr_garbage, drop_leading_sounds, normalize_for_tts, strip_markers,
)
from nova.shared.protocol import (
    AudioChunk, AudioSegment, CinemaMode, Clip, DetectorEvent, Frame,
    Hotkey, SpeakEnd, SpeakStart,
)

Send = Callable[[object], Awaitable[None]]

# Кадры прикладываются к ответу только когда вопрос реально про экран:
# с постоянными кадрами модель пересказывает экран вместо ответа на вопрос.
_SCREEN_WORDS = (
    "экран", "монитор", "видишь", "видно", "посмотр", "смотри", "покажи",
    "что это", "что тут", "что здесь", "что происходит", "окн", "вкладк",
    "приложени", "программ", "сайт", "страниц", "игр", "видео", "ролик",
    # системные вопросы «про сейчас»: без кадра мозг сочиняет из головы
    # («какая дата на компе?» -> «24 мая 2024»)
    "дата", "число", "врем", "час", "тре", "раскладк", "комп", "ноут",
    "включ", "открыт", "курсор", "таймер", "счёт", "счет",
)


def wants_screen(text: str) -> bool:
    t = text.lower()
    return any(w in t for w in _SCREEN_WORDS)


# кино-режим голосом: только устойчивые формы — одиночное «смотри,»
# не триггер («смотри, какой анлак!» не должно включать кино)
_CINEMA_ON = ("смотрим фильм", "смотрим кино", "смотрим видос",
              "давай смотреть", "смотри внимательно", "следи за экраном")
_CINEMA_OFF = ("хватит смотреть", "не смотри", "можешь расслабиться",
               "можешь не смотреть")


def cinema_command(text: str) -> bool | None:
    t = text.lower()
    if any(w in t for w in _CINEMA_OFF):
        return False
    if any(w in t for w in _CINEMA_ON):
        return True
    return None


class Memory:
    """Связка памяти для сессии: хранилище, конденсер, git-синк, пульс."""

    def __init__(self, store, condenser=None, sync=None,
                 chronicle_s: float = 10.0):
        self.store = store
        self.condenser = condenser
        self.sync = sync
        self.chronicle_s = chronicle_s


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
        memory: Memory | None = None,
    ):
        self._tts_timeout_s = tts_timeout_s
        self._memory = memory
        self._last_chronicle = 0.0
        self._send = send
        self._engine = engine
        self._asr = asr
        self._llm = llm
        self._tts = tts
        self._feedback_path = feedback_path
        self._frames: deque[bytes] = deque(maxlen=8)
        self._last_clip = ""
        self._last_user_ts = 0.0
        # история текстовая (кадры в неё не пишутся), поэтому дешёвая:
        # 100 реплик ~ 4к токенов из 16к окна
        self._history: deque[dict] = deque(maxlen=100)
        self._last_text: str = ""

    def _quiet_hint(self) -> str:
        """Тишина вежливости (мягкая): в окно после реплики Джея события
        не блокируются, а уходят мозгу с наказом «только если реально
        стоит» — решение за ней (зачаток судьи 3Б)."""
        quiet = float(os.environ.get("NOVA_QUIET_AFTER_USER_S", "30"))
        if time.time() - self._last_user_ts < quiet:
            return (" (Джей только что говорил — вклинивайся ТОЛЬКО если "
                    f"событие реально того стоит, иначе ответь {NO_COMMENT})")
        return ""

    async def handle(self, msg) -> None:
        if isinstance(msg, Frame):
            self._frames.append(base64.b64decode(msg.jpeg_b64))
            await self._chronicle_pulse()
        elif isinstance(msg, AudioSegment):
            self._last_user_ts = time.time()
            t0 = time.time()
            try:
                text = await self._asr.transcribe(base64.b64decode(msg.pcm_b64), msg.sample_rate)
                if asr_garbage(text):
                    # виспер нагаллюцинировал на бормотании — пусть переспросит
                    text = "(неразборчивое бормотание)"
                cmd = cinema_command(text)
                if cmd is not None:
                    # команда взгляда перехватывается ДО мозга
                    await self._send(CinemaMode(on=cmd))
                    if self._memory:
                        self._memory.store.append_event(
                            f"кино-режим {'вкл' if cmd else 'выкл'} голосом")
                    await self._speak(
                        "Смотрю во все глаза!" if cmd else "Ладно, расслабляюсь.",
                        reason="reply", heard=text)
                    return
                t_asr = time.time()
                frames = list(self._frames) if wants_screen(text) else []
                text_llm = self._with_memories(text)
                if frames and self._last_clip:
                    # вопрос «что происходит?» отвечается и по движухе;
                    # вставка — только мозгу (дневник/история чистые)
                    text_llm = f"[последний клип: {self._last_clip}]\n{text_llm}"
                reply = await self._llm.reply_to_user(text_llm, frames, list(self._history))
                t_brain = time.time()
            except Exception as exc:
                print(f"[nova] ошибка модели (reply): {exc!r}")
                return
            # секундомер этапов: где реально теряются секунды ответа
            print(f"[nova] тайминг: asr {t_asr - t0:.1f}с | "
                  f"глаза+мозг {t_brain - t_asr:.1f}с", flush=True)
            print(f"[nova] reply: {reply!r}")
            # история — чистым текстом, иначе модель копирует свои же
            # ремарки из прошлых реплик и злоупотребляет ими
            self._history.append({"role": "user", "content": text})
            self._history.append({"role": "assistant", "content": strip_markers(reply)})
            if self._memory:
                self._memory.store.append_reply("NOVA", strip_markers(reply))
                if self._memory.sync:
                    self._memory.sync.request_push()
            await self._speak(reply, reason="reply", heard=text)
        elif isinstance(msg, Clip):
            describe = getattr(self._llm, "describe_clip", None)
            if describe is None:
                return
            from nova.server.game_hints import pick_hint

            hint = pick_hint(self._last_clip + " " + (
                self._memory.store._last_seen if self._memory else ""))
            summary = await describe(base64.b64decode(msg.mp4_b64), hint=hint)
            if not summary:
                return
            self._last_clip = summary
            decision = self._engine.on_event("clip", now=time.time())
            if decision.speak:
                await self._comment(f"клип: {summary}{self._quiet_hint()}",
                                    reason="proactive")
        elif isinstance(msg, DetectorEvent):
            decision = self._engine.on_event(msg.event, now=time.time())
            if decision.speak:
                await self._comment(f"{msg.event}{self._quiet_hint()}",
                                    reason="proactive")
        elif isinstance(msg, Hotkey):
            await self._handle_hotkey(msg)

    def _with_memories(self, text: str) -> str:
        """Дневник + вспоминание: по вопросу (ур.1) или ассоциативно (ур.1.5).
        Вставка уходит только мозгу; история и дневник хранят чистый текст."""
        mem = self._memory
        if not mem:
            return text
        if mem.condenser:
            mem.condenser.interrupt()  # реплика Джея всегда главнее сжатия
        mem.store.append_reply("Джей", text)
        from nova.server.memory.recall import (
            associate, keywords, recall, wants_recall,
        )
        today = time.strftime("%Y-%m-%d")
        rec = recall(mem.store, text, today) if wants_recall(text) else ""
        if not rec:
            ctx = keywords(text) + keywords(mem.store._last_seen)
            rec = associate(mem.store, ctx, today, cooldown_days=int(
                os.environ.get("NOVA_RECALL_COOLDOWN_D", "5")))
        return f"{rec}\n{text}" if rec else text

    async def _chronicle_pulse(self) -> None:
        """Летопись по пульсу: раз в chronicle_s глаза описывают свежий
        кадр; запись в дневник делает on_seen-хук глаз (см. main)."""
        if not self._memory or not self._frames:
            return
        if time.time() - self._last_chronicle < self._memory.chronicle_s:
            return
        describe = getattr(self._llm, "describe", None)
        if describe is None:
            return
        self._last_chronicle = time.time()
        try:
            await describe([self._frames[-1]])
        except Exception as exc:
            print(f"[nova] летопись: {exc!r}")

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
            # не больше лимита vLLM на картинки в запросе (см. runner.sh)
            limit = int(os.environ.get("NOVA_IMG_LIMIT", "6"))
            comment = await self._llm.comment_on_event(
                event, list(self._frames)[-limit:], list(self._history)
            )
        except Exception as exc:
            print(f"[nova] ошибка модели (comment): {exc!r}")
            return
        if comment.strip() == NO_COMMENT:
            return
        print(f"[nova] comment: {comment!r}")
        self._history.append({"role": "assistant", "content": strip_markers(comment)})
        if self._memory:
            self._memory.store.append_reply("NOVA", strip_markers(comment))
            if self._memory.sync:
                self._memory.sync.request_push()
        await self._speak(comment, reason=reason)

    def _write_feedback(self, direction: str) -> None:
        if self._feedback_path is None:
            return
        self._feedback_path.parent.mkdir(parents=True, exist_ok=True)
        record = {"ts": time.time(), "direction": direction, "text": self._last_text}
        with self._feedback_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    async def _speak(self, text: str, reason: str, heard: str = "") -> None:
        text = drop_leading_sounds(text)
        # на экран и в фидбек — чистый текст; голосовые ремарки только в TTS
        display = strip_markers(text)
        self._last_text = display
        uid = uuid.uuid4().hex[:8]
        await self._send(
            SpeakStart(utterance_id=uid, text=display, reason=reason,
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
