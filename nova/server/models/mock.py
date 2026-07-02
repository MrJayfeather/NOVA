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
