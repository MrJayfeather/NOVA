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
