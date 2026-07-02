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
