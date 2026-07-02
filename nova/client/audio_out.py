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
