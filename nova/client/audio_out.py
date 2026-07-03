import base64
import queue
import threading
from abc import ABC, abstractmethod

from nova.shared.protocol import AudioChunk, SpeakEnd, SpeakStart


class StreamSink(ABC):
    @abstractmethod
    def start(self, sample_rate: int) -> None: ...

    @abstractmethod
    def write(self, pcm: bytes) -> None: ...

    @abstractmethod
    def stop(self) -> None: ...


class SounddeviceStreamSink(StreamSink):
    def __init__(self):
        self._stream = None

    def start(self, sample_rate: int) -> None:
        import sounddevice as sd

        self.stop()
        self._stream = sd.RawOutputStream(samplerate=sample_rate, channels=1, dtype="int16")
        self._stream.start()

    def write(self, pcm: bytes) -> None:
        if self._stream is not None:
            self._stream.write(pcm)

    def stop(self) -> None:
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None


class Player:
    """Стриминговое воспроизведение: звук начинается с первого чанка.

    handle() не блокируется — запись в устройство идёт в рабочем потоке.
    """

    def __init__(self, sink: StreamSink):
        self._sink = sink
        self.muted = False
        self._uid: str | None = None
        self._q: queue.Queue = queue.Queue()
        threading.Thread(target=self._run, daemon=True).start()

    def handle(self, msg) -> None:
        if isinstance(msg, SpeakStart):
            if self.muted:
                self._uid = None
                return
            self._uid = msg.utterance_id
            self._q.put(("start", msg.sample_rate))
        elif isinstance(msg, AudioChunk):
            if msg.utterance_id == self._uid:
                self._q.put(("write", base64.b64decode(msg.pcm_b64)))
        elif isinstance(msg, SpeakEnd):
            if msg.utterance_id == self._uid:
                self._q.put(("stop", None))
            self._uid = None

    def drain(self) -> None:
        self._q.join()

    def _run(self) -> None:
        while True:
            kind, payload = self._q.get()
            try:
                if kind == "start":
                    self._sink.start(payload)
                elif kind == "write":
                    self._sink.write(payload)
                elif kind == "stop":
                    self._sink.stop()
            except Exception as exc:
                print(f"[nova] ошибка воспроизведения: {exc!r}")
            finally:
                self._q.task_done()
