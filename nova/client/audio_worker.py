"""Микрофон + VAD в отдельном процессе.

PortAudio-вход в одном процессе с остальными нативными библиотеками
клиента (onnxruntime, pynput, multiprocessing-пайпы) вызывает access
violation. Изоляция входного аудио в дочерний процесс убирает конфликт;
вывод звука (sd.play) в основном процессе работает без проблем.
"""


def run_audio_in(out_queue, sample_rate: int = 16000, chunk_samples: int = 512) -> None:
    import queue as q

    import sounddevice as sd

    from nova.client.audio_in import SileroVAD, VADSegmenter

    segmenter = VADSegmenter(SileroVAD())
    chunks: q.Queue = q.Queue()

    def callback(indata, frames, time_info, status):
        chunks.put(bytes(indata))

    stream = sd.RawInputStream(
        samplerate=sample_rate,
        blocksize=chunk_samples,
        channels=1,
        dtype="int16",
        callback=callback,
    )
    stream.start()
    while True:
        segment = segmenter.feed(chunks.get())
        if segment is not None:
            if out_queue.full():
                try:
                    out_queue.get_nowait()
                except Exception:
                    pass
            out_queue.put(segment)


class ProcessAudioSource:
    """Готовые сегменты речи из дочернего процесса.

    get() -> bytes | None (PCM16 mono 16kHz)
    """

    def __init__(self):
        import multiprocessing as mp

        self._queue = mp.Queue(maxsize=4)
        self._proc = mp.Process(target=run_audio_in, args=(self._queue,), daemon=True)
        self._proc.start()

    def get(self):
        import queue

        try:
            return self._queue.get(timeout=1.0)
        except queue.Empty:
            return None
