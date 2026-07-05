import asyncio

from nova.server.models.base import ASRModel


class WhisperASR(ASRModel):
    def __init__(self, model_name: str = "large-v3-turbo", device: str = "cuda"):
        from faster_whisper import WhisperModel

        compute = "int8_float16" if device == "cuda" else "int8"
        self._model = WhisperModel(model_name, device=device, compute_type=compute)

    def _transcribe_sync(self, pcm: bytes, sample_rate: int) -> str:
        import numpy as np

        audio = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
        segments, _ = self._model.transcribe(audio, language="ru", beam_size=5)
        return " ".join(s.text.strip() for s in segments).strip()

    async def transcribe(self, pcm: bytes, sample_rate: int) -> str:
        return await asyncio.to_thread(self._transcribe_sync, pcm, sample_rate)

    def _words_sync(self, pcm: bytes, sample_rate: int) -> list[tuple[str, float]]:
        import numpy as np

        audio = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
        # faster-whisper считает вход 16кГц: кратную частоту децимируем,
        # некратную отдаём как есть и переводим времена в реальную шкалу
        if sample_rate % 16000 == 0 and sample_rate > 16000:
            audio = audio[:: sample_rate // 16000]
            scale = 1.0
        else:
            scale = 16000.0 / sample_rate
        segments, _ = self._model.transcribe(
            audio, language="ru", word_timestamps=True)
        return [(w.word, w.start * scale)
                for s in segments for w in (s.words or [])]

    async def word_timestamps(
        self, pcm: bytes, sample_rate: int
    ) -> list[tuple[str, float]]:
        return await asyncio.to_thread(self._words_sync, pcm, sample_rate)
