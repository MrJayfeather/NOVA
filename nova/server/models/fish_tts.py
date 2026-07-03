import wave
from io import BytesIO
from pathlib import Path
from typing import AsyncIterator

import httpx
import ormsgpack

from nova.server.models.base import TTSModel
from nova.server.models.xtts_tts import split_for_tts


def build_tts_request(text: str, ref_audio: bytes, ref_text: str) -> dict:
    return {
        "text": text,
        "references": [{"audio": ref_audio, "text": ref_text}],
        "format": "wav",
        "streaming": False,
    }


def wav_to_pcm(wav_bytes: bytes) -> tuple[bytes, int]:
    with wave.open(BytesIO(wav_bytes)) as w:
        rate = w.getframerate()
        pcm = w.readframes(w.getnframes())
        if w.getnchannels() == 2:
            import numpy as np

            arr = np.frombuffer(pcm, dtype=np.int16).reshape(-1, 2)
            pcm = arr.mean(axis=1).astype(np.int16).tobytes()
    return pcm, rate


class FishTTS(TTSModel):
    """Голос через fish-speech api_server (OpenAudio S1-mini)."""

    sample_rate = 44100  # DAC-кодек S1; сверяется с реальным ответом

    def __init__(self, url: str, reference_wav: Path, reference_text: str,
                 timeout: float = 120.0):
        self._url = url
        self._ref_audio = Path(reference_wav).read_bytes()
        self._ref_text = reference_text
        self._client = httpx.AsyncClient(timeout=timeout)

    async def _tts_call(self, sentence: str) -> bytes:
        req = build_tts_request(sentence, self._ref_audio, self._ref_text)
        r = await self._client.post(
            self._url,
            content=ormsgpack.packb(req),
            headers={"Content-Type": "application/msgpack"},
        )
        r.raise_for_status()
        return r.content

    async def synthesize(self, text: str) -> AsyncIterator[bytes]:
        for sentence in split_for_tts(text)[:5]:
            try:
                pcm, rate = wav_to_pcm(await self._tts_call(sentence))
            except Exception as exc:
                print(f"[nova] ошибка fish-tts: {exc!r}")
                return
            if rate != self.sample_rate:
                print(f"[nova] fish-tts: частота {rate} (ожидалась {self.sample_rate})")
                self.sample_rate = rate
            yield pcm
