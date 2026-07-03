import wave
from io import BytesIO
from pathlib import Path
from typing import AsyncIterator

import httpx
import ormsgpack

from nova.server.models.base import TTSModel
from nova.server.models.xtts_tts import split_for_tts


def build_tts_request(text: str, ref_audio: bytes = b"", ref_text: str = "",
                      reference_id: str = "") -> dict:
    req = {
        "text": text,
        "format": "wav",
        "streaming": False,
    }
    if reference_id:
        # облачный fish.audio: готовая модель голоса по id
        req["reference_id"] = reference_id
    else:
        req["references"] = [{"audio": ref_audio, "text": ref_text}]
        # сервер кэширует закодированный референс — без этого он
        # перекодирует образец на каждое предложение (~1.5с)
        req["use_memory_cache"] = "on"
    return req


def wav_to_pcm(wav_bytes: bytes) -> tuple[bytes, int]:
    import numpy as np

    with wave.open(BytesIO(wav_bytes)) as w:
        rate = w.getframerate()
        pcm = w.readframes(w.getnframes())
        if w.getnchannels() == 2:
            arr = np.frombuffer(pcm, dtype=np.int16).reshape(-1, 2)
            pcm = arr.mean(axis=1).astype(np.int16).tobytes()
    # нормализация громкости: модель, выученная на тихом датасете, и сама
    # говорит тихо — поднимаем пик к ~70% шкалы
    arr = np.frombuffer(pcm, dtype=np.int16).astype(np.float32)
    peak = float(np.abs(arr).max())
    if 0 < peak < 23000:
        pcm = (arr * (23000.0 / peak)).astype(np.int16).tobytes()
    return pcm, rate


class FishTTS(TTSModel):
    """Голос через fish-speech api_server (локальный S1-mini) или облачный
    api.fish.audio (api_key + model + reference_id готового голоса)."""

    sample_rate = 44100  # DAC-кодек S1; сверяется с реальным ответом

    def __init__(self, url: str, reference_wav: Path | None = None,
                 reference_text: str = "", timeout: float = 120.0,
                 api_key: str = "", model: str = "", reference_id: str = ""):
        self._url = url
        self._ref_audio = Path(reference_wav).read_bytes() if reference_wav else b""
        self._ref_text = reference_text
        self._reference_id = reference_id
        self._headers = {"Content-Type": "application/msgpack"}
        if api_key:
            self._headers["Authorization"] = f"Bearer {api_key}"
        if model:
            # облако выбирает модель синтеза заголовком
            self._headers["model"] = model
        self._client = httpx.AsyncClient(timeout=timeout)

    async def _tts_call(self, sentence: str) -> bytes:
        req = build_tts_request(sentence, self._ref_audio, self._ref_text,
                                reference_id=self._reference_id)
        r = await self._client.post(self._url, content=ormsgpack.packb(req),
                                    headers=self._headers)
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
