import asyncio
import re
import threading
from pathlib import Path
from typing import AsyncIterator

from nova.server.models.base import TTSModel


def split_for_tts(text: str, limit: int = 180) -> list[str]:
    """XTTS искажает русскую речь на текстах длиннее ~182 символов —
    режем на предложения, длинные предложения — по словам."""
    parts = re.split(r"(?<=[.!?…])\s+", text.strip())
    out: list[str] = []
    for p in parts:
        p = p.strip()
        if not p:
            continue
        while len(p) > limit:
            cut = p[:limit]
            idx = max(cut.rfind(", "), cut.rfind(" "))
            if idx < 40:
                idx = limit
            out.append(p[:idx].strip())
            p = p[idx:].strip()
        if p:
            out.append(p)
    return out


class XttsTTS(TTSModel):
    sample_rate = 24000

    def __init__(self, speaker_wav: Path | None = None, default_speaker: str = "Ana Florence"):
        from TTS.tts.configs.xtts_config import XttsConfig
        from TTS.tts.models.xtts import Xtts
        from TTS.utils.manage import ModelManager

        model_dir, _, _ = ModelManager().download_model(
            "tts_models/multilingual/multi-dataset/xtts_v2"
        )
        config = XttsConfig()
        config.load_json(str(Path(model_dir) / "config.json"))
        self._model = Xtts.init_from_config(config)
        self._model.load_checkpoint(config, checkpoint_dir=str(model_dir), eval=True)
        try:
            self._model.cuda()
        except Exception:
            pass  # cpu-режим для smoke-теста
        if speaker_wav is not None and Path(speaker_wav).exists():
            self._latent, self._embedding = self._model.get_conditioning_latents(
                audio_path=[str(speaker_wav)]
            )
            print(f"[nova] голос клонирован из {speaker_wav}")
        else:
            spk = self._model.speaker_manager.speakers[default_speaker]
            self._latent = spk["gpt_cond_latent"]
            self._embedding = spk["speaker_embedding"]
            print(f"[nova] голос по умолчанию: {default_speaker}")

    async def synthesize(self, text: str) -> AsyncIterator[bytes]:
        out: asyncio.Queue = asyncio.Queue()
        loop = asyncio.get_running_loop()

        def produce():
            import numpy as np

            try:
                # максимум 5 предложений: длинные монологи рвутся паузами синтеза
                for sentence in split_for_tts(text)[:5]:
                    stream = self._model.inference_stream(
                        sentence, "ru", self._latent, self._embedding,
                        temperature=0.7,
                        repetition_penalty=5.0,
                        top_k=50,
                        top_p=0.85,
                    )
                    for chunk in stream:
                        pcm = (
                            (chunk.squeeze().clamp(-1, 1).cpu().numpy() * 32767)
                            .astype(np.int16)
                            .tobytes()
                        )
                        loop.call_soon_threadsafe(out.put_nowait, pcm)
            except Exception as exc:
                print(f"[nova] ошибка TTS: {exc!r}")
            finally:
                loop.call_soon_threadsafe(out.put_nowait, None)

        threading.Thread(target=produce, daemon=True).start()
        while True:
            item = await out.get()
            if item is None:
                break
            yield item
