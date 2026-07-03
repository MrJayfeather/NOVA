import asyncio
import threading
from pathlib import Path
from typing import AsyncIterator

from nova.server.models.base import TTSModel


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
                stream = self._model.inference_stream(
                    text, "ru", self._latent, self._embedding
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
