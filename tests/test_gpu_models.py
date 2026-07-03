"""Smoke-тесты GPU-моделей: на ноуте пропускаются (пакеты не установлены),
на инстансе гоняются вручную: uv run pytest tests/test_gpu_models.py"""
import pytest


def test_whisper_asr_transcribes_silence():
    pytest.importorskip("faster_whisper")
    from nova.server.models.whisper_asr import WhisperASR

    asr = WhisperASR(model_name="tiny", device="cpu")
    import asyncio

    text = asyncio.run(asr.transcribe(b"\x00\x00" * 16000, 16000))
    assert isinstance(text, str)


def test_xtts_yields_pcm_chunks():
    pytest.importorskip("TTS")
    import asyncio

    from nova.server.models.xtts_tts import XttsTTS

    tts = XttsTTS()

    async def run():
        return [c async for c in tts.synthesize("привет")]

    chunks = asyncio.run(run())
    assert chunks and all(isinstance(c, bytes) for c in chunks)
