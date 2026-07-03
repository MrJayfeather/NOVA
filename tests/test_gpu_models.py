"""Smoke-тесты GPU-моделей: на ноуте пропускаются (пакеты не установлены),
на инстансе гоняются вручную: uv run pytest tests/test_gpu_models.py"""
import pytest

from nova.server.models.xtts_tts import split_for_tts


def test_split_short_text_untouched():
    assert split_for_tts("Привет, как дела?") == ["Привет, как дела?"]


def test_split_long_text_by_sentences():
    text = "Первое предложение о чём-то важном. " * 3 + "И второе!"
    parts = split_for_tts(text, limit=60)
    assert all(len(p) <= 60 for p in parts)
    assert "".join(parts).replace(" ", "") == text.replace(" ", "")


def test_split_monster_sentence_by_words():
    text = "слово " * 60  # одно «предложение» на 360 символов
    parts = split_for_tts(text, limit=100)
    assert len(parts) >= 3
    assert all(len(p) <= 100 for p in parts)


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
