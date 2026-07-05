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


async def test_whisper_word_timestamps_decimates_48k():
    from nova.server.models.whisper_asr import WhisperASR

    class W:  # слово faster-whisper
        def __init__(self, word, start):
            self.word, self.start = word, start

    class Seg:
        def __init__(self, words):
            self.words = words

    captured = {}

    class FakeModel:
        def transcribe(self, audio, **kw):
            captured["n"] = len(audio)
            captured["kw"] = kw
            return [Seg([W(" Слушай", 1.0), W(" я", 1.5)])], None

    asr = WhisperASR.__new__(WhisperASR)  # без загрузки весов
    asr._model = FakeModel()
    pcm = b"\x01\x00" * 48000  # 1 секунда 48кГц
    words = await asr.word_timestamps(pcm, 48000)
    assert captured["n"] == 16000            # децимация 3:1
    assert captured["kw"]["word_timestamps"] is True
    assert words == [(" Слушай", 1.0), (" я", 1.5)]  # времена уже честные


async def test_whisper_word_timestamps_scales_odd_rate():
    from nova.server.models.whisper_asr import WhisperASR

    class W:
        def __init__(self, word, start):
            self.word, self.start = word, start

    class Seg:
        def __init__(self, words):
            self.words = words

    class FakeModel:
        def transcribe(self, audio, **kw):
            return [Seg([W("привет", 2.756)])], None

    asr = WhisperASR.__new__(WhisperASR)
    asr._model = FakeModel()
    words = await asr.word_timestamps(b"\x01\x00" * 44100, 44100)
    # 44.1к некратна 16к: время whisper * 16000/44100
    assert abs(words[0][1] - 2.756 * 16000 / 44100) < 1e-6


def test_xtts_yields_pcm_chunks():
    pytest.importorskip("TTS")
    import asyncio

    from nova.server.models.xtts_tts import XttsTTS

    tts = XttsTTS()

    async def run():
        return [c async for c in tts.synthesize("привет")]

    chunks = asyncio.run(run())
    assert chunks and all(isinstance(c, bytes) for c in chunks)
