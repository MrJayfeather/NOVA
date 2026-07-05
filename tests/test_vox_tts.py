import numpy as np

from nova.server.models.vox_tts import (
    cut_spoken_head, norm_word, normalize_peak, stress_to_acute,
)


def test_stress_plus_to_acute():
    # RUAccent пишет «пр+омах», VoxCPM2 понимает «про́мах» (U+0301 после гласной)
    assert stress_to_acute("каждый твой пр+омах") == "каждый твой про́мах"
    assert stress_to_acute("молок+о и хл+еб") == "молоко́ и хле́б"
    assert stress_to_acute("без ударений") == "без ударений"


def test_norm_word_strips_punct_and_yo():
    assert norm_word("Слушай,") == "слушай"
    assert norm_word("ЕЩЁ!") == "еще"


def test_cut_spoken_head_cuts_before_first_word():
    rate = 100  # секунда = 100 сэмплов, удобно считать
    pcm = np.arange(500, dtype=np.int16)  # 5 «секунд»
    words = [("and", 0.5), ("speaking", 1.0), ("Слушай,", 3.0), ("я", 3.5)]
    out = cut_spoken_head(pcm, rate, words, "Слушай", margin=0.12, fade=0.0)
    # срез на 3.0 - 0.12 = 2.88с -> сэмпл 288
    assert len(out) == 500 - 288
    assert out[0] == 288


def test_cut_spoken_head_fade_in():
    rate = 100
    pcm = np.full(500, 1000, dtype=np.int16)
    words = [("Слушай", 1.0)]
    out = cut_spoken_head(pcm, rate, words, "Слушай", margin=0.0, fade=0.1)
    assert out[0] == 0            # начало фейда — тишина
    assert out[20] == 1000        # после 10 сэмплов фейда — полная громкость


def test_cut_spoken_head_word_missing_returns_all():
    pcm = np.arange(100, dtype=np.int16)
    out = cut_spoken_head(pcm, 100, [("другое", 0.1)], "Слушай")
    assert len(out) == 100


def test_normalize_peak_boosts_quiet():
    pcm = np.array([0, 100, -100], dtype=np.int16)
    out = normalize_peak(pcm)
    assert int(np.abs(out).max()) == 23000


def test_normalize_peak_keeps_loud():
    pcm = np.array([0, 30000], dtype=np.int16)
    assert normalize_peak(pcm)[1] == 30000


def make_vox(**kw):
    """VoxTTS без загрузки моделей: генератор подменяется в тестах."""
    import asyncio

    from nova.server.models.vox_tts import VoxTTS

    tts = VoxTTS.__new__(VoxTTS)
    tts._ref_wav = "ref.wav"
    tts._ref_text = "текст"
    tts._tag = kw.get("tag", "(tag)")
    tts._stress = kw.get("stress", False)
    tts._seed = 42
    tts._timestamps = kw.get("word_timestamps")
    tts._validator = kw.get("validator")
    tts._model = object()   # «загружена»
    tts._accents = kw.get("accents")
    tts.sample_rate = 100
    tts._lock = asyncio.Lock()
    return tts


def test_prepare_adds_tag_and_stress():
    class Acc:
        def process_all(self, s):
            return s.replace("промах", "пр+омах")

    tts = make_vox(accents=Acc(), tag="(slow)")
    out = tts.prepare("[laughing] Твой промах.")
    assert out == "(slow)Твой про́мах."   # маркер снят, тег в начале


def test_prepare_no_tag_no_stress():
    tts = make_vox(tag="")
    assert tts.prepare("Привет.") == "Привет."


async def test_synthesize_sequential_cut_and_normalize():
    calls = []

    async def stamps(pcm, rate):
        return [("tag", 0.1), ("Привет", 1.0)]

    tts = make_vox(word_timestamps=stamps, tag="(slow)")

    def fake_gen(prepared, seed):
        calls.append((prepared, seed))
        return np.full(300, 100, dtype=np.int16)  # 3 «секунды» при rate=100

    tts._gen_sync = fake_gen
    chunks = [c async for c in tts.synthesize("Привет. Как дела.")]
    assert len(chunks) == 2
    assert calls[0][1] == 42 and calls[1][1] == 42
    first = np.frombuffer(chunks[0], dtype=np.int16)
    # срез: 1.0 - 0.12 = 0.88с -> 88 сэмплов долой из 300
    assert len(first) == 300 - 88
    assert int(np.abs(first).max()) == 23000   # нормализация


async def test_validator_fail_regenerates_with_new_seed():
    seeds = []

    async def guard(sentence, pcm, rate):
        return len(seeds) > 1   # первая генерация «заскок», вторая ок

    tts = make_vox(validator=guard, tag="")

    def fake_gen(prepared, seed):
        seeds.append(seed)
        return np.full(10, 500, dtype=np.int16)

    tts._gen_sync = fake_gen
    chunks = [c async for c in tts.synthesize("Одно предложение.")]
    assert seeds == [42, 43]    # пересинтез другим seed
    assert len(chunks) == 1


async def test_failed_sentence_skipped_not_fatal():
    n = [0]

    def fake_gen(prepared, seed):
        n[0] += 1
        if n[0] == 1:
            raise RuntimeError("модель икнула")
        return np.full(10, 500, dtype=np.int16)

    tts = make_vox(tag="")
    tts._gen_sync = fake_gen
    chunks = [c async for c in tts.synthesize("Первое. Второе.")]
    assert len(chunks) == 1


def test_build_vox_tts_reads_env(monkeypatch, tmp_path):
    from nova.server.models.vox_tts import VoxTTS, build_vox_tts

    (tmp_path / "voice_sample.wav").write_bytes(b"RIFF")
    (tmp_path / "voice_sample.txt").write_text("текст", encoding="utf-8")
    monkeypatch.setenv("NOVA_VOX_TAG", "(мой тег)")
    monkeypatch.setenv("NOVA_VOX_STRESS", "0")
    monkeypatch.setenv("NOVA_VOX_SEED", "7")

    class FakeASR:
        async def word_timestamps(self, pcm, rate):
            return []

    tts = build_vox_tts(FakeASR(), tmp_path, validator=None)
    assert isinstance(tts, VoxTTS)
    assert tts._tag == "(мой тег)"
    assert tts._stress is False
    assert tts._seed == 7
