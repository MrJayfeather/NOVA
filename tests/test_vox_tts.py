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
    out, cut = cut_spoken_head(pcm, rate, words, "Слушай", margin=0.12, fade=0.0)
    # срез на 3.0 - 0.12 = 2.88с -> сэмпл 288
    assert abs(cut - 2.88) < 1e-9
    assert len(out) == 500 - 288
    assert out[0] == 288


def test_cut_spoken_head_fuzzy_and_second_word():
    # whisper слышит «Слушай» как «слушои» — нечёткое совпадение;
    # или первое слово слилось с тегом — ловим по второму
    rate = 100
    pcm = np.arange(500, dtype=np.int16)
    fuzzy = [("слушои", 2.0)]
    out, cut = cut_spoken_head(pcm, rate, fuzzy, ["Слушай", "я"], margin=0.0)
    assert cut == 2.0
    # совпало ВТОРОЕ слово («я») — отступаем на одно слово whisper назад,
    # чтобы не съесть «Слушай» (оно и есть та «тарабарщина» перед ним)
    second = [("тарабарщина", 1.0), ("я", 2.5)]
    out, cut = cut_spoken_head(pcm, rate, second, ["Слушай", "я"], margin=0.0)
    assert cut == 1.0


def test_cut_spoken_head_fade_in():
    rate = 100
    pcm = np.full(500, 1000, dtype=np.int16)
    words = [("Слушай", 1.0)]
    out, cut = cut_spoken_head(pcm, rate, words, "Слушай", margin=0.0, fade=0.1)
    assert out[0] == 0            # начало фейда — тишина
    assert out[20] == 1000        # после 10 сэмплов фейда — полная громкость


def test_cut_spoken_head_word_missing_returns_all():
    pcm = np.arange(100, dtype=np.int16)
    out, cut = cut_spoken_head(pcm, 100, [("другое", 0.1)], "Слушай")
    assert cut is None
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
    tts._check_speech = kw.get("check_speech", False)
    tts._pause_factor = kw.get("pause_factor", 0.0)
    tts._use_dfn = False
    tts._dfn = kw.get("dfn")
    tts._model = object()   # «загружена»
    tts._emo_refs = kw.get("emo_refs", {})
    tts._accents = kw.get("accents")
    tts.sample_rate = 100
    tts._lock = asyncio.Lock()
    tts._gen_lock = asyncio.Lock()
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
        return [("tag", 0.1), ("Привет", 1.0), ("Как", 1.0)]

    tts = make_vox(word_timestamps=stamps, tag="(slow)")

    def fake_gen(prepared, seed, *a):
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


async def test_guard_fail_regenerates_with_new_seed():
    # страж на ТОМ ЖЕ прогоне whisper: первая генерация «заскок»
    calls = {"n": 0}
    seeds = []

    async def stamps(pcm, rate):
        calls["n"] += 1
        if calls["n"] == 1:
            return [("бурбуляция", 0.1)]        # услышано не то
        return [("одно", 0.1), ("предложение", 0.5)]

    tts = make_vox(word_timestamps=stamps, tag="", check_speech=True)

    def fake_gen(prepared, seed, *a):
        seeds.append(seed)
        return np.full(10, 500, dtype=np.int16)

    tts._gen_sync = fake_gen
    chunks = [c async for c in tts.synthesize("Одно предложение.")]
    assert seeds == [42, 43]    # пересинтез другим seed
    assert len(chunks) == 1


async def test_failed_sentence_skipped_not_fatal():
    n = [0]

    def fake_gen(prepared, seed, *a):
        n[0] += 1
        if n[0] == 1:
            raise RuntimeError("модель икнула")
        return np.full(10, 500, dtype=np.int16)

    tts = make_vox(tag="")
    tts._gen_sync = fake_gen
    chunks = [c async for c in tts.synthesize("Первое. Второе.")]
    assert len(chunks) == 1


def test_find_silence_cut_finds_gap():
    from nova.server.models.vox_tts import find_silence_cut

    rate = 1000
    loud = np.full(2000, 10000, dtype=np.int16)   # 2с речи (шапка)
    gap = np.zeros(400, dtype=np.int16)           # 0.4с паузы
    tail = np.full(3000, 10000, dtype=np.int16)   # реплика
    pcm = np.concatenate([loud, gap, tail])
    cut = find_silence_cut(pcm, rate)
    # конец паузы ~2.4с минус запас 0.05
    assert cut is not None
    assert abs(cut - 2.35) < 0.1


def test_find_silence_cut_none_when_no_gap():
    from nova.server.models.vox_tts import find_silence_cut

    pcm = np.full(5000, 10000, dtype=np.int16)    # сплошная речь
    assert find_silence_cut(pcm, 1000) is None


async def test_head_miss_falls_back_to_silence_then_skip():
    from nova.server.models.vox_tts import HeadCutMiss  # noqa: F401

    async def stamps(pcm, rate):
        return [("тарабарщина", 0.5)]              # слова реплики не найдены

    tts = make_vox(word_timestamps=stamps, tag="(slow)")
    rate = tts.sample_rate                         # 100

    # аудио с паузой: шапка 1с, пауза 0.5с, реплика 2с
    with_gap = np.concatenate([
        np.full(100, 9000, dtype=np.int16),
        np.zeros(50, dtype=np.int16),
        np.full(200, 9000, dtype=np.int16),
    ])
    tts._gen_sync = lambda prepared, seed, *a: with_gap
    chunks = [c async for c in tts.synthesize("Привет.")]
    assert len(chunks) == 1
    # срезано по паузе: осталась только реплика (~2с и чуть паузы)
    assert len(np.frombuffer(chunks[0], dtype=np.int16)) < 260

    # без паузы и без совпадения слов — предложение пропускается целиком
    tts2 = make_vox(word_timestamps=stamps, tag="(slow)")
    tts2._gen_sync = lambda prepared, seed, *a: np.full(400, 9000, dtype=np.int16)
    chunks2 = [c async for c in tts2.synthesize("Привет.")]
    assert chunks2 == []                           # тишина лучше английского


def test_stretch_pauses_extends_only_silence():
    from nova.server.models.vox_tts import stretch_pauses

    rate = 1000
    speech1 = np.full(500, 10000, dtype=np.int16)
    gap = np.zeros(200, dtype=np.int16)          # 0.2с тишины
    speech2 = np.full(500, 10000, dtype=np.int16)
    pcm = np.concatenate([speech1, gap, speech2])
    out = stretch_pauses(pcm, rate, factor=2.0)
    # пауза удвоилась (~+200 сэмплов), речь не тронута
    assert len(out) - len(pcm) >= 150
    assert int(np.abs(out).max()) == 10000


def test_stretch_pauses_noop_when_off():
    from nova.server.models.vox_tts import stretch_pauses

    pcm = np.full(100, 5000, dtype=np.int16)
    assert len(stretch_pauses(pcm, 100, factor=1.0)) == 100


async def test_dfn_polish_applied_when_loaded():
    called = []

    def fake_dfn(arr):
        called.append(len(arr))
        return arr

    tts = make_vox(tag="", dfn=fake_dfn)
    tts.sample_rate = 48000
    tts._gen_sync = lambda prepared, seed, *a: np.full(100, 500, dtype=np.int16)
    chunks = [c async for c in tts.synthesize("Привет.")]
    assert called == [100]           # полировка прошла по каждому предложению
    assert len(chunks) == 1


def test_build_vox_tts_prefers_vox_reference(monkeypatch, tmp_path):
    from nova.server.models.vox_tts import build_vox_tts

    (tmp_path / "voice_sample.wav").write_bytes(b"RIFF")
    (tmp_path / "voice_sample.txt").write_text("общий", encoding="utf-8")
    (tmp_path / "voice_sample_vox.wav").write_bytes(b"RIFF")
    (tmp_path / "voice_sample_vox.txt").write_text("миксовый", encoding="utf-8")
    monkeypatch.setenv("NOVA_VOX_PAUSES", "1.5")

    class FakeASR:
        async def word_timestamps(self, pcm, rate):
            return []

    tts = build_vox_tts(FakeASR(), tmp_path)
    assert tts._ref_text == "миксовый"
    assert tts._ref_wav.endswith("voice_sample_vox.wav")
    assert tts._pause_factor == 1.5
    assert tts._tag == ""            # тег по умолчанию выключен


async def test_warmup_loads_and_generates_once():
    gens = []
    tts = make_vox(tag="(slow)")

    def fake_gen(prepared, seed, *a):
        gens.append(prepared)
        return np.full(10, 500, dtype=np.int16)

    tts._gen_sync = fake_gen
    await tts.warmup()
    assert len(gens) == 1        # одна холостая генерация — модель горячая


def test_build_vox_tts_reads_env(monkeypatch, tmp_path):
    from nova.server.models.vox_tts import VoxTTS, build_vox_tts

    (tmp_path / "voice_sample.wav").write_bytes(b"RIFF")
    (tmp_path / "voice_sample.txt").write_text("текст", encoding="utf-8")
    monkeypatch.setenv("NOVA_VOX_TAG", "(мой тег)")
    monkeypatch.setenv("NOVA_VOX_STRESS", "0")
    monkeypatch.setenv("NOVA_VOX_SEED", "7")
    monkeypatch.setenv("NOVA_TTS_GUARD", "0")

    class FakeASR:
        async def word_timestamps(self, pcm, rate):
            return []

    tts = build_vox_tts(FakeASR(), tmp_path)
    assert isinstance(tts, VoxTTS)
    assert tts._tag == "(мой тег)"
    assert tts._stress is False
    assert tts._seed == 7
    assert tts._check_speech is False


# ---- эмоции: референс по маркеру мозга ----

def test_emotion_for_markers():
    from nova.server.models.vox_tts import emotion_for

    assert emotion_for("[laughing] Ну ты дал!") == "joy"
    assert emotion_for("[excited] Погнали!") == "joy"
    assert emotion_for("[sarcastic] Ну конечно.") == "tease"
    assert emotion_for("[whispering] Тихо...") == "soft"
    assert emotion_for("[soft tone] Всё хорошо.") == "soft"
    assert emotion_for("Обычная фраза.") is None
    # маркер не в начале — тоже считается
    assert emotion_for("Ага [chuckling] смешно.") == "joy"


async def test_synthesize_switches_reference_by_marker(tmp_path):
    import numpy as np

    joy_ref = tmp_path / "joy.wav"
    joy_ref.write_bytes(b"RIFF")
    used_refs = []

    tts = make_vox(tag="")
    tts._emo_refs = {"joy": (str(joy_ref), "радостный текст")}

    def fake_gen(prepared, seed, ref_wav=None, ref_text=None):
        used_refs.append(ref_wav)
        return np.full(10, 500, dtype=np.int16)

    tts._gen_sync = fake_gen
    chunks = [c async for c in tts.synthesize(
        "[laughing] Смешно же! Обычное предложение.")]
    assert len(chunks) == 2
    assert used_refs[0] == str(joy_ref)      # маркерное — бодрым референсом
    assert used_refs[1] == "ref.wav"         # обычное — базовым


# ---- точечные ударения: свой словарик, знак U+0301, без RUAccent ----

def test_apply_stress_fixes():
    from nova.server.models.vox_tts import apply_stress_fixes

    assert apply_stress_fixes("под присмотром камер") == \
        "под присмо́тром камер"
    assert apply_stress_fixes("Присмотром") == "Присмо́тром"
    # чужие слова не трогаем, часть слова не матчится
    assert apply_stress_fixes("присмотрелась") == "присмотрелась"


def test_prepare_applies_stress_fixes_without_ruaccent():
    tts = make_vox(tag="", accents=None)
    assert tts.prepare("Под присмотром.") == "Под присмо́тром."


# ---- разрывная речь: без растяжки пауз, тишина на стыках срезается ----

def test_trim_edge_silence_cuts_dead_air():
    from nova.server.models.vox_tts import trim_edge_silence

    rate = 100
    pcm = np.concatenate([
        np.zeros(80, dtype=np.int16),            # 0.8с тишины в начале
        np.full(100, 9000, dtype=np.int16),      # 1с речи
        np.zeros(120, dtype=np.int16),           # 1.2с тишины в конце
    ])
    out = trim_edge_silence(pcm, rate, keep=0.15)
    # осталось: 0.15с + 1с + 0.15с = 130 сэмплов
    assert len(out) == 130
    assert out[20] == 9000


def test_trim_edge_silence_keeps_silent_and_short():
    from nova.server.models.vox_tts import trim_edge_silence

    silent = np.zeros(100, dtype=np.int16)
    assert len(trim_edge_silence(silent, 100)) == 100  # вся тишина — не трогаем
    tight = np.full(50, 9000, dtype=np.int16)
    assert len(trim_edge_silence(tight, 100)) == 50    # нечего резать


def test_merge_lone_markers():
    from nova.server.models.vox_tts import merge_lone_markers

    # одинокий маркер приклеивается к следующему предложению
    assert merge_lone_markers(["[laughing]", "Ну ты дал."]) == \
        ["[laughing] Ну ты дал."]
    # хвостовой одинокий маркер выбрасывается
    assert merge_lone_markers(["Всё, довольен?", "[laughing]"]) == \
        ["Всё, довольен?"]
    assert merge_lone_markers(["Привет."]) == ["Привет."]


async def test_synthesize_skips_marker_only_sentence():
    tts = make_vox(tag="")
    calls = []

    def fake_gen(prepared, seed, *a):
        calls.append(prepared)
        return np.full(10, 500, dtype=np.int16)

    tts._gen_sync = fake_gen
    chunks = [c async for c in tts.synthesize("Всё, довольен? [laughing]")]
    assert len(chunks) == 1
    assert calls == ["Всё, довольен?"]   # пустышка в модель не ушла
