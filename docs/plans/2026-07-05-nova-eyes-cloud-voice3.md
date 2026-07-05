# План: «Глаза в облако + Голос 3.0»

> Спека: docs/specs/2026-07-05-nova-eyes-cloud-voice3.md. Задачи с
> чекбоксами, каждая заканчивается зелёными тестами и коммитом.

**Цель:** зрение NOVA уезжает в Gemini (описания кадров вместо картинок в
мозг), освободившаяся VRAM отдаётся локальному голосу VoxCPM2 (48кГц,
рецепт-чемпион из scripts/voxcpm2_bench.py).

**Архитектура:** GeminiEyes — обёртка над QwenVLM (тот же интерфейс
VisionLLM): описывает кадры через Gemini REST, мозг вызывается без
картинок. VoxTTS — движок TTSModel: VoxCPM2 в процессе сервера, RUAccent
для ударений, срез наговоренного тега по word-timestamps whisper-модели
СТТ-стража. Сервер NOVA на инстансе переезжает в venv /workspace/vox
(--system-site-packages): voxcpm/ruaccent не трогают системный питон vLLM.

**Стек:** httpx (Gemini REST, без SDK), voxcpm 2.0.3, ruaccent,
faster-whisper (уже есть), numpy.

## Глобальные ограничения

- НИКАКИХ упоминаний Claude/Anthropic/AI-инструментов в коде/коммитах.
- Ветка eyes-voice3 от master; merge в master без PR; на инстанс код
  приходит только после push в master.
- Русские комментарии, стиль существующего кода (см. fish_tts.py).
- Тесты: `uv run pytest -q` — 95 passed, 2 skipped до начала; после
  каждой задачи столько же + новые, ноль красных.
- Секреты: GEMINI_KEY только в .env (гитигнорен) и /workspace/gemini_key.
- Дефолтная модель глаз: gemini-3.1-flash-lite (проверена на ключе).
- Рецепт голоса (не менять без новых ушей): тег
  «(Speaking very slowly, at a calm and relaxed pace)», ударения U+0301,
  seed 42, срез шапки margin 0.12с + fade 0.02с, пик-нормализация 23000.

## Карта файлов

| Файл | Роль |
|---|---|
| nova/server/models/vox_tts.py (новый) | движок VoxTTS + чистые хелперы (ударения, срез, нормализация) |
| nova/server/models/gemini_vision.py (новый) | GeminiEyes + фабрика wrap_eyes |
| nova/server/models/whisper_asr.py | + метод word_timestamps |
| nova/server/main.py | страж наружу из ветки fishcloud; ветка voxcpm; wrap_eyes |
| deploy/onstart.sh | провижининг /workspace/vox + веса VoxCPM2 |
| deploy/runner.sh | util 0.70 при облачных глазах; env; сервер из vox-venv |
| scripts/vast.py | проброс GEMINI_KEY/NOVA_EYES/NOVA_TTS |
| tests/test_vox_tts.py, tests/test_gemini_vision.py (новые), tests/test_gpu_models.py, tests/test_vast.py | тесты |

---

### Задача 1: чистые хелперы голоса (vox_tts.py, без модели)

**Файлы:** создать `nova/server/models/vox_tts.py`, создать
`tests/test_vox_tts.py`.

**Производит:** `stress_to_acute(text) -> str`, `norm_word(w) -> str`,
`cut_spoken_head(pcm: np.ndarray[int16], rate: int, words: list[tuple[str, float]], first_word: str, margin=0.12, fade=0.02) -> np.ndarray`,
`normalize_peak(pcm: np.ndarray[int16], target=23000) -> np.ndarray`,
константа `DEFAULT_TAG`.

- [ ] **Шаг 1.1: тесты (падают)** — в `tests/test_vox_tts.py`:

```python
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
```

- [ ] **Шаг 1.2:** `uv run pytest tests/test_vox_tts.py -q` → FAIL
  (ModuleNotFoundError).

- [ ] **Шаг 1.3: реализация** — `nova/server/models/vox_tts.py`:

```python
import re

import numpy as np

# рецепт-чемпион (отслушан 05.07): темп «slower»; без very — «летит»
DEFAULT_TAG = "(Speaking very slowly, at a calm and relaxed pace)"

_VOWELS = "аеёиоуыэюяАЕЁИОУЫЭЮЯ"


def stress_to_acute(text: str) -> str:
    """RUAccent ставит «+» перед ударной гласной («пр+омах»); VoxCPM2
    проверенно понимает combining acute ПОСЛЕ неё («про́мах»)."""
    return re.sub(rf"\+([{_VOWELS}])", "\\1́", text)


def norm_word(w: str) -> str:
    return re.sub(r"[^\wёа-яЁА-Я]", "", w.lower()).replace("ё", "е")


def cut_spoken_head(pcm: np.ndarray, rate: int,
                    words: list[tuple[str, float]], first_word: str,
                    margin: float = 0.12, fade: float = 0.02) -> np.ndarray:
    """Модель наговаривает стилевой тег в начале — режем всё до первого
    слова реплики. Запас margin, фейд-ин против дрожи на границе среза."""
    target = norm_word(first_word)
    start = None
    for w, t in words:
        if norm_word(w) == target:
            start = max(0.0, t - margin)
            break
    if start is None:
        return pcm
    out = pcm[int(start * rate):].copy()
    n = min(int(fade * rate), len(out))
    if n:
        ramp = np.linspace(0.0, 1.0, n, dtype=np.float32)
        out[:n] = (out[:n].astype(np.float32) * ramp).astype(np.int16)
    return out


def normalize_peak(pcm: np.ndarray, target: int = 23000) -> np.ndarray:
    """Пик к ~70% шкалы — как у остальных движков (см. fish_tts)."""
    peak = float(np.abs(pcm).max()) if len(pcm) else 0.0
    if 0 < peak < target:
        pcm = (pcm.astype(np.float32) * (target / peak)).astype(np.int16)
    return pcm
```

- [ ] **Шаг 1.4:** `uv run pytest tests/test_vox_tts.py -q` → 7 passed.

- [ ] **Шаг 1.5:**

```bash
git add nova/server/models/vox_tts.py tests/test_vox_tts.py
git commit -m "feat: voice 3.0 helpers - stress notation, head cut, peak"
```

---

### Задача 2: word_timestamps у WhisperASR

**Файлы:** изменить `nova/server/models/whisper_asr.py`, тесты в
`tests/test_gpu_models.py` (там уже мокается WhisperASR).

**Производит:** `WhisperASR.word_timestamps(pcm: bytes, sample_rate: int)
-> list[tuple[str, float]]` — слова и стартовые секунды В РЕАЛЬНОМ
времени записи (не в whisper-домене).

**Тонкость:** faster-whisper считает вход 16кГц. Существующий transcribe
кормит его 44.1к «как есть» (работает — страж ловит заскоки), но для
таймштампов нужна честная шкала: при кратной частоте (48к = 3×16к)
децимируем массив, при некратной — масштабируем времена.

- [ ] **Шаг 2.1: тест (падает)** — добавить в `tests/test_gpu_models.py`:

```python
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
    pcm = (b"\x01\x00" * 48000)  # 1 секунда 48кГц
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
```

- [ ] **Шаг 2.2:** `uv run pytest tests/test_gpu_models.py -q` → FAIL
  (нет атрибута word_timestamps).

- [ ] **Шаг 2.3: реализация** — добавить в класс WhisperASR:

```python
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
```

- [ ] **Шаг 2.4:** `uv run pytest tests/test_gpu_models.py -q` → passed.

- [ ] **Шаг 2.5:**

```bash
git add nova/server/models/whisper_asr.py tests/test_gpu_models.py
git commit -m "feat: word timestamps from whisper for tts head cut"
```

---

### Задача 3: движок VoxTTS

**Файлы:** дополнить `nova/server/models/vox_tts.py`, тесты в
`tests/test_vox_tts.py`.

**Потребляет:** хелперы задачи 1; `split_for_tts` из xtts_tts;
`strip_markers` из tts_text; `word_timestamps` (сигнатура задачи 2).

**Производит:** `VoxTTS(reference_wav: Path, reference_text: str,
tag: str = DEFAULT_TAG, stress: bool = True, seed: int = 42,
word_timestamps=None, validator=None)` — TTSModel, sample_rate 48000.
Фабрика `build_vox_tts(asr, ref_dir: Path, validator) -> VoxTTS`
(читает env NOVA_VOX_TAG/NOVA_VOX_STRESS/NOVA_VOX_SEED).

**Решения:** предложения синтезируются ПОСЛЕДОВАТЕЛЬНО (одна GPU-модель,
параллелить нечем — не копировать газер из fish). При провале стража
пересинтез с seed+1 (заскок детерминирован по (текст×seed) — тот же seed
дал бы тот же заскок). Ошибка предложения — пропуск, реплика продолжается.
Ошибка загрузки RUAccent — ударения молча выключаются (голос важнее).

- [ ] **Шаг 3.1: тесты (падают)** — добавить в `tests/test_vox_tts.py`:

```python
import numpy as np


def make_vox(monkeypatch=None, **kw):
    """VoxTTS без загрузки моделей: генератор подменён."""
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
    import asyncio
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
```

- [ ] **Шаг 3.2:** `uv run pytest tests/test_vox_tts.py -q` → FAIL.

- [ ] **Шаг 3.3: реализация** — дополнить vox_tts.py:

```python
import asyncio
import os
from pathlib import Path
from typing import AsyncIterator

from nova.server.models.base import TTSModel
from nova.server.models.xtts_tts import split_for_tts
from nova.server.tts_text import strip_markers


class VoxTTS(TTSModel):
    """Локальный голос 3.0: VoxCPM2 (48кГц) по рецепту-чемпиону — тег
    темпа, ударения RUAccent (U+0301), срез наговоренной шапки."""

    sample_rate = 48000

    def __init__(self, reference_wav: Path, reference_text: str,
                 tag: str = DEFAULT_TAG, stress: bool = True, seed: int = 42,
                 word_timestamps=None, validator=None):
        # word_timestamps(pcm, rate) -> [(слово, старт_с)] — whisper стража
        # validator(text, pcm, rate) -> bool — тот же страж, что у fish
        self._ref_wav = str(reference_wav)
        self._ref_text = reference_text
        self._tag = tag
        self._stress = stress
        self._seed = seed
        self._timestamps = word_timestamps
        self._validator = validator
        self._model = None
        self._accents = None
        self._lock = asyncio.Lock()

    def _load_sync(self) -> None:
        from voxcpm import VoxCPM

        self._model = VoxCPM.from_pretrained(
            "openbmb/VoxCPM2", load_denoiser=False)
        self.sample_rate = self._model.tts_model.sample_rate
        if self._stress:
            try:
                from ruaccent import RUAccent

                acc = RUAccent()
                acc.load(omograph_model_size="turbo", use_dictionary=True)
                self._accents = acc
            except Exception as exc:
                # голос важнее ударений: страховка stress0
                print(f"[nova] RUAccent не поднялся, ударения выключены: {exc!r}")

    def prepare(self, sentence: str) -> str:
        s = strip_markers(sentence)
        if self._accents is not None:
            s = stress_to_acute(self._accents.process_all(s))
        return self._tag + s if self._tag else s

    def _gen_sync(self, prepared: str, seed: int):
        import numpy as np
        import torch

        torch.manual_seed(seed)
        wav = self._model.generate(
            text=prepared, prompt_wav_path=self._ref_wav,
            prompt_text=self._ref_text, reference_wav_path=self._ref_wav)
        arr = np.asarray(wav, dtype=np.float32) * 32767.0
        return arr.clip(-32768, 32767).astype(np.int16)

    async def _sentence_pcm(self, sentence: str, seed: int) -> bytes:
        import numpy as np

        pcm = await asyncio.to_thread(self._gen_sync, self.prepare(sentence), seed)
        if self._tag and self._timestamps:
            words = await self._timestamps(pcm.tobytes(), self.sample_rate)
            first = strip_markers(sentence).split()
            if first:
                pcm = cut_spoken_head(pcm, self.sample_rate, words, first[0])
        return normalize_peak(pcm).tobytes()

    async def synthesize(self, text: str) -> AsyncIterator[bytes]:
        async with self._lock:
            if self._model is None:
                await asyncio.to_thread(self._load_sync)
        # одна GPU-модель — последовательно (в отличие от облака fish)
        for sentence in split_for_tts(strip_markers(text))[:20]:
            try:
                pcm = await self._sentence_pcm(sentence, self._seed)
                if self._validator and not await self._validator(
                        sentence, pcm, self.sample_rate):
                    # заскок детерминирован по (текст, seed): тот же seed
                    # дал бы тот же заскок — пересинтез со сдвигом
                    print(f"[nova] vox-tts: сверка провалена, пересинтез: {sentence[:50]!r}")
                    pcm = await self._sentence_pcm(sentence, self._seed + 1)
            except Exception as exc:
                print(f"[nova] ошибка vox-tts (предложение пропущено): {exc!r}")
                continue
            yield pcm


def build_vox_tts(asr, ref_dir: Path, validator) -> VoxTTS:
    return VoxTTS(
        reference_wav=ref_dir / "voice_sample.wav",
        reference_text=(ref_dir / "voice_sample.txt").read_text(
            encoding="utf-8").strip(),
        tag=os.environ.get("NOVA_VOX_TAG", DEFAULT_TAG),
        stress=os.environ.get("NOVA_VOX_STRESS", "1") != "0",
        seed=int(os.environ.get("NOVA_VOX_SEED", "42")),
        word_timestamps=asr.word_timestamps,
        validator=validator,
    )
```

Внимание: в тесте `make_vox` НЕ вызывает `__init__` — если меняешь поля,
меняй и фабрику теста.

- [ ] **Шаг 3.4:** `uv run pytest tests/test_vox_tts.py -q` → passed
  (13 тестов).

- [ ] **Шаг 3.5:**

```bash
git add nova/server/models/vox_tts.py tests/test_vox_tts.py
git commit -m "feat: voxcpm engine - local 48khz voice with champion recipe"
```

---

### Задача 4: GeminiEyes + wrap_eyes

**Файлы:** создать `nova/server/models/gemini_vision.py`, создать
`tests/test_gemini_vision.py`.

**Производит:** `GeminiEyes(inner: VisionLLM, api_key: str,
model="gemini-3.1-flash-lite", timeout=20.0, max_frames=4)` — VisionLLM;
`GeminiEyes.describe(frames: list[bytes]) -> str`;
`wrap_eyes(inner: VisionLLM, env: dict | None = None) -> VisionLLM`.

- [ ] **Шаг 4.1: тесты (падают)** — `tests/test_gemini_vision.py`:

```python
from nova.server.models.gemini_vision import GeminiEyes, wrap_eyes


class FakeInner:
    def __init__(self):
        self.calls = []

    async def reply_to_user(self, text, frames, history):
        self.calls.append(("reply", text, frames))
        return "ответ"

    async def comment_on_event(self, event, frames, history):
        self.calls.append(("comment", event, frames))
        return "коммент"


def make_eyes(inner, describe_text="1: рабочий стол"):
    eyes = GeminiEyes.__new__(GeminiEyes)
    eyes._inner = inner
    eyes._model = "m"
    eyes._max_frames = 4
    eyes._cache = {}
    eyes._last_summary = ""
    eyes.gemini_calls = 0

    async def fake_call(frames, prompt):
        eyes.gemini_calls += 1
        return describe_text

    eyes._call_gemini = fake_call
    return eyes


async def test_reply_injects_description_and_drops_frames():
    inner = FakeInner()
    eyes = make_eyes(inner)
    out = await eyes.reply_to_user("что видишь?", [b"jpeg1"], [])
    assert out == "ответ"
    kind, text, frames = inner.calls[0]
    assert frames == []                      # мозг больше не получает картинок
    assert "рабочий стол" in text
    assert "что видишь?" in text


async def test_describe_cached_by_frame_bytes():
    eyes = make_eyes(FakeInner())
    await eyes.describe([b"jpeg1"])
    await eyes.describe([b"jpeg1"])          # тот же кадр — без запроса
    assert eyes.gemini_calls == 1


async def test_describe_failure_honest_stub():
    eyes = make_eyes(FakeInner())

    async def boom(frames, prompt):
        raise RuntimeError("облако закрылось")

    eyes._call_gemini = boom
    desc = await eyes.describe([b"jpeg1"])
    assert desc == "экран виден плохо"       # честно, без выдумок


async def test_comment_includes_description():
    inner = FakeInner()
    eyes = make_eyes(inner)
    await eyes.comment_on_event("окно сменилось", [b"jpeg1"], [])
    kind, event, frames = inner.calls[0]
    assert frames == []
    assert "рабочий стол" in event


async def test_no_frames_reply_passthrough():
    inner = FakeInner()
    eyes = make_eyes(inner)
    await eyes.reply_to_user("привет", [], [])
    assert inner.calls[0][1] == "привет"     # без вставки [экран: ]


def test_wrap_eyes_modes():
    inner = FakeInner()
    assert wrap_eyes(inner, {}) is inner                      # нет ключа
    assert wrap_eyes(inner, {"GEMINI_KEY": "k",
                             "NOVA_EYES": "local"}) is inner  # выключено
    wrapped = wrap_eyes(inner, {"GEMINI_KEY": "k"})
    assert isinstance(wrapped, GeminiEyes)                    # дефолт gemini
```

- [ ] **Шаг 4.2:** `uv run pytest tests/test_gemini_vision.py -q` → FAIL.

- [ ] **Шаг 4.3: реализация** — `nova/server/models/gemini_vision.py`:

```python
import base64
import hashlib

import httpx

from nova.server.models.base import VisionLLM

GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta"
BAD_SCREEN = "экран виден плохо"

DESCRIBE_PROMPT = (
    "Это кадры экрана пользователя по порядку, последний — самый свежий. "
    "Опиши каждый кадр одной короткой строкой по-русски, только факты: "
    "какие программы/игра/текст/действие видно. Формат строки: «N: ...». "
    "Если кадр почти не отличается от предыдущего — пиши «N: то же». {prev}"
)


class GeminiEyes(VisionLLM):
    """Облачные глаза: Gemini описывает кадры текстом, мозг получает
    описания вместо картинок — контекст в ~20 раз дешевле, KV-кэшу
    хватает util 0.70 (см. спеку eyes-cloud-voice3)."""

    def __init__(self, inner: VisionLLM, api_key: str,
                 model: str = "gemini-3.1-flash-lite",
                 timeout: float = 20.0, max_frames: int = 4):
        self._inner = inner
        self._model = model
        self._max_frames = max_frames
        self._cache: dict[str, str] = {}   # sha1(jpeg) -> строка описания
        self._last_summary = ""
        self._client = httpx.AsyncClient(
            base_url=GEMINI_URL, timeout=timeout,
            headers={"x-goog-api-key": api_key})

    async def _call_gemini(self, frames: list[bytes], prompt: str) -> str:
        parts = [{"inline_data": {"mime_type": "image/jpeg",
                                  "data": base64.b64encode(f).decode()}}
                 for f in frames]
        parts.append({"text": prompt})
        r = await self._client.post(
            f"/models/{self._model}:generateContent",
            json={"contents": [{"parts": parts}]})
        r.raise_for_status()
        return r.json()["candidates"][0]["content"]["parts"][0]["text"].strip()

    async def describe(self, frames: list[bytes]) -> str:
        frames = frames[-self._max_frames:]
        if not frames:
            return ""
        keys = [hashlib.sha1(f).hexdigest() for f in frames]
        fresh = [(k, f) for k, f in zip(keys, frames) if k not in self._cache]
        if fresh:
            prev = (f"Прошлая сводка: {self._last_summary}"
                    if self._last_summary else "")
            try:
                text = await self._call_gemini(
                    [f for _, f in fresh], DESCRIBE_PROMPT.format(prev=prev))
            except Exception as exc:
                # цензура/лимит/сеть: честная заглушка, НЕ выдумываем
                print(f"[nova] глаза-облако недоступны: {exc!r}")
                text = ""
            lines = [l.split(":", 1)[-1].strip()
                     for l in text.splitlines() if l.strip()]
            for (k, _), line in zip(fresh, lines):
                self._cache[k] = line
            for k, _ in fresh:
                self._cache.setdefault(k, BAD_SCREEN)
            # кадры уходят из deque клиента — чистим и кэш
            while len(self._cache) > 64:
                del self._cache[next(iter(self._cache))]
        desc = "; ".join(self._cache[k] for k in keys)
        self._last_summary = desc
        return desc

    async def reply_to_user(
        self, text: str, frames: list[bytes], history: list[dict]
    ) -> str:
        if frames:
            desc = await self.describe(frames[-2:])
            text = f"[экран: {desc}]\n{text}"
        return await self._inner.reply_to_user(text, [], history)

    async def comment_on_event(
        self, event: str, frames: list[bytes], history: list[dict]
    ) -> str:
        desc = await self.describe(frames) or "кадров нет"
        return await self._inner.comment_on_event(
            f"{event}; на экране: {desc}", [], history)


def wrap_eyes(inner: VisionLLM, env: dict | None = None) -> VisionLLM:
    """NOVA_EYES=gemini (дефолт при наличии ключа) | local (как раньше)."""
    if env is None:
        import os

        env = dict(os.environ)
    if env.get("NOVA_EYES", "gemini") != "gemini" or not env.get("GEMINI_KEY"):
        if env.get("NOVA_EYES", "gemini") == "gemini":
            print("[nova] нет GEMINI_KEY — глаза остаются локальными")
        return inner
    return GeminiEyes(
        inner, api_key=env["GEMINI_KEY"],
        model=env.get("NOVA_GEMINI_MODEL", "gemini-3.1-flash-lite"))
```

- [ ] **Шаг 4.4:** `uv run pytest tests/test_gemini_vision.py -q` → passed.

- [ ] **Шаг 4.5:**

```bash
git add nova/server/models/gemini_vision.py tests/test_gemini_vision.py
git commit -m "feat: cloud eyes - frame descriptions instead of images"
```

---

### Задача 5: подключение в build_models

**Файлы:** изменить `nova/server/main.py:17-79`.

**Потребляет:** `wrap_eyes` (задача 4), `build_vox_tts` (задача 3).

Правки в build_models:
1. `_guard` выезжает из ветки fishcloud наверх (нужен и voxcpm);
2. `llm = wrap_eyes(llm)` сразу после создания QwenVLM;
3. новая ветка `voxcpm` ПЕРВОЙ в цепочке (до fishcloud).

- [ ] **Шаг 5.1: реализация** — build_models становится таким
  (полностью, вместо строк 17–79):

```python
def build_models(mock: bool, persona_prompt: str):
    if mock:
        return MockASR(), MockLLM(persona_prompt=persona_prompt), MockTTS()
    from nova.server.models.gemini_vision import wrap_eyes
    from nova.server.models.qwen_llm import QwenVLM
    from nova.server.models.whisper_asr import WhisperASR
    from nova.server.models.xtts_tts import XttsTTS
    from nova.server.tts_text import speech_matches, strip_markers

    asr = WhisperASR(model_name=os.environ.get("NOVA_WHISPER", "large-v3-turbo"))
    llm = wrap_eyes(QwenVLM(
        persona_prompt=persona_prompt,
        base_url=os.environ.get("NOVA_VLLM_URL", "http://127.0.0.1:5000/v1"),
        model=os.environ.get("NOVA_MODEL", "Qwen/Qwen3.6-27B-FP8"),
    ))
    persona = os.environ.get("NOVA_PERSONA", "nova")
    ref_dir = Path("personas") / persona
    ref_txt = ref_dir / "voice_sample.txt"
    mode = os.environ.get("NOVA_TTS", "xtts")

    async def _guard(sentence: str, pcm: bytes, rate: int) -> bool:
        # СТТ-страж от робо-заскоков: сверяем синтез с текстом; при
        # любой ошибке стража голос важнее — пропускаем как есть
        if os.environ.get("NOVA_TTS_GUARD", "1") != "1":
            return True
        try:
            heard = await asr.transcribe(pcm, rate)
        except Exception as exc:
            print(f"[nova] страж синтеза недоступен: {exc!r}")
            return True
        return speech_matches(strip_markers(sentence), heard)

    if mode == "voxcpm" and ref_txt.exists():
        from nova.server.models.vox_tts import build_vox_tts

        tts = build_vox_tts(asr, ref_dir, validator=_guard)
    elif mode == "fishcloud" and os.environ.get("NOVA_FISH_KEY"):
        from nova.server.models.fish_tts import FishTTS

        tts = FishTTS(
            validator=_guard,
            url="https://api.fish.audio/v1/tts",
            api_key=os.environ["NOVA_FISH_KEY"],
            model=os.environ.get("NOVA_FISH_MODEL", "s2.1-pro-free"),
            reference_id=os.environ.get(
                "NOVA_FISH_REF_ID", "4075192824f64dc6aabbbf70124d6a01"),
            # затор free-очереди: лучше быстро пропустить предложение,
            # чем молчать минуту
            timeout=35.0,
            # ниже температура — стабильнее голос (меньше «выпадений из
            # роли»), живость добирается эмоциональными ремарками
            temperature=float(os.environ.get("NOVA_FISH_TEMP", "0.5")),
            top_p=float(os.environ.get("NOVA_FISH_TOP_P", "0.6")),
        )
    elif mode in ("fish", "fishcloud", "voxcpm") and ref_txt.exists():
        if mode != "fish":
            print(f"[nova] {mode} недоступен — откатываюсь на локальный fish")
        from nova.server.models.fish_tts import FishTTS

        tts = FishTTS(
            url=os.environ.get("NOVA_FISH_URL", "http://127.0.0.1:8081/v1/tts"),
            reference_wav=ref_dir / "voice_sample.wav",
            reference_text=ref_txt.read_text(encoding="utf-8").strip(),
        )
    else:
        if mode != "xtts":
            print("[nova] нет voice_sample.txt — откатываюсь на xtts")
        tts = XttsTTS(speaker_wav=ref_dir / "voice_sample.wav")
    return asr, llm, tts
```

(Ветка voxcpm без voice_sample.txt падает в локальный fish — тот же
референс-механизм, что и раньше. Импорты speech_matches/strip_markers
подняты наверх функции — из ветки fishcloud убраны.)

- [ ] **Шаг 5.2:** `uv run pytest -q` → все зелёные (mock-путь не тронут,
  реальный путь кроют тесты задач 3–4).

- [ ] **Шаг 5.3:**

```bash
git add nova/server/main.py
git commit -m "feat: wire cloud eyes and voxcpm voice into model builder"
```

---

### Задача 6: деплой (onstart, runner, vast)

**Файлы:** изменить `deploy/onstart.sh`, `deploy/runner.sh`,
`scripts/vast.py`, тест в `tests/test_vast.py`.

- [ ] **Шаг 6.1: onstart.sh** — после блока fishenv (строка 41), перед
  блоком checkpoints добавить:

```bash
# голос 3.0 (VoxCPM2) + сервер NOVA: venv поверх системных пакетов —
# системный python не трогаем (в нём живёт vLLM и его зависимости)
if [ ! -d /workspace/vox ]; then
  python3 -m venv --system-site-packages /workspace/vox
  /workspace/vox/bin/pip install voxcpm ruaccent -i https://pypi.org/simple \
    > /workspace/voxpip.log 2>&1
  /workspace/vox/bin/pip install -e /workspace/NOVA >> /workspace/voxpip.log 2>&1
fi
# веса VoxCPM2 — с повторами (сети хостов рвут долгие скачивания)
cat > /workspace/dl_vox.py <<'PY'
import time
from huggingface_hub import snapshot_download
for attempt in range(12):
    try:
        snapshot_download('openbmb/VoxCPM2')
        print('VOX_DONE')
        break
    except Exception as e:
        print(f'attempt {attempt}: {type(e).__name__}: {e}')
        time.sleep(5)
else:
    raise SystemExit(1)
PY
HF_HOME=/workspace/hf /workspace/vox/bin/python /workspace/dl_vox.py \
  > /workspace/voxdl.log 2>&1
```

- [ ] **Шаг 6.2: runner.sh** — три правки:

Строка 5 (grep-список env) — добавить новые переменные:

```bash
export $(tr '\0' '\n' < /proc/1/environ | grep -E '^(NOVA_MOCK|NOVA_TOKEN|NOVA_TTS|NOVA_FISH_CKPT|NOVA_FISH_KEY|NOVA_FISH_REF_ID|NOVA_FISH_TEMP|NOVA_FISH_TOP_P|NOVA_MODEL|NOVA_IDLE_LIMIT|NOVA_EYES|GEMINI_KEY|NOVA_GEMINI_MODEL|NOVA_VOX_TAG|NOVA_VOX_STRESS|NOVA_VOX_SEED|NOVA_GPU_UTIL|VAST_API_KEY|VAST_CONTAINERLABEL|HF_TOKEN)=' | tr '\n' ' ')
```

После строки 14 (fish_key) — ключ глаз:

```bash
[ -f /workspace/gemini_key ] && export GEMINI_KEY=$(cat /workspace/gemini_key)
```

Блок vLLM (строки 25–37) — утилизация зависит от того, где глаза:

```bash
  # облачные глаза: кадры в мозг не ходят, KV-кэшу хватает 0.70 —
  # высвобождаем ~7ГБ под VoxCPM2. Локальные глаза — прежние 0.85.
  if [ "${NOVA_EYES:-gemini}" = "gemini" ] && [ -n "$GEMINI_KEY" ]; then
    export NOVA_GPU_UTIL=${NOVA_GPU_UTIL:-0.70}
  else
    export NOVA_GPU_UTIL=${NOVA_GPU_UTIL:-0.85}
  fi
  export NOVA_IMG_LIMIT=${NOVA_IMG_LIMIT:-6}
  nohup vllm serve "$NOVA_MODEL" \
    --host 127.0.0.1 --port 5000 --max-model-len 32768 \
    --gpu-memory-utilization "$NOVA_GPU_UTIL" \
    --limit-mm-per-prompt "{\"image\":$NOVA_IMG_LIMIT}" \
    --enforce-eager \
    > /workspace/vllm.log 2>&1 &
```

Строка 72 (запуск сервера) — из vox-venv, если он есть (там voxcpm и
ruaccent; venv видит системные пакеты):

```bash
PYBIN=python3
[ -x /workspace/vox/bin/python ] && PYBIN=/workspace/vox/bin/python
nohup "$PYBIN" -m nova.server.main > /workspace/nova.log 2>&1 &
```

- [ ] **Шаг 6.3: тест vast (падает)** — в `tests/test_vast.py` добавить:

```python
def test_create_env_passes_eyes_and_voice(monkeypatch):
    import scripts.vast as vast

    sent = {}

    def fake_put(url, headers=None, json=None, timeout=None):
        sent.update(json["env"])

        class R:
            def raise_for_status(self):
                pass

        return R()

    monkeypatch.setattr(vast.httpx, "put", fake_put)
    vast.create_instance("key", 1, "token", env={
        "GEMINI_KEY": "gk", "NOVA_TTS": "voxcpm", "HF_TOKEN": "hf",
    })
    assert sent["GEMINI_KEY"] == "gk"
    assert sent["NOVA_EYES"] == "gemini"
    assert sent["NOVA_TTS"] == "voxcpm"
```

- [ ] **Шаг 6.4:** `uv run pytest tests/test_vast.py -q` → FAIL (нет
  GEMINI_KEY в env).

- [ ] **Шаг 6.5: vast.py** — в create_instance env-блок (строки 94–108)
  добавить/поменять строки:

```python
            "NOVA_TTS": env.get("NOVA_TTS", "voxcpm"),
            "NOVA_EYES": env.get("NOVA_EYES", "gemini"),
            "GEMINI_KEY": env.get("GEMINI_KEY", ""),
```

(строка NOVA_TTS уже есть — у неё меняется дефолт fishcloud → voxcpm;
fish-переменные не трогаем: откат = NOVA_TTS=fishcloud в .env).

- [ ] **Шаг 6.6:** `uv run pytest -q` → все зелёные.

- [ ] **Шаг 6.7:**

```bash
git add deploy/onstart.sh deploy/runner.sh scripts/vast.py tests/test_vast.py
git commit -m "feat: deploy voice 3.0 - vox venv, gpu util by eyes mode, env"
```

---

### Задача 7: merge, деплой, живая приёмка (с Джеем)

- [ ] **Шаг 7.1:** `uv run pytest -q` → всё зелёное; merge:

```bash
git checkout master && git merge eyes-voice3 && git push
```

- [ ] **Шаг 7.2:** на инстансе: `echo <ключ> > /workspace/gemini_key`,
  затем `bash /workspace/NOVA/deploy/onstart.sh` (дольёт vox-venv и веса),
  проверить `grep -E 'VOX_DONE' /workspace/voxdl.log`.
  ВНИМАНИЕ: vLLM после смены util надо перезапустить:
  `pkill -9 -f 'vllm [s]erve'` перед runner.sh.

- [ ] **Шаг 7.3:** прогрев и smoke: `curl localhost:8000/health`;
  в nova.log нет «RUAccent не поднялся» (если есть — применить
  monkeypatch token_type_ids из STATUS, раздел F5-эпопеи);
  `nvidia-smi` — суммарно ≤ 45ГБ.

- [ ] **Шаг 7.4: приёмка по спеке (уши Джея):**
  1. голос VoxCPM2: темп slower, без шапки, без слоу-мо, заскоков ≤ облака;
  2. пауза до первого звука ≤ облака D (3–6с);
  3. комментарии по экрану через Gemini не хуже прежних (игра/код/ролик);
  4. длинный диалог без OOM/провалов KV;
  5. откаты: NOVA_TTS=fishcloud и NOVA_EYES=local возвращают старое
     поведение одной переменной.

- [ ] **Шаг 7.5:** STATUS.md — новый раздел «Голос 3.0 + глаза в облако:
  ПРОД» с фактическими замерами (латентность, nvidia-smi, расход Gemini
  за день) и коммит `docs: voice 3.0 and cloud eyes live`.
