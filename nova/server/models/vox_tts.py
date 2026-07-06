import asyncio
import os
import re
from pathlib import Path
from typing import AsyncIterator

import numpy as np

from nova.server.models.base import TTSModel
from nova.server.models.xtts_tts import split_for_tts
from nova.server.tts_text import speech_matches, strip_markers

# Рецепт-чемпион 2.0 (отслушан 05.07, «r3p идеально»): БЕЗ стилевого тега.
# Темп задаёт референс (медленные мягкие + бодрые тёплые реплики) +
# растяжка пауз. Тег оставлен как выключенная опция: наговаривал шапку.
DEFAULT_TAG = ""

# Эмоции БЕЗ тегов: маркер мозга -> свой референс (отслушано: «супер,
# шикарно»). Ключи — группы маркеров персоны, значения — имя референса.
MARKER_EMOTIONS = {
    "laughing": "joy", "chuckling": "joy", "excited": "joy",
    "surprised": "joy",
    "sarcastic": "tease", "curious": "tease",
    "whispering": "soft", "soft tone": "soft", "sighing": "soft",
}


def emotion_for(sentence: str) -> str | None:
    """Первый известный маркер предложения -> имя эмо-референса."""
    for m in re.findall(r"\[([a-z ]+)\]", sentence.lower()):
        if m in MARKER_EMOTIONS:
            return MARKER_EMOTIONS[m]
    return None

_VOWELS = "аеёиоуыэюяАЕЁИОУЫЭЮЯ"


def stress_to_acute(text: str) -> str:
    """RUAccent ставит «+» перед ударной гласной («пр+омах»); VoxCPM2
    проверенно понимает combining acute ПОСЛЕ неё («про́мах»)."""
    return re.sub(rf"\+([{_VOWELS}])", "\\1́", text)


# Слова, где модель врёт ударение, — с явным знаком U+0301 (тот самый
# «символ ударения», отслушанный в бенче stress2). Пополняется по мере
# отлова вживую; авто-RUAccent в рецепт-чемпион НЕ входит.
STRESS_FIXES = {
    "присмотром": "присмо́тром",
}
_FIXES_RE = re.compile(
    r"\b(" + "|".join(STRESS_FIXES) + r")\b", re.IGNORECASE)


def apply_stress_fixes(text: str) -> str:
    """Точечные ударения по словарику, регистр первой буквы сохраняем."""
    def repl(m):
        w = m.group(0)
        fix = STRESS_FIXES[w.lower()]
        return fix[0].upper() + fix[1:] if w[0].isupper() else fix

    return _FIXES_RE.sub(repl, text)


def norm_word(w: str) -> str:
    return re.sub(r"[^\wёа-яЁА-Я]", "", w.lower()).replace("ё", "е")


def cut_spoken_head(pcm: np.ndarray, rate: int,
                    words: list[tuple[str, float]],
                    first_words: list[str] | str,
                    margin: float = 0.12, fade: float = 0.02,
                    ) -> tuple[np.ndarray, float | None]:
    """Модель наговаривает стилевой тег в начале — режем всё до начала
    реплики. Whisper слышит первое слово не всегда точно («Слушай» →
    «слушои»), поэтому сравнение нечёткое и по нескольким первым словам.
    Возвращает (аудио, секунда среза | None)."""
    import difflib

    if isinstance(first_words, str):
        first_words = [first_words]
    targets = [t for t in (norm_word(w) for w in first_words) if t]
    start = None
    for i, (w, _) in enumerate(words):
        nw = norm_word(w)
        if not nw:
            continue
        for k, target in enumerate(targets):
            # 0.65: «слушои»/«слушай» (две ослышки) проходит, английская
            # тарабарщина тега (~0.3 к русским словам) — нет
            if nw == target or difflib.SequenceMatcher(
                    None, nw, target).ratio() >= 0.65:
                # совпало k-е слово реплики (первое whisper мог слить с
                # шапкой) — отступаем на k слов назад, начало не съедаем
                start = max(0.0, words[max(0, i - k)][1] - margin)
                break
        if start is not None:
            break
    if start is None:
        return pcm, None
    out = pcm[int(start * rate):].copy()
    n = min(int(fade * rate), len(out))
    if n:
        ramp = np.linspace(0.0, 1.0, n, dtype=np.float32)
        out[:n] = (out[:n].astype(np.float32) * ramp).astype(np.int16)
    return out, start


class HeadCutMiss(Exception):
    """Наговоренная шапка не найдена — предложение нельзя выпускать в эфир."""


class GuardMiss(Exception):
    """Сверка с текстом провалена (робо-заскок); аудио приложено."""

    def __init__(self, reason: str, pcm: bytes):
        super().__init__(reason)
        self.pcm = pcm


def find_silence_cut(pcm: np.ndarray, rate: int, search_s: float = 4.5,
                     min_gap_s: float = 0.25, margin: float = 0.05,
                     min_head_s: float = 1.0) -> float | None:
    """Запасной срез: после шапки модель делает вдох-паузу — режем по
    ПЕРВОМУ достаточному провалу тишины после min_head_s (дальние паузы
    могут быть уже внутри реплики — их не трогаем)."""
    n = min(len(pcm), int(search_s * rate))
    if n < rate // 2:
        return None
    win = max(1, int(0.03 * rate))
    head = np.abs(pcm[:n].astype(np.float32))
    peak = float(head.max())
    if peak <= 0:
        return None
    # огибающая по окнам 30мс: тихо = ниже 5% пика
    frames = head[: (n // win) * win].reshape(-1, win).max(axis=1)
    quiet = frames < 0.05 * peak
    run = 0
    for i, q in enumerate(quiet):
        if q:
            run += 1
            continue
        if (run * win >= min_gap_s * rate
                and (i - run) * win >= min_head_s * rate):
            return max(0.0, i * win / rate - margin)
        run = 0
    return None


def normalize_peak(pcm: np.ndarray, target: int = 23000) -> np.ndarray:
    """Пик к ~70% шкалы — как у остальных движков (см. fish_tts)."""
    peak = float(np.abs(pcm).max()) if len(pcm) else 0.0
    if 0 < peak < target:
        pcm = (pcm.astype(np.float32) * (target / peak)).astype(np.int16)
    return pcm


def _load_dfn():
    """DeepFilterNet-полировка выхода. df 0.5.6 импортирует выпиленный
    torchaudio.backend — подсовываем шим до импорта."""
    import sys
    import types

    import torchaudio

    shim = types.ModuleType("torchaudio.backend.common")
    shim.AudioMetaData = getattr(torchaudio, "AudioMetaData", object)
    backend = types.ModuleType("torchaudio.backend")
    backend.common = shim
    sys.modules.setdefault("torchaudio.backend", backend)
    sys.modules["torchaudio.backend.common"] = shim

    import torch
    from df.enhance import enhance, init_df

    model, state, _ = init_df()

    def run(arr: np.ndarray) -> np.ndarray:  # float32 [-1..1], 48кГц
        t = torch.from_numpy(arr).unsqueeze(0)
        return enhance(model, state, t).squeeze(0).numpy()

    return run


def stretch_pauses(pcm: np.ndarray, rate: int, factor: float = 2.0,
                   min_gap_s: float = 0.12) -> np.ndarray:
    """Замедление без артефактов: удлиняем ТОЛЬКО тишину между словами,
    сами слова не трогаем вообще (вердикт r3p: «идеально»)."""
    if factor <= 1.0 or not len(pcm):
        return pcm
    amp = np.abs(pcm.astype(np.float32))
    peak = float(amp.max())
    if peak <= 0:
        return pcm
    win = max(1, int(0.03 * rate))
    frames = amp[: (len(amp) // win) * win].reshape(-1, win).max(axis=1)
    quiet = frames < 0.04 * peak
    out = []
    i = 0
    while i < len(quiet):
        j = i
        while j < len(quiet) and quiet[j] == quiet[i]:
            j += 1
        seg = pcm[i * win: j * win]
        out.append(seg)
        # крайние тишины не трогаем — только вдохи между словами
        if quiet[i] and (j - i) * win >= min_gap_s * rate and i > 0 and j < len(quiet):
            out.append(np.zeros(int(len(seg) * (factor - 1)), dtype=pcm.dtype))
        i = j
    out.append(pcm[len(quiet) * win:])
    return np.concatenate(out)


class VoxTTS(TTSModel):
    """Локальный голос 3.0: VoxCPM2 (48кГц) по рецепту-чемпиону — тег
    темпа, ударения RUAccent (U+0301), срез наговоренной шапки."""

    sample_rate = 48000

    def __init__(self, reference_wav: Path, reference_text: str,
                 tag: str = DEFAULT_TAG, stress: bool = False, seed: int = 42,
                 word_timestamps=None, check_speech: bool = True,
                 pause_factor: float = 2.0, use_dfn: bool = True,
                 emo_refs: dict | None = None):
        # word_timestamps(pcm, rate) -> [(слово, старт_с)] — один прогон
        # whisper на предложение: и срез шапки, и СТТ-страж по нему же
        self._ref_wav = str(reference_wav)
        self._ref_text = reference_text
        self._tag = tag
        self._stress = stress
        self._seed = seed
        self._timestamps = word_timestamps
        self._check_speech = check_speech
        self._pause_factor = pause_factor
        self._use_dfn = use_dfn
        self._dfn = None
        # имя эмоции -> (путь wav, транскрипт): [laughing] бодрит референс
        self._emo_refs = emo_refs or {}
        self._model = None
        self._accents = None
        self._lock = asyncio.Lock()
        self._gen_lock = asyncio.Lock()  # GPU-генерация строго по одному

    def _load_sync(self) -> None:
        # DFN тянет numpy 1.26, где выпилены старые алиасы, а зависимости
        # voxcpm/ruaccent местами зовут np.long и др. — возвращаем шимом
        for old, new in (("long", np.int64), ("ulong", np.uint64),
                         ("uint", np.uint64), ("int", int), ("float", float),
                         ("bool", bool), ("object", object),
                         ("unicode", str), ("str", str), ("complex", complex)):
            if not hasattr(np, old):
                setattr(np, old, new)
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
        if self._use_dfn:
            try:
                self._dfn = _load_dfn()
            except Exception as exc:
                # голос важнее полировки
                print(f"[nova] DFN не поднялся, полировка выключена: {exc!r}")

    def prepare(self, sentence: str) -> str:
        s = apply_stress_fixes(strip_markers(sentence))
        if self._accents is not None:
            s = stress_to_acute(self._accents.process_all(s))
        return self._tag + s if self._tag else s

    def _gen_sync(self, prepared: str, seed: int,
                  ref_wav: str | None = None,
                  ref_text: str | None = None) -> np.ndarray:
        import torch

        torch.manual_seed(seed)
        wav = self._model.generate(
            text=prepared,
            prompt_wav_path=ref_wav or self._ref_wav,
            prompt_text=ref_text or self._ref_text,
            reference_wav_path=ref_wav or self._ref_wav)
        arr = np.asarray(wav, dtype=np.float32) * 32767.0
        return arr.clip(-32768, 32767).astype(np.int16)

    async def _sentence_pcm(self, sentence: str, seed: int) -> bytes:
        prepared = self.prepare(sentence)
        # эмоция предложения выбирает референс (маркер мозга -> голос)
        emo = emotion_for(sentence)
        ref_wav, ref_text = self._emo_refs.get(
            emo, (self._ref_wav, self._ref_text))
        async with self._gen_lock:
            pcm = await asyncio.to_thread(self._gen_sync, prepared, seed,
                                          ref_wav, ref_text)
        cut = 0.0
        words: list[tuple[str, float]] = []
        need_whisper = self._timestamps and (self._tag or self._check_speech)
        if need_whisper:
            words = await self._timestamps(pcm.tobytes(), self.sample_rate)
        if self._tag and self._timestamps:
            first = strip_markers(sentence).split()[:3]
            if first:
                trimmed, found = cut_spoken_head(
                    pcm, self.sample_rate, words, first)
                if found is None:
                    # по словам не нашли — запасной срез по паузе после шапки
                    found = find_silence_cut(pcm, self.sample_rate)
                    if found is None:
                        # английская шапка НЕ выходит в эфир никогда
                        raise HeadCutMiss(sentence[:40])
                    trimmed = pcm[int(found * self.sample_rate):]
                    print(f"[nova] vox-tts: срез по паузе {found:.2f}с: {sentence[:40]!r}")
                else:
                    print(f"[nova] vox-tts: срез {found:.2f}с: {sentence[:40]!r}")
                pcm, cut = trimmed, found
        if self._pause_factor > 1.0:
            pcm = stretch_pauses(pcm, self.sample_rate, self._pause_factor)
        if self._dfn is not None and self.sample_rate == 48000:
            f = pcm.astype(np.float32) / 32768.0
            f = await asyncio.to_thread(self._dfn, f)
            pcm = (f * 32767.0).clip(-32768, 32767).astype(np.int16)
        out = normalize_peak(pcm).tobytes()
        if self._check_speech and need_whisper:
            # страж по ТОМУ ЖЕ прогону whisper: слова после среза
            heard = " ".join(w for w, t in words if t >= cut)
            if not speech_matches(strip_markers(sentence), heard):
                raise GuardMiss(sentence[:40], out)
        return out

    async def warmup(self) -> None:
        """VoxCPM2 грузится ~1.5 мин — греем при старте сервера, а не
        посреди первой реплики пользователя."""
        try:
            async with self._lock:
                if self._model is None:
                    await asyncio.to_thread(self._load_sync)
            await asyncio.to_thread(
                self._gen_sync, self.prepare("Привет."), self._seed)
            print("[nova] vox-tts: прогрет")
        except Exception as exc:
            print(f"[nova] vox-tts: прогрев не удался: {exc!r}")

    async def _sentence_checked(self, sentence: str) -> bytes | None:
        try:
            try:
                return await self._sentence_pcm(sentence, self._seed)
            except (HeadCutMiss, GuardMiss):
                # заскок/шапка детерминированы по (текст, seed): тот же seed
                # дал бы то же самое — пересинтез со сдвигом
                print(f"[nova] vox-tts: пересинтез (seed+1): {sentence[:50]!r}")
                try:
                    return await self._sentence_pcm(sentence, self._seed + 1)
                except GuardMiss as exc:
                    return exc.pcm  # дважды заскок — отдаём как есть (как fish)
        except Exception as exc:
            print(f"[nova] ошибка vox-tts (предложение пропущено): {exc!r}")
            return None

    async def synthesize(self, text: str) -> AsyncIterator[bytes]:
        async with self._lock:
            if self._model is None:
                await asyncio.to_thread(self._load_sync)
        # маркеры остаются в предложениях до выбора эмо-референса;
        # prepare() уберёт их перед синтезом
        sentences = split_for_tts(text)[:20]
        # конвейер внахлёст: GPU-генерация по одному (gen_lock), но пока
        # предложение идёт через whisper — следующее уже генерится
        tasks = [asyncio.create_task(self._sentence_checked(s))
                 for s in sentences]
        for task in tasks:
            pcm = await task
            if pcm:
                yield pcm


def build_vox_tts(asr, ref_dir: Path) -> VoxTTS:
    # у voxcpm свой референс (микс темпа r3): дефолтный voice_sample.wav
    # слишком бодрый — модель клонирует и спешку тоже
    ref = ref_dir / "voice_sample_vox.wav"
    ref_txt = ref_dir / "voice_sample_vox.txt"
    if not ref.exists():
        ref = ref_dir / "voice_sample.wav"
        ref_txt = ref_dir / "voice_sample.txt"
    emo_refs = {}
    for name in ("joy", "tease", "soft"):
        w = ref_dir / f"voice_{name}.wav"
        t = ref_dir / f"voice_{name}.txt"
        if w.exists() and t.exists():
            emo_refs[name] = (str(w), t.read_text(encoding="utf-8").strip())
    return VoxTTS(
        emo_refs=emo_refs,
        reference_wav=ref,
        reference_text=ref_txt.read_text(encoding="utf-8").strip(),
        tag=os.environ.get("NOVA_VOX_TAG", DEFAULT_TAG),
        # авто-RUAccent не в рецепте: идеал отслушан на чистом тексте,
        # точечные слова правит STRESS_FIXES; включение — NOVA_VOX_STRESS=1
        stress=os.environ.get("NOVA_VOX_STRESS", "0") == "1",
        seed=int(os.environ.get("NOVA_VOX_SEED", "42")),
        word_timestamps=asr.word_timestamps,
        check_speech=os.environ.get("NOVA_TTS_GUARD", "1") == "1",
        pause_factor=float(os.environ.get("NOVA_VOX_PAUSES", "2.0")),
        use_dfn=os.environ.get("NOVA_VOX_DFN", "1") == "1",
    )
