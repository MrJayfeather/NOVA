import asyncio
import os
import re
from pathlib import Path
from typing import AsyncIterator

import numpy as np

from nova.server.models.base import TTSModel
from nova.server.models.xtts_tts import split_for_tts
from nova.server.tts_text import strip_markers

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
    for w, t in words:
        nw = norm_word(w)
        if not nw:
            continue
        for target in targets:
            # 0.65: «слушои»/«слушай» (две ослышки) проходит, английская
            # тарабарщина тега (~0.3 к русским словам) — нет
            if nw == target or difflib.SequenceMatcher(
                    None, nw, target).ratio() >= 0.65:
                start = max(0.0, t - margin)
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


def find_silence_cut(pcm: np.ndarray, rate: int, search_s: float = 6.0,
                     min_gap_s: float = 0.25, margin: float = 0.05,
                     ) -> float | None:
    """Запасной срез: после шапки модель делает паузу перед репликой —
    ищем самый длинный тихий провал в начале аудио и режем по его концу."""
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
    best_len, best_end, run = 0, None, 0
    for i, q in enumerate(quiet):
        run = run + 1 if q else 0
        if run > best_len:
            best_len, best_end = run, i + 1
    if best_len * win < min_gap_s * rate or best_end is None:
        return None
    return max(0.0, best_end * win / rate - margin)


def normalize_peak(pcm: np.ndarray, target: int = 23000) -> np.ndarray:
    """Пик к ~70% шкалы — как у остальных движков (см. fish_tts)."""
    peak = float(np.abs(pcm).max()) if len(pcm) else 0.0
    if 0 < peak < target:
        pcm = (pcm.astype(np.float32) * (target / peak)).astype(np.int16)
    return pcm


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

    def _gen_sync(self, prepared: str, seed: int) -> np.ndarray:
        import torch

        torch.manual_seed(seed)
        wav = self._model.generate(
            text=prepared, prompt_wav_path=self._ref_wav,
            prompt_text=self._ref_text, reference_wav_path=self._ref_wav)
        arr = np.asarray(wav, dtype=np.float32) * 32767.0
        return arr.clip(-32768, 32767).astype(np.int16)

    async def _sentence_pcm(self, sentence: str, seed: int) -> bytes:
        pcm = await asyncio.to_thread(self._gen_sync, self.prepare(sentence), seed)
        if self._tag and self._timestamps:
            words = await self._timestamps(pcm.tobytes(), self.sample_rate)
            first = strip_markers(sentence).split()[:3]
            if first:
                pcm, cut = cut_spoken_head(pcm, self.sample_rate, words, first)
                if cut is None:
                    # по словам не нашли — запасной срез по паузе после шапки
                    cut = find_silence_cut(pcm, self.sample_rate)
                    if cut is None:
                        # английская шапка НЕ выходит в эфир никогда
                        raise HeadCutMiss(sentence[:40])
                    pcm = pcm[int(cut * self.sample_rate):]
                    print(f"[nova] vox-tts: срез по паузе {cut:.2f}с: {sentence[:40]!r}")
                else:
                    print(f"[nova] vox-tts: срез {cut:.2f}с: {sentence[:40]!r}")
        return normalize_peak(pcm).tobytes()

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

    async def synthesize(self, text: str) -> AsyncIterator[bytes]:
        async with self._lock:
            if self._model is None:
                await asyncio.to_thread(self._load_sync)
        # одна GPU-модель — последовательно (в отличие от облака fish)
        for sentence in split_for_tts(strip_markers(text))[:20]:
            try:
                try:
                    pcm = await self._sentence_pcm(sentence, self._seed)
                except HeadCutMiss:
                    # другой seed — другая генерация, шапка обычно находится
                    print(f"[nova] vox-tts: шапка не найдена, пересинтез: {sentence[:50]!r}")
                    pcm = await self._sentence_pcm(sentence, self._seed + 1)
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
