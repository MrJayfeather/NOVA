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
