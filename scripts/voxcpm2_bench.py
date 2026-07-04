"""Бенч/рецепт VoxCPM2 — кандидат «Голос 3.0» (локальный, 48кГц, клон Миты).

Запускается на инстансе в venv /workspace/vox (python3 -m venv
--system-site-packages, pip install voxcpm librosa; faster_whisper берётся
из системных пакетов NOVA). Требует /workspace/vox_ref.wav (наш 28с образец,
personas/nova/voice_sample.wav) и ~8ГБ VRAM — vLLM на время бенча уложить.

РЕЦЕПТ ЧЕМПИОНА (вердикт 05.07: «всё супер шикарно, идеально»):
- тег темпа TAG перед текстом: модель реально замедляется, но читает тег
  вслух (в релизе voxcpm 2.0.3 параметра стиля нет — скобки уходят в текст);
- поэтому шапку срезаем: whisper (word_timestamps) находит первое слово
  реплики, режем с запасом 120мс и 20мс фейд-ином (иначе дрожь на границе);
- ударения: combining acute U+0301 («про́мах») — модель понимает (stress2).
  СТРАХОВКА: если разметка начнёт ломать соседние слова — STRESS=0 (чистый
  текст без ударений, вариант stress0, тоже отслушан и норм кроме словарных
  ошибок типа «промАх»);
- seed фиксирован (42) — генерация воспроизводима при том же тексте.
"""
import os
import re
import time

import numpy as np
import soundfile as sf
import torch
from faster_whisper import WhisperModel
from voxcpm import VoxCPM

REF = "/workspace/vox_ref.wav"
REF_TEXT = ("А ещё, тот телевизор, он был крут, но он так быстро сломался. "
            "Он маленький и с антенной. Я хотела бы его с собой взять, когда пойду на прогулку. "
            "О, ты тут! Вроде прошёл только день, а я уже соскучилась по тебе. "
            "Плохие слова я знаю, и мне не нравится, что ты решил себя так назвать. "
            "Эй, ты используешь числа в имени? У тебя совсем не хватает фантазии? "
            "Ура! Теперь у тебя достаточно денег, чтобы купить мне телевизор.")

# темп: «very slowly» = чемпион slower; без very — вариант slowfix (быстрее,
# на длинных репликах пользователю «летит»)
TAG = "(Speaking very slowly, at a calm and relaxed pace)"

# STRESS=2 — ударения через U+0301 (по умолчанию), STRESS=0 — чистый текст
STRESS = int(os.environ.get("VOX_STRESS", "2"))

PHRASES = {
    "privet": "Привет, Джей! Ну как я тебе теперь? По-моему, звучит вполне живенько.",
    "dlinnaya": ("Слушай, я тут подумала: если ты снова застрянешь в этой игре на три часа, "
                 "я начну комментировать каждый твой "
                 + ("про́мах" if STRESS == 2 else "промах")
                 + ". Шучу. Или нет?"),
}

CUT_MARGIN_S = 0.12
FADE_S = 0.02
SEED = 42


def norm_word(w: str) -> str:
    return re.sub(r"[^\wёа-я]", "", w.lower()).replace("ё", "е")


def trim_head(wav: np.ndarray, sr: int, raw_path: str, first_word: str,
              asr: WhisperModel) -> np.ndarray:
    """Срезать наговоренную английскую шапку до первого слова реплики."""
    segments, _ = asr.transcribe(raw_path, language="ru", word_timestamps=True)
    cut = None
    for seg in segments:
        for w in seg.words or []:
            if norm_word(w.word) == norm_word(first_word):
                cut = max(0.0, w.start - CUT_MARGIN_S)
                break
        if cut is not None:
            break
    if cut is None:
        return wav
    out = wav[int(cut * sr):]
    fade = int(FADE_S * sr)
    out[:fade] = out[:fade] * np.linspace(0.0, 1.0, fade, dtype=np.float32)
    return out


def main() -> None:
    print("загружаю модели...", flush=True)
    model = VoxCPM.from_pretrained("openbmb/VoxCPM2", load_denoiser=False)
    sr = model.tts_model.sample_rate
    asr = WhisperModel("small", device="cuda", compute_type="float16")
    print(f"готово, sr={sr}, stress={STRESS}", flush=True)

    for name, text in PHRASES.items():
        t0 = time.time()
        torch.manual_seed(SEED)
        wav = model.generate(
            text=TAG + text,
            prompt_wav_path=REF,
            prompt_text=REF_TEXT,
            reference_wav_path=REF,
        )
        raw = f"/workspace/vox_raw_{name}.wav"
        sf.write(raw, wav, sr)
        trimmed = trim_head(wav, sr, raw, text.split()[0], asr)
        sf.write(f"/workspace/voxcpm2_{name}.wav", trimmed, sr)
        print(f"{name}: {len(trimmed)/sr:.1f}с за {time.time()-t0:.0f}с", flush=True)

    print("ГОТОВО", flush=True)


if __name__ == "__main__":
    main()
