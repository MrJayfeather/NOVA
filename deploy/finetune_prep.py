"""Подготовка датасета для дообучения S1-mini: ogg -> wav 44.1k моно + .lab.

Транскрипция faster-whisper large-v3-turbo на GPU. Пустые распознавания
(вздохи, междометия без слов) отбрасываются.

Запуск на инстансе:
  python3 deploy/finetune_prep.py /workspace/mita_raw /workspace/mita_data/mita
"""

import sys
import wave
from pathlib import Path

import numpy as np
from faster_whisper import WhisperModel, decode_audio


def main() -> None:
    src = Path(sys.argv[1] if len(sys.argv) > 1 else "/workspace/mita_raw")
    dst = Path(sys.argv[2] if len(sys.argv) > 2 else "/workspace/mita_data/mita")
    dst.mkdir(parents=True, exist_ok=True)

    files = sorted(src.rglob("*.ogg")) + sorted(src.rglob("*.wav"))
    print(f"клипов: {len(files)}")
    model = WhisperModel("large-v3-turbo", device="cuda", compute_type="int8_float16")

    kept = skipped = 0
    for i, p in enumerate(files):
        audio44 = decode_audio(str(p), sampling_rate=44100)
        audio16 = decode_audio(str(p), sampling_rate=16000)
        segments, _ = model.transcribe(audio16, language="ru", beam_size=5)
        text = " ".join(s.text.strip() for s in segments).strip()
        if len(text) < 2:
            skipped += 1
            continue
        pcm = (np.clip(audio44, -1.0, 1.0) * 32767).astype("<i2")
        name = f"{i:04d}"
        with wave.open(str(dst / f"{name}.wav"), "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(44100)
            w.writeframes(pcm.tobytes())
        (dst / f"{name}.lab").write_text(text, encoding="utf-8")
        kept += 1
        if (i + 1) % 50 == 0:
            print(f"  ... {i + 1}/{len(files)}")

    print(f"готово: {kept}, пропущено пустых: {skipped}")
    print("PREP_DONE")


if __name__ == "__main__":
    main()
