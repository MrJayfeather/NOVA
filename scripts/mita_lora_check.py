"""Автопроверка чекпоинтов LoRA Миты: стоп-токен + верность тексту.

Для каждого чекпоинта (хот-своп) синтезирует 2 контрольные фразы с
прод-референсом и проверяет два симптома перезубривания:
  - длительность: раздутый хвост (>2.5х медианы по чекпоинтам) = потеря
    стоп-токена;
  - whisper-сверка: доля совпавших слов с исходным текстом <0.75 = модель
    жуёт/заменяет слова.
Пишет /workspace/lora2_report.csv и сохраняет все wav в
/workspace/samples_lora3/ (они же — материал на отслушку).

Запуск: /workspace/vox/bin/python /workspace/mita_lora_check.py [ckpt_dir]
"""

import csv
import difflib
import re
import sys
import time
from pathlib import Path

import soundfile as sf
import torch
from faster_whisper import WhisperModel
from voxcpm import VoxCPM

CKPT_DIR = Path(sys.argv[1] if len(sys.argv) > 1 else "/workspace/ckpts_mita_lora2")
OUT = Path("/workspace/samples_lora3")
REPORT = Path("/workspace/lora2_report.csv")
REF = "/workspace/NOVA/personas/nova/voice_sample_vox.wav"
REF_TEXT = Path("/workspace/NOVA/personas/nova/voice_sample_vox.txt").read_text(
    encoding="utf-8").strip()
SEED = 42

PHRASES = {
    "vopros": ("Погоди, а ты вообще ел сегодня? Или опять весь день "
               "на кофе и упрямстве?"),
    "dlinnaya2": ("Знаешь, я тут пересматривала наши старые разговоры и "
                  "поняла забавную вещь: ты споришь со мной ровно до тех пор, "
                  "пока я не окажусь права. А потом делаешь вид, "
                  "что так и задумано."),
}


def words(s: str) -> list[str]:
    return [re.sub(r"[^\wё]", "", w.lower()).replace("ё", "е")
            for w in s.split() if re.sub(r"[^\wё]", "", w)]


def match_ratio(expected: str, heard: str) -> float:
    return difflib.SequenceMatcher(None, words(expected), words(heard)).ratio()


def main() -> None:
    ckpts = sorted(CKPT_DIR.glob("step_*"))
    assert ckpts, f"нет чекпоинтов в {CKPT_DIR}"
    OUT.mkdir(exist_ok=True)
    print(f"чекпоинтов: {len(ckpts)}", flush=True)

    model = VoxCPM.from_pretrained(
        "openbmb/VoxCPM2", load_denoiser=False,
        lora_weights_path=str(ckpts[0]),
    )
    sr = model.tts_model.sample_rate
    asr = WhisperModel("small", device="cuda", compute_type="float16")

    rows = []
    for ck in ckpts:
        step = int(ck.name.split("_")[1])
        model.load_lora(str(ck))
        for name, text in PHRASES.items():
            t0 = time.time()
            torch.manual_seed(SEED)
            wav = model.generate(text=text, prompt_wav_path=REF,
                                 prompt_text=REF_TEXT)
            dur = len(wav) / sr
            path = OUT / f"mlora3_s{step:03d}_{name}.wav"
            sf.write(str(path), wav, sr)
            segs, _ = asr.transcribe(str(path), language="ru", beam_size=5)
            heard = " ".join(s.text.strip() for s in segs).strip()
            ratio = match_ratio(text, heard)
            rows.append({"step": step, "phrase": name,
                         "dur_s": round(dur, 2), "match": round(ratio, 3),
                         "heard": heard})
            print(f"s{step:03d} {name}: {dur:.1f}с match={ratio:.2f} "
                  f"({time.time()-t0:.0f}с)", flush=True)

    # вердикты: длительность против медианы ЭТОЙ фразы по всем чекпоинтам
    for name in PHRASES:
        durs = sorted(r["dur_s"] for r in rows if r["phrase"] == name)
        med = durs[len(durs) // 2]
        for r in rows:
            if r["phrase"] == name:
                bloat = r["dur_s"] > 2.5 * med
                garble = r["match"] < 0.75
                r["verdict"] = "FAIL" if (bloat or garble) else "ok"
                if bloat:
                    r["verdict"] += "+хвост"
                if garble:
                    r["verdict"] += "+жуёт"

    with REPORT.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    bad = [r for r in rows if r["verdict"] != "ok"]
    print(f"итог: {len(rows)} проверок, провалов {len(bad)}", flush=True)
    for r in bad:
        print(f"  FAIL s{r['step']:03d} {r['phrase']}: dur={r['dur_s']} "
              f"match={r['match']}", flush=True)
    print("CHECK_DONE", flush=True)


if __name__ == "__main__":
    main()
