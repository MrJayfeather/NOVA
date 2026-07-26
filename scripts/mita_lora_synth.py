"""Свип сэмплов по чекпоинтам LoRA Миты (запускается на боксе в vox-venv).

Хот-своп LoRA в одном процессе: для каждого чекпоинта — «привет» без
референса (голос целиком из LoRA) и с прод-референсом voice_sample_vox;
для указанных в DEEP шагов — ещё длинная фраза. Рецепт прода: без тега,
ударения U+0301, seed 42. Выход: /workspace/samples_lora/mlora_*.wav

Запуск: /workspace/vox/bin/python /workspace/mita_lora_synth.py
"""

import os
import time
from pathlib import Path

import soundfile as sf
import torch
from voxcpm import VoxCPM

CKPT_DIR = Path("/workspace/ckpts_mita_lora")
OUT = Path("/workspace/samples_lora")
REF = "/workspace/NOVA/personas/nova/voice_sample_vox.wav"
REF_TEXT = Path("/workspace/NOVA/personas/nova/voice_sample_vox.txt").read_text(
    encoding="utf-8").strip()
SEED = 42
DEEP = {100, 200, 400}  # этим шагам — ещё и длинная фраза

PHRASES = {
    "privet": "Привет, Джей! Ну как я тебе теперь? По-моему, звучит вполне живенько.",
    "dlinnaya": ("Слушай, я тут подумала: если ты снова застрянешь в этой игре "
                 "на три часа, я начну комментировать каждый твой про́мах. "
                 "Шучу. Или нет?"),
}


def main() -> None:
    ckpts = sorted(CKPT_DIR.glob("step_*"))
    assert ckpts, "нет чекпоинтов"
    OUT.mkdir(exist_ok=True)
    print(f"чекпоинтов: {len(ckpts)}", flush=True)

    model = VoxCPM.from_pretrained(
        "openbmb/VoxCPM2",
        load_denoiser=False,
        lora_weights_path=str(ckpts[0]),
    )
    sr = model.tts_model.sample_rate

    for ck in ckpts:
        step = int(ck.name.split("_")[1])
        model.load_lora(str(ck))
        jobs = [("privet", PHRASES["privet"])]
        if step in DEEP:
            jobs.append(("dlinnaya", PHRASES["dlinnaya"]))
        for name, text in jobs:
            for variant in ("noref", "ref"):
                t0 = time.time()
                torch.manual_seed(SEED)
                kwargs = {"text": text}
                if variant == "ref":
                    kwargs.update(prompt_wav_path=REF, prompt_text=REF_TEXT)
                wav = model.generate(**kwargs)
                out = OUT / f"mlora_s{step:03d}_{variant}_{name}.wav"
                sf.write(str(out), wav, sr)
                print(f"{out.name}: {len(wav)/sr:.1f}с за {time.time()-t0:.0f}с",
                      flush=True)
    print("SYNTH_DONE", flush=True)


if __name__ == "__main__":
    main()
