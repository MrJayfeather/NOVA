"""Сборка LoRA-датасета VoxCPM2 из кураторского отбора (final_manifest.csv).

Из ОРИГИНАЛЬНЫХ ogg (не 24кГц-копий):
  voices/_куратор/lora_data/wavs/*.wav — 16кГц моно, RMS-нормализация,
    обрезка тишины по краям (хвост <0.5с — требование конвейера VoxCPM2)
  voices/_куратор/lora_data/train.jsonl / val.jsonl — манифесты OpenBMB
    (пути уже целевые /workspace/mita_lora/wavs/; ~40% сэмплов получают
    ref_audio ДРУГОГО клипа ТОЙ ЖЕ эмоции — учим клон с эмо-референсов)
  voices/_куратор/lora_data/refs_emo/<эмоция>/*.ogg — пулы эмо-референсов
    (train=ref), копии оригиналов
  voices/mita_lora_data.tgz — всё вместе для заливки на инстанс

Правила: train=yes и длительность >=1.0с (гайд не велит короче);
val — 20 клипов стратифицированно по эмоциям; seed 42.

Запуск: uv run --with soundfile --with numpy python scripts/build_lora_dataset.py
"""

import csv
import json
import random
import shutil
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio.functional as AF

ROOT = Path(__file__).resolve().parent.parent
VOICES = ROOT / "voices"
MANIFEST = VOICES / "_куратор" / "final_manifest.csv"
OUT = VOICES / "_куратор" / "lora_data"
TGZ = VOICES / "mita_lora_data.tgz"
BOX_WAVS = "/workspace/mita_lora/wavs"

SR = 16000
TARGET_RMS_DB = -20.0
PEAK_CAP = 0.95
SIL_THR_DB = -45.0
TAIL_PAD_S = 0.25
HEAD_PAD_S = 0.15
VAL_N = 20
REF_FRAC = 0.4


def load_16k(path: Path) -> np.ndarray:
    wav, sr = sf.read(path, dtype="float32", always_2d=True)
    mono = wav.mean(axis=1)
    if sr != SR:
        mono = AF.resample(torch.from_numpy(mono), sr, SR).numpy()
    return mono


def trim_and_norm(x: np.ndarray) -> np.ndarray:
    win, hop = int(SR * 0.05), int(SR * 0.025)
    n = max(1, 1 + (len(x) - win) // hop)
    frames = np.lib.stride_tricks.as_strided(
        x, shape=(n, win), strides=(x.strides[0] * hop, x.strides[0])
    )
    rms = np.sqrt(np.mean(frames**2, axis=1))
    db = 20 * np.log10(rms + 1e-9)
    active = np.where(db > db.max() + SIL_THR_DB)[0]
    if len(active):
        start = max(0, int(active[0] * hop - HEAD_PAD_S * SR))
        end = min(len(x), int((active[-1] * hop + win) + TAIL_PAD_S * SR))
        x = x[start:end]
    cur_rms = np.sqrt(np.mean(x**2)) + 1e-9
    gain = 10 ** (TARGET_RMS_DB / 20) / cur_rms
    x = x * gain
    peak = np.abs(x).max()
    if peak > PEAK_CAP:
        x = x * (PEAK_CAP / peak)
    return x


def main() -> None:
    random.seed(42)
    if OUT.exists():
        shutil.rmtree(OUT)
    (OUT / "wavs").mkdir(parents=True)
    (OUT / "refs_emo").mkdir()

    rows = []
    with MANIFEST.open(encoding="utf-8-sig", newline="") as fh:
        rows = list(csv.DictReader(fh))

    train_rows, skipped_short, ref_rows = [], [], []
    for r in rows:
        dur = float(r["dur"])
        if r["train"] == "yes":
            if dur < 1.0:
                skipped_short.append(r["file"])
            else:
                train_rows.append(r)
        elif r["train"] == "ref":
            ref_rows.append(r)

    print(f"в обучение: {len(train_rows)}, короткие мимо: {len(skipped_short)}, "
          f"эмо-референсы: {len(ref_rows)}", flush=True)
    for f in skipped_short:
        print(f"  скип <1с: {f}", flush=True)

    total = 0.0
    for i, r in enumerate(train_rows, 1):
        if i % 100 == 0:
            print(f"  ... {i}/{len(train_rows)}", flush=True)
        x = trim_and_norm(load_16k(VOICES / r["path"]))
        r["_wav"] = Path(r["file"]).stem + ".wav"
        r["_dur16"] = len(x) / SR
        total += r["_dur16"]
        sf.write(OUT / "wavs" / r["_wav"], x, SR, subtype="PCM_16")
    print(f"обработано: {len(train_rows)} клипов, {total/60:.1f} мин", flush=True)

    # валидация: стратифицированно по эмоциям
    by_emo = defaultdict(list)
    for r in train_rows:
        by_emo[r["emotion"]].append(r)
    val = []
    for emo, group in sorted(by_emo.items()):
        k = max(1, round(VAL_N * len(group) / len(train_rows)))
        val.extend(random.sample(group, min(k, len(group))))
    val_names = {r["file"] for r in val}
    train = [r for r in train_rows if r["file"] not in val_names]
    print(f"train: {len(train)}, val: {len(val)}", flush=True)

    def jsonl_row(r, pool):
        item = {
            "audio": f"{BOX_WAVS}/{r['_wav']}",
            "text": r["text"],
            "duration": round(r["_dur16"], 2),
        }
        if random.random() < REF_FRAC:
            others = [o for o in pool
                      if o["emotion"] == r["emotion"] and o["file"] != r["file"]]
            if others:
                item["ref_audio"] = f"{BOX_WAVS}/{random.choice(others)['_wav']}"
        return item

    with (OUT / "train.jsonl").open("w", encoding="utf-8") as fh:
        for r in train:
            fh.write(json.dumps(jsonl_row(r, train), ensure_ascii=False) + "\n")
    with (OUT / "val.jsonl").open("w", encoding="utf-8") as fh:
        for r in val:
            fh.write(json.dumps(jsonl_row(r, train), ensure_ascii=False) + "\n")

    for r in ref_rows:
        d = OUT / "refs_emo" / r["emotion"]
        d.mkdir(exist_ok=True)
        shutil.copy2(VOICES / r["path"], d / r["file"])

    emo_stat = defaultdict(float)
    for r in train_rows:
        emo_stat[r["emotion"]] += r["_dur16"]
    print("--- минуты по эмоциям (train+val) ---", flush=True)
    for e, d in sorted(emo_stat.items(), key=lambda x: -x[1]):
        print(f"  {e}: {d/60:.1f} мин", flush=True)

    if TGZ.exists():
        TGZ.unlink()
    subprocess.run(
        ["tar", "czf", str(TGZ), "-C", str(OUT.parent), "lora_data"],
        check=True,
    )
    print(f"архив: {TGZ} ({TGZ.stat().st_size/1e6:.1f} МБ)", flush=True)
    print("BUILD_DONE", flush=True)


if __name__ == "__main__":
    sys.stdout.reconfigure(line_buffering=True)
    main()
