"""Сортировка игровых реплик по голосу: сверка с эталонными клипами Миты.

Строит голосовой отпечаток по проверенным клипам (voices/*.ogg|wav),
затем раскладывает кучу (voices/голоса_в_куче/**/*.ogg) по папкам:
  voices/_отбор/mita_sure   — уверенно её голос
  voices/_отбор/mita_check  — пограничные, прослушать вручную
  voices/_отбор/too_short   — короче секунды, отпечаток ненадёжен
Остальные никуда не копируются (остаются только в куче).
Отчёт с численными оценками: voices/_отбор/report.csv

Запуск (качает модель спикер-эмбеддингов ~80МБ при первом запуске):
  uv run --with speechbrain --with soundfile python scripts/sort_voices.py
"""

import csv
import shutil
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio.functional as AF

ROOT = Path(__file__).resolve().parent.parent
REF_DIR = ROOT / "voices"
PILE_DIR = ROOT / "voices" / "голоса_в_куче"
OUT_DIR = ROOT / "voices" / "_отбор"
TARGET_SR = 16000
MIN_DUR_S = 1.0  # короче — отпечаток шаткий, в отдельную папку


def load_16k(path: Path) -> np.ndarray | None:
    try:
        wav, sr = sf.read(path, dtype="float32", always_2d=True)
    except Exception as exc:
        print(f"  ! не читается {path.name}: {exc}")
        return None
    mono = wav.mean(axis=1)
    if sr != TARGET_SR:
        mono = AF.resample(torch.from_numpy(mono), sr, TARGET_SR).numpy()
    return mono


def main() -> None:
    from speechbrain.inference.speaker import EncoderClassifier

    enc = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        savedir=str(ROOT / ".cache" / "ecapa"),
        run_opts={"device": "cpu"},
    )

    def embed(mono: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            e = enc.encode_batch(torch.from_numpy(mono)[None]).squeeze().numpy()
        return e / np.linalg.norm(e)

    refs = sorted(
        p for p in REF_DIR.iterdir()
        if p.is_file() and p.suffix.lower() in (".ogg", ".wav")
    )
    print(f"эталонных клипов: {len(refs)}")
    ref_embs = []
    for p in refs:
        mono = load_16k(p)
        if mono is not None and len(mono) >= TARGET_SR * 0.5:
            ref_embs.append(embed(mono))
    ref_embs = np.stack(ref_embs)

    # leave-one-out: насколько каждый эталон похож на центроид остальных —
    # это и валидация метода, и основа для порогов
    loo_sims = []
    total = ref_embs.sum(axis=0)
    for i, e in enumerate(ref_embs):
        c = total - e
        c = c / np.linalg.norm(c)
        loo_sims.append(float(e @ c))
    loo = np.array(loo_sims)
    print(f"валидация на эталонах: min={loo.min():.3f} p10={np.percentile(loo, 10):.3f} "
          f"медиана={np.median(loo):.3f} max={loo.max():.3f}")

    sure_thr = float(np.percentile(loo, 10))
    check_thr = sure_thr - 0.15
    print(f"пороги: точно >= {sure_thr:.3f}, проверить >= {check_thr:.3f}")

    centroid = ref_embs.sum(axis=0)
    centroid = centroid / np.linalg.norm(centroid)

    pile = sorted(PILE_DIR.rglob("*.ogg"))
    print(f"в куче: {len(pile)}")
    for sub in ("mita_sure", "mita_check", "too_short"):
        (OUT_DIR / sub).mkdir(parents=True, exist_ok=True)

    rows = []
    counts = {"mita_sure": 0, "mita_check": 0, "other": 0, "too_short": 0, "error": 0}
    sure_dur = 0.0
    for i, p in enumerate(pile, 1):
        if i % 200 == 0:
            print(f"  ... {i}/{len(pile)}")
        mono = load_16k(p)
        if mono is None:
            counts["error"] += 1
            rows.append((str(p.relative_to(PILE_DIR)), 0.0, "", "error"))
            continue
        dur = len(mono) / TARGET_SR
        uniq_name = f"{p.parent.name}_{p.name}".replace(" ", "_")
        if dur < MIN_DUR_S:
            counts["too_short"] += 1
            shutil.copy2(p, OUT_DIR / "too_short" / uniq_name)
            rows.append((str(p.relative_to(PILE_DIR)), round(dur, 2), "", "too_short"))
            continue
        sim = float(embed(mono) @ centroid)
        if sim >= sure_thr:
            verdict = "mita_sure"
            sure_dur += dur
        elif sim >= check_thr:
            verdict = "mita_check"
        else:
            verdict = "other"
        counts[verdict] += 1
        if verdict != "other":
            shutil.copy2(p, OUT_DIR / verdict / uniq_name)
        rows.append((str(p.relative_to(PILE_DIR)), round(dur, 2), round(sim, 3), verdict))

    with (OUT_DIR / "report.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["file", "dur_s", "sim", "verdict"])
        w.writerows(rows)

    print("--- итог ---")
    for k, v in counts.items():
        print(f"{k}: {v}")
    print(f"длительность mita_sure: {sure_dur / 60:.1f} мин")
    print("SORT_DONE")


if __name__ == "__main__":
    sys.stdout.reconfigure(line_buffering=True)
    main()
