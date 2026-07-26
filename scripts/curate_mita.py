"""Куратор датасета Миты: метрики по оригинальным ogg + страница отслушивания.

Анализирует оригиналы (voices/*.ogg + voices/_отбор/mita_sure/*.ogg):
  - спикер-эмбеддинг ECAPA: похожесть на эталонный центроид (тембр-выбросы)
  - клиппинг, RMS, оценка шума/SNR по тихим кадрам
  - спектральные аномалии (эффекты: рация, глюки) через z-оценки
  - питч (yin/pyin): крик, писк, шёпот
  - дубликаты по md5
Сопоставляет каждый клип с f5data (0000..0556) по длительности + корреляции
огибающих — подтягивает тексты реплик и разметку ударений.

Выход:
  voices/_куратор/report.csv   — все метрики, флаги, корзина
  voices/отбор_LoRA.html       — страница отслушивания (открыть в браузере),
                                 галочки сохраняются в localStorage,
                                 кнопка «Скачать решения» отдаёт CSV.

Запуск (модель эмбеддингов уже в .cache/ecapa):
  uv run --with speechbrain --with soundfile --with librosa python scripts/curate_mita.py <путь_к_f5data>
"""

import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio.functional as AF

ROOT = Path(__file__).resolve().parent.parent
REF_DIR = ROOT / "voices"
SURE_DIR = ROOT / "voices" / "_отбор" / "mita_sure"
OUT_DIR = ROOT / "voices" / "_куратор"
PAGE = ROOT / "voices" / "отбор_LoRA.html"
SR = 16000


def load_clip(path: Path):
    """(native_mono, native_sr, mono_16k) или None."""
    try:
        wav, sr = sf.read(path, dtype="float32", always_2d=True)
    except Exception as exc:
        print(f"  ! не читается {path.name}: {exc}", flush=True)
        return None
    mono = wav.mean(axis=1)
    if sr != SR:
        m16 = AF.resample(torch.from_numpy(mono), sr, SR).numpy()
    else:
        m16 = mono
    return mono, sr, m16


def frame_rms_db(x: np.ndarray, sr: int, win_s=0.05, hop_s=0.025) -> np.ndarray:
    win, hop = int(sr * win_s), int(sr * hop_s)
    if len(x) < win:
        return np.array([20 * np.log10(np.sqrt(np.mean(x**2)) + 1e-9)])
    n = 1 + (len(x) - win) // hop
    frames = np.lib.stride_tricks.as_strided(
        x, shape=(n, win), strides=(x.strides[0] * hop, x.strides[0])
    )
    rms = np.sqrt(np.mean(frames**2, axis=1))
    return 20 * np.log10(rms + 1e-9)


def envelope(x16: np.ndarray) -> np.ndarray:
    e = frame_rms_db(x16, SR)
    e = e - e.mean()
    s = e.std()
    return e / s if s > 1e-6 else e


def env_corr(a: np.ndarray, b: np.ndarray, max_lag=24) -> float:
    best = -1.0
    for lag in range(-max_lag, max_lag + 1):
        if lag >= 0:
            aa, bb = a[lag:], b[: len(b) - lag if lag else len(b)]
        else:
            aa, bb = a[: len(a) + lag], b[-lag:]
        n = min(len(aa), len(bb))
        if n < 8:
            continue
        c = float(np.dot(aa[:n], bb[:n]) / n)
        best = max(best, c)
    return best


def main() -> None:
    import librosa

    # Прогрев ленивых импортов librosa СТРОГО до импорта speechbrain:
    # lazy_loader librosa зовёт inspect.stack(), тот трогает ленивый модуль
    # speechbrain.integrations.k2_fsa, который падает на отсутствующем k2.
    _w = np.zeros(SR, dtype=np.float32)
    _S = np.abs(librosa.stft(_w, n_fft=1024, hop_length=512))
    librosa.feature.spectral_centroid(S=_S, sr=SR)
    librosa.feature.spectral_flatness(S=_S)
    librosa.fft_frequencies(sr=SR, n_fft=1024)
    librosa.pyin(_w, fmin=80, fmax=800, sr=SR, frame_length=1024, hop_length=512)

    from speechbrain.inference.speaker import EncoderClassifier

    f5 = Path(sys.argv[1]) if len(sys.argv) > 1 else None

    enc = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        savedir=str(ROOT / ".cache" / "ecapa"),
        run_opts={"device": "cpu"},
    )

    def embed(m16: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            e = enc.encode_batch(torch.from_numpy(m16)[None]).squeeze().numpy()
        return e / np.linalg.norm(e)

    refs = sorted(
        p for p in REF_DIR.iterdir()
        if p.is_file() and p.suffix.lower() in (".ogg", ".wav")
    )
    sure = sorted(SURE_DIR.glob("*.ogg")) if SURE_DIR.is_dir() else []
    clips = [("ref", p) for p in refs] + [("sure", p) for p in sure]
    print(f"эталонов: {len(refs)}, mita_sure: {len(sure)}, всего: {len(clips)}", flush=True)

    rows = []
    ref_embs = []
    envs = {}
    for i, (src, p) in enumerate(clips, 1):
        if i % 50 == 0:
            print(f"  ... {i}/{len(clips)}", flush=True)
        loaded = load_clip(p)
        if loaded is None:
            rows.append({"file": p.name, "src": src, "error": True})
            continue
        native, nsr, m16 = loaded
        dur = len(m16) / SR
        md5 = hashlib.md5(p.read_bytes()).hexdigest()

        clip_ratio = float(np.mean(np.abs(native) > 0.985))
        fdb = frame_rms_db(m16, SR)
        speech_db = float(np.percentile(fdb, 80))
        floor_db = float(np.percentile(fdb, 5))
        snr = speech_db - floor_db

        S = np.abs(librosa.stft(m16, n_fft=1024, hop_length=512))
        cent = float(librosa.feature.spectral_centroid(S=S, sr=SR).mean())
        flat = float(librosa.feature.spectral_flatness(S=S).mean())
        freqs = librosa.fft_frequencies(sr=SR, n_fft=1024)
        tot = S.sum() + 1e-9
        hf_ratio = float(S[freqs > 5000].sum() / tot)

        try:
            f0, voiced, _ = librosa.pyin(
                m16, fmin=80, fmax=800, sr=SR,
                frame_length=1024, hop_length=512,
            )
            vf = f0[voiced] if voiced is not None else f0[~np.isnan(f0)]
            vf = vf[~np.isnan(vf)]
            f0_med = float(np.median(vf)) if len(vf) else 0.0
            f0_p90 = float(np.percentile(vf, 90)) if len(vf) else 0.0
            voiced_frac = float(np.mean(voiced)) if voiced is not None else 0.0
        except Exception:
            f0_med = f0_p90 = voiced_frac = 0.0

        emb = embed(m16)
        if src == "ref":
            ref_embs.append(emb)
        rows.append({
            "file": p.name, "src": src, "path": p, "dur": dur, "md5": md5,
            "clip_ratio": clip_ratio, "speech_db": speech_db, "snr": snr,
            "cent": cent, "flat": flat, "hf": hf_ratio,
            "f0_med": f0_med, "f0_p90": f0_p90, "voiced": voiced_frac,
            "emb": emb, "error": False,
        })
        envs[p.name] = envelope(m16)

    good = [r for r in rows if not r["error"]]

    # центроид эталонов + leave-one-out пороги (как в sort_voices)
    R = np.stack(ref_embs)
    total = R.sum(axis=0)
    loo = []
    for e in R:
        c = total - e
        c = c / np.linalg.norm(c)
        loo.append(float(e @ c))
    anchor = float(np.percentile(loo, 10))
    centroid = total / np.linalg.norm(total)
    for r in good:
        r["sim"] = float(r["emb"] @ centroid)
    print(f"LOO эталонов: p10={anchor:.3f} медиана={np.median(loo):.3f}", flush=True)

    # z-оценки спектра по корпусу
    for key in ("cent", "flat", "hf"):
        vals = np.array([r[key] for r in good])
        mu, sd = vals.mean(), vals.std() + 1e-9
        for r in good:
            r["z_" + key] = (r[key] - mu) / sd

    # дубликаты по md5
    seen = {}
    for r in good:
        seen.setdefault(r["md5"], []).append(r)
    for group in seen.values():
        for r in group[1:]:
            r["dup_of"] = group[0]["file"]

    # сопоставление с f5data по длительности + огибающей
    matched = 0
    if f5 and (f5 / "wavs").is_dir():
        stressed = {}
        with (f5 / "metadata.csv").open(encoding="utf-8") as fh:
            next(fh)
            for line in fh:
                af, text = line.rstrip("\n").split("|", 1)
                stressed[Path(af).stem] = text
        ds = []
        for w in sorted((f5 / "wavs").glob("*.wav")):
            loaded = load_clip(w)
            if loaded is None:
                continue
            _, _, m16 = loaded
            lab = w.with_suffix(".lab")
            ds.append({
                "id": w.stem, "dur": len(m16) / SR, "env": envelope(m16),
                "text": lab.read_text(encoding="utf-8").strip() if lab.exists() else "",
                "stress": stressed.get(w.stem, ""),
            })
        print(f"клипов в f5data: {len(ds)}", flush=True)
        for i, r in enumerate(good, 1):
            if i % 100 == 0:
                print(f"  сопоставление ... {i}/{len(good)}", flush=True)
            cands = [d for d in ds if abs(d["dur"] - r["dur"]) < 0.6]
            best, best_c = None, 0.5
            for d in cands:
                c = env_corr(envs[r["file"]], d["env"])
                if c > best_c:
                    best, best_c = d, c
            if best:
                r["ds_id"], r["text"], r["stress"], r["match_corr"] = (
                    best["id"], best["text"], best["stress"], best_c)
                matched += 1
    print(f"сопоставлено с датасетом: {matched}/{len(good)}", flush=True)

    # флаги и корзины
    for r in good:
        hard, soft = [], []
        if r["clip_ratio"] > 0.005:
            hard.append("клиппинг")
        elif r["clip_ratio"] > 0.0005:
            soft.append("клиппинг?")
        if r["snr"] < 15:
            hard.append("шум")
        elif r["snr"] < 22:
            soft.append("шум?")
        if r["sim"] < anchor - 0.15:
            hard.append("тембр")
        elif r["sim"] < anchor - 0.05:
            soft.append("тембр?")
        zbad = max(abs(r["z_cent"]), abs(r["z_flat"]), abs(r["z_hf"]))
        if zbad > 3.0:
            hard.append("эффекты")
        elif zbad > 2.2:
            soft.append("эффекты?")
        if r["dur"] < 1.0:
            soft.append("коротко")
        if r["speech_db"] > -14 and r["f0_p90"] > 450:
            soft.append("крик")
        if r["voiced"] < 0.3 and r["speech_db"] < -25:
            soft.append("шёпот")
        if r.get("dup_of"):
            soft.append("дубль")
        r["hard"], r["soft"] = hard, soft
        if hard:
            r["bucket"] = "мусор"
        elif soft:
            r["bucket"] = "пограничные"
        else:
            r["bucket"] = "ядро"
        # тон — информационная бирка
        r["tone"] = ""

    f0s = np.array([r["f0_med"] for r in good if r["f0_med"] > 0])
    dbs = np.array([r["speech_db"] for r in good])
    f0_lo, f0_hi = np.percentile(f0s, 33), np.percentile(f0s, 66)
    db_lo, db_hi = np.percentile(dbs, 33), np.percentile(dbs, 66)
    for r in good:
        calm = r["f0_med"] <= f0_lo and r["speech_db"] <= db_hi
        bright = r["f0_med"] >= f0_hi or r["speech_db"] >= db_hi
        r["tone"] = "спокойная" if calm else ("бодрая" if bright else "нейтральная")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cols = ["file", "src", "bucket", "dur", "tone", "sim", "clip_ratio", "snr",
            "speech_db", "f0_med", "f0_p90", "voiced", "z_cent", "z_flat",
            "z_hf", "hard", "soft", "dup_of", "ds_id", "text", "stress",
            "match_corr"]
    with (OUT_DIR / "report.csv").open("w", newline="", encoding="utf-8-sig") as fh:
        w = csv.writer(fh)
        w.writerow(cols)
        for r in good:
            w.writerow([
                ";".join(r[c]) if c in ("hard", "soft") else round(r[c], 4)
                if isinstance(r.get(c), float) else r.get(c, "")
                for c in cols
            ])

    # данные страницы
    items = []
    for r in good:
        rel = r["file"] if r["src"] == "ref" else f"_отбор/mita_sure/{r['file']}"
        items.append({
            "f": r["file"], "p": rel, "b": r["bucket"], "d": round(r["dur"], 2),
            "tone": r["tone"], "sim": round(r["sim"], 3),
            "flags": r["hard"] + r["soft"], "text": r.get("text", ""),
            "ds": r.get("ds_id", ""),
        })
    order = {"мусор": 0, "пограничные": 1, "ядро": 2}
    items.sort(key=lambda x: (order[x["b"]], x["sim"]))

    durs = {b: sum(i["d"] for i in items if i["b"] == b) / 60 for b in order}
    counts = {b: sum(1 for i in items if i["b"] == b) for b in order}
    print("--- корзины ---", flush=True)
    for b in order:
        print(f"{b}: {counts[b]} клипов, {durs[b]:.1f} мин", flush=True)

    tpl = PAGE_TPL.replace("__DATA__", json.dumps(items, ensure_ascii=False))
    PAGE.write_text(tpl, encoding="utf-8")
    print(f"страница: {PAGE}", flush=True)
    print("CURATE_DONE", flush=True)


PAGE_TPL = """<!doctype html><html lang="ru"><head><meta charset="utf-8">
<title>Отбор датасета Миты для LoRA</title>
<style>
body{font-family:Segoe UI,sans-serif;margin:16px;background:#1b1b22;color:#ddd}
h1{font-size:20px} h2{font-size:16px;margin:24px 0 8px;position:sticky;top:0;background:#1b1b22;padding:6px 0}
.row{display:flex;align-items:center;gap:8px;padding:4px 6px;border-bottom:1px solid #2c2c36}
.row:hover{background:#23232c;outline:1px solid #3a3a48}
.row.cur{background:#252533;outline:2px solid #4a6aaa}
.row.drop{opacity:.35;text-decoration:line-through}
audio{height:28px;width:220px}
.f{width:270px;font-size:12px;color:#9ad;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.t{flex:1;font-size:13px}
.badge{font-size:11px;padding:1px 6px;border-radius:8px;background:#444;margin-left:2px;white-space:nowrap}
.badge.hard{background:#7a2b2b}.badge.tone{background:#2b4a7a}
.d{width:46px;text-align:right;font-size:12px;color:#999}
.sim{width:50px;text-align:right;font-size:12px;color:#999}
button{margin:4px 8px 4px 0;padding:6px 12px;background:#2b4a7a;color:#fff;border:0;border-radius:6px;cursor:pointer}
.stats{font-size:13px;color:#aaa;margin:8px 0}
input[type=checkbox]{transform:scale(1.3)}
.emo{display:flex;gap:2px;align-items:center}
.eb{margin:0;padding:0 3px;line-height:20px;font-size:13px;background:#2c2c36;
 border:1px solid #444;border-radius:4px;cursor:pointer}
.eb.sel{background:#2b4a7a;border-color:#9ad}
.row.ncalm{border-left:3px solid #e90}
</style></head><body>
<h1>Отбор датасета Миты для LoRA</h1>
<p class="stats">Галочка = ВЫКИНУТЬ клип. Мои предложения уже проставлены
(корзина «мусор» помечена на выброс). Решения хранятся в браузере,
в конце нажми «Скачать решения». Эмоция клипа — кнопки-смайлы или клавиши 1–7:
1 😌 спокойная · 2 😄 радостная · 3 😏 игривая/ехидная · 4 😢 грустная ·
5 😠 злая · 6 😨 испуганная · 7 🤫 шёпот. Мои догадки грубые (бодрое = радостная) — правь по ушам.
Плеер: пробел — пауза, стрелки ↑/↓ по списку, x — галочка «выкинуть».</p>
<div><button onclick="exportCsv()">Скачать решения</button>
<button onclick="resetAll()">Сбросить к моим предложениям</button>
<span id="tally" class="stats"></span></div>
<div id="list"></div>
<script>
const DATA=__DATA__;
const KEY="mita_curate_v1";
const KEY2="mita_curate_tone_v1";
const KEY3="mita_curate_emo_v1";
const EMOS=["спокойная","радостная","игривая","грустная","злая","испуганная","шёпот"];
const EMOJI={"спокойная":"\\ud83d\\ude0c","радостная":"\\ud83d\\ude04",
 "игривая":"\\ud83d\\ude0f","грустная":"\\ud83d\\ude22","злая":"\\ud83d\\ude20",
 "испуганная":"\\ud83d\\ude28","шёпот":"\\ud83e\\udd2b"};
let st=JSON.parse(localStorage.getItem(KEY)||"null");
if(!st){st={};DATA.forEach(i=>st[i.f]=(i.b==="мусор"));}
let emo=JSON.parse(localStorage.getItem(KEY3)||"null");
if(!emo){
 emo={};
 const old=JSON.parse(localStorage.getItem(KEY2)||"null");
 DATA.forEach(i=>{
  if(old&&typeof old[i.f]==="boolean")emo[i.f]=old[i.f]?"радостная":"спокойная";
  else emo[i.f]=(i.tone==="бодрая")?"радостная":"спокойная";});
}
function save(){localStorage.setItem(KEY,JSON.stringify(st));
 localStorage.setItem(KEY3,JSON.stringify(emo));tally();}
function markCur(){rows.forEach(([r],j)=>r.classList.toggle("cur",j===cur));}
function setEmo(r,eg,i,em){emo[i.f]=em;
 const bs=eg.querySelectorAll(".eb");
 bs.forEach((x,j)=>x.classList.toggle("sel",EMOS[j]===em));
 r.classList.toggle("ncalm",em!=="спокойная");save();}
function tally(){
 let keep=0,kd=0,drop=0,dd=0;
 DATA.forEach(i=>{if(st[i.f]){drop++;dd+=i.d}else{keep++;kd+=i.d}});
 const cnt={};DATA.forEach(i=>{if(!st[i.f])cnt[emo[i.f]]=(cnt[emo[i.f]]||0)+1;});
 const parts=EMOS.filter(em=>cnt[em]).map(em=>EMOJI[em]+cnt[em]).join(" ");
 document.getElementById("tally").textContent=
  `остаётся: ${keep} клипов (${(kd/60).toFixed(1)} мин) [${parts}] · выкинуто: ${drop} (${(dd/60).toFixed(1)} мин)`;
}
const list=document.getElementById("list");
let cur=-1;const rows=[];
["мусор","пограничные","ядро"].forEach(b=>{
 const h=document.createElement("h2");
 const n=DATA.filter(i=>i.b===b).length;
 h.textContent=`${b} (${n})`;list.appendChild(h);
 DATA.filter(i=>i.b===b).forEach(i=>{
  const r=document.createElement("div");r.className="row";
  const cb=document.createElement("input");cb.type="checkbox";cb.checked=st[i.f];
  cb.onchange=()=>{st[i.f]=cb.checked;r.classList.toggle("drop",cb.checked);save();};
  const au=document.createElement("audio");au.controls=true;au.preload="none";
  au.src=encodeURI(i.p);
  au.onplay=()=>{rows.forEach(([q,a])=>{if(a!==au)a.pause()});
   cur=rows.findIndex(([q,a])=>a===au);markCur();};
  const f=document.createElement("span");f.className="f";f.textContent=i.f;
  const d=document.createElement("span");d.className="d";d.textContent=i.d+"с";
  const sim=document.createElement("span");sim.className="sim";sim.textContent=i.sim;
  const t=document.createElement("span");t.className="t";t.textContent=i.text||"—";
  const tone=document.createElement("span");tone.className="badge tone";tone.textContent=i.tone;
  const eg=document.createElement("span");eg.className="emo";
  EMOS.forEach(em=>{
   const b=document.createElement("button");b.type="button";b.className="eb";
   b.textContent=EMOJI[em];b.title=em;
   if(emo[i.f]===em)b.classList.add("sel");
   b.onclick=()=>{setEmo(r,eg,i,em);b.blur();};
   eg.appendChild(b);});
  r.append(cb,au,f,d,sim,tone,eg);
  i.flags.forEach(fl=>{const s=document.createElement("span");
   s.className="badge"+(["клиппинг","шум","тембр","эффекты"].includes(fl)?" hard":"");
   s.textContent=fl;r.appendChild(s);});
  r.appendChild(t);
  if(st[i.f])r.classList.add("drop");
  if(emo[i.f]!=="спокойная")r.classList.add("ncalm");
  list.appendChild(r);rows.push([r,au,i,cb,eg]);
 });
});
document.addEventListener("keydown",e=>{
 if(e.target.tagName==="INPUT"&&e.target.type!=="checkbox")return;
 if(e.code==="Space"){e.preventDefault();if(cur>=0){const a=rows[cur][1];a.paused?a.play():a.pause();}}
 if(e.code==="ArrowDown"||e.code==="ArrowUp"){e.preventDefault();
  cur=Math.min(rows.length-1,Math.max(0,cur+(e.code==="ArrowDown"?1:-1)));
  const[r,a]=rows[cur];r.scrollIntoView({block:"center"});markCur();a.play();}
 if(e.key==="x"||e.key==="ч"){if(cur>=0){const[r,a,i,cb]=rows[cur];
  cb.checked=!cb.checked;st[i.f]=cb.checked;r.classList.toggle("drop",cb.checked);save();}}
 const n=parseInt(e.key,10);
 if(n>=1&&n<=EMOS.length){if(cur>=0){const[r,a,i,cb,eg]=rows[cur];
  setEmo(r,eg,i,EMOS[n-1]);}}
});
function exportCsv(){
 let out="\\ufefffile,verdict,emotion\\n";
 DATA.forEach(i=>{out+=`${i.f},${st[i.f]?"drop":"keep"},${emo[i.f]}\\n`;});
 const b=new Blob([out],{type:"text/csv"});
 const a=document.createElement("a");a.href=URL.createObjectURL(b);
 a.download="решения_отбора.csv";a.click();
}
function resetAll(){st={};emo={};
 DATA.forEach(i=>{st[i.f]=(i.b==="мусор");
  emo[i.f]=(i.tone==="бодрая")?"радостная":"спокойная";});
 save();location.reload();}
tally();
</script></body></html>"""


if __name__ == "__main__":
    sys.stdout.reconfigure(line_buffering=True)
    main()
