#!/bin/bash
# Файнтюн F5-TTS_RUSSIAN (Misha24-10, v2) на датасете Миты.
# Запускается на отдельном тренировочном инстансе (24ГБ+).
# Ожидает: /workspace/mita_dataset.tgz (ogg-клипы), deploy/finetune_prep.py.
set -x
cd /workspace
export HF_HOME=/workspace/hf
unset HF_ENDPOINT

# 1. зависимости
pip install -q f5-tts faster-whisper ruaccent -i https://pypi.org/simple \
  > /workspace/pip.log 2>&1 || { echo DEPS_FAIL; exit 1; }
echo DEPS_OK

# 2. русский чекпоинт и словарь
python - <<'PY' || { echo CKPT_FAIL; exit 1; }
import time
from huggingface_hub import hf_hub_download
for attempt in range(10):
    try:
        hf_hub_download('Misha24-10/F5-TTS_RUSSIAN', 'F5TTS_v1_Base_v2/model_last.pt',
                        local_dir='/workspace/f5ru')
        hf_hub_download('Misha24-10/F5-TTS_RUSSIAN', 'F5TTS_v1_Base/vocab.txt',
                        local_dir='/workspace/f5ru')
        break
    except Exception as e:
        print('attempt', attempt, e); time.sleep(5)
else:
    raise SystemExit(1)
PY
echo CKPT_OK

# 3. датасет: ogg -> wav 24к + транскрипты виспером
mkdir -p /workspace/mita_raw && tar xzf /workspace/mita_dataset.tgz -C /workspace/mita_raw
python /workspace/finetune_prep.py /workspace/mita_raw /workspace/f5data/wavs 24000 \
  > /workspace/prep.log 2>&1 || { echo PREP_FAIL; exit 1; }
echo PREP_OK

# 4. metadata.csv с ударениями RUAccent
python - <<'PY' || { echo META_FAIL; exit 1; }
from pathlib import Path
from ruaccent import RUAccent
acc = RUAccent(); acc.load(omograph_model_size='turbo', use_dictionary=True)
rows = []
for lab in sorted(Path('/workspace/f5data/wavs').glob('*.lab')):
    text = lab.read_text(encoding='utf-8').strip()
    rows.append(f"wavs/{lab.stem}.wav|{acc.process_all(text)}")
Path('/workspace/f5data/metadata.csv').write_text(
    "audio_file|text\n" + "\n".join(rows), encoding='utf-8')
print('строк:', len(rows))
PY
echo META_OK

# 5. подготовка arrow-датасета (словарь — от претрейна!)
SITE=$(python -c 'import f5_tts, pathlib; print(pathlib.Path(f5_tts.__file__).parent)')
python "$SITE/train/datasets/prepare_csv_wavs.py" /workspace/f5data \
  /workspace/f5arrow --pretrain /workspace/f5ru/F5TTS_v1_Base/vocab.txt \
  > /workspace/arrow.log 2>&1 || \
python "$SITE/train/datasets/prepare_csv_wavs.py" /workspace/f5data /workspace/f5arrow \
  > /workspace/arrow.log 2>&1 || { echo ARROW_FAIL; exit 1; }
echo ARROW_OK

# 6. тренировка от русского чекпоинта
ln -sfn /workspace/f5arrow "$SITE/../../data/mita_custom" 2>/dev/null || true
mkdir -p /workspace/runs
accelerate launch --mixed_precision=fp16 "$SITE/train/finetune_cli.py" \
  --exp_name F5TTS_v1_Base \
  --pretrain /workspace/f5ru/F5TTS_v1_Base_v2/model_last.pt \
  --dataset_name mita_custom \
  --tokenizer custom \
  --tokenizer_path /workspace/f5ru/F5TTS_v1_Base/vocab.txt \
  --learning_rate 1e-5 \
  --batch_size_per_gpu 3200 \
  --batch_size_type frame \
  --grad_accumulation_steps 1 \
  --epochs 120 \
  --num_warmup_updates 300 \
  --save_per_updates 2000 \
  --keep_last_n_checkpoints 3 \
  --finetune \
  > /workspace/train.log 2>&1 || { echo TRAIN_FAIL; exit 1; }
echo TRAIN_OK
