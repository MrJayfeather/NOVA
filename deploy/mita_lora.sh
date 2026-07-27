#!/bin/bash
# LoRA-дообучение VoxCPM2 на кураторском датасете Миты (450 клипов, 14 мин).
# Ожидает: /workspace/mita_lora/{train.jsonl,val.jsonl,wavs/} (заливается с
# ноута), vox-venv и веса openbmb/VoxCPM2 из onstart. GPU должен быть
# свободен: runner.sh заглушен ДО его запуска, vLLM не стартует.
set -x
export HF_HOME=/workspace/hf
unset HF_ENDPOINT
cd /workspace
VOXPY=/workspace/vox/bin/python
# самолечение: пробуждение может утащить numpy венва в 1.x (DFN-блок
# onstart тянет deepfilternet с numpy<2, а системный scipy собран под
# numpy 2) — чиним ДО любой работы
$VOXPY -c 'import scipy.signal' 2>/dev/null || \
  /workspace/vox/bin/pip install -q 'numpy==2.2.*' -i https://pypi.org/simple
# параметры прогона (переопределяются окружением):
export LORA_ITERS=${LORA_ITERS:-400}
export LORA_SAVE_INT=${LORA_SAVE_INT:-25}
export LORA_CKPT_DIR=${LORA_CKPT_DIR:-/workspace/ckpts_mita_lora}

# страховка от забытого бокса: через 3 часа без /workspace/keep_alive —
# самостоп (после подмены runner вачдог не работает: он не запускается)
(
  sleep 10800
  [ -f /workspace/keep_alive ] && exit 0
  KEY=$(tr '\0' '\n' < /proc/1/environ | grep '^VAST_API_KEY=' | cut -d= -f2-)
  IID=$(tr '\0' '\n' < /proc/1/environ | grep '^VAST_CONTAINERLABEL=' | grep -o '[0-9]\+')
  curl -s -X PUT "https://console.vast.ai/api/v0/instances/$IID/" \
    -H "Authorization: Bearer $KEY" -H 'Content-Type: application/json' \
    -d '{"state":"stopped"}'
) > /workspace/selfstop.log 2>&1 &

# 1. репо с тренировочными скриптами OpenBMB
if [ ! -d /workspace/voxrepo ]; then
  for i in 1 2 3; do
    git clone https://github.com/OpenBMB/VoxCPM.git /workspace/voxrepo && break
    sleep 5
  done
fi
[ -d /workspace/voxrepo ] || { echo CLONE_FAIL; exit 1; }
/workspace/vox/bin/pip install -e /workspace/voxrepo -i https://pypi.org/simple \
  > /workspace/loradeps.log 2>&1 || { echo DEPS_FAIL; exit 1; }
echo DEPS_OK

# 2. конфиг: шаблон репо с нашими путями и параметрами
SNAP=$($VOXPY -c "from huggingface_hub import snapshot_download; print(snapshot_download('openbmb/VoxCPM2'))") \
  || { echo SNAP_FAIL; exit 1; }
SNAP="$SNAP" $VOXPY - <<'PY' || { echo CONF_FAIL; exit 1; }
import os

import yaml

# СТРОГО v2-шаблон: v1.5 несёт sample_rate 44100, а AudioVAE VoxCPM2
# требует 16000 — соседний шаблон по глобу уже наступал на эти грабли
tpl = "/workspace/voxrepo/conf/voxcpm_v2/voxcpm_finetune_lora.yaml"
cfg = yaml.safe_load(open(tpl))
print("шаблон:", tpl, "ключи:", sorted(cfg))
cfg["pretrained_path"] = os.environ["SNAP"]
cfg["train_manifest"] = "/workspace/mita_lora/train.jsonl"
cfg["val_manifest"] = "/workspace/mita_lora/val.jsonl"
cfg["save_path"] = os.environ["LORA_CKPT_DIR"]
# 430 клипов × эфф. батч 16 ≈ 27 шагов/эпоха; TTS переобучается быстро —
# частые чекпоинты, отбор по val-лоссу и ушам
for k in ("num_iters", "max_iters", "total_iters", "max_steps"):
    if k in cfg:
        cfg[k] = int(os.environ["LORA_ITERS"])
for k in ("save_interval", "save_every", "ckpt_interval",
          "val_interval", "valid_interval", "eval_interval"):
    if k in cfg:
        cfg[k] = int(os.environ["LORA_SAVE_INT"])
lora = cfg.get("lora") or {}
lora.update({"r": 32, "alpha": 32, "enable_lm": True, "enable_dit": True})
cfg["lora"] = lora
with open("/workspace/mita_lora/lora.yaml", "w") as fh:
    yaml.safe_dump(cfg, fh, allow_unicode=True)
PY
echo CONF_OK

# 3. тренировка (имя train_voxcpm* — busy-паттерн вачдога на будущее)
cd /workspace/voxrepo
$VOXPY scripts/train_voxcpm_finetune.py \
  --config_path /workspace/mita_lora/lora.yaml \
  > /workspace/lora_train.log 2>&1
code=$?
if [ $code -eq 0 ]; then echo TRAIN_OK; else echo "TRAIN_FAIL:$code"; fi
ls -la /workspace/ckpts_mita_lora/ || true
echo SCRIPT_DONE
