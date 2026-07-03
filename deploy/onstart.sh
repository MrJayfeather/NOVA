#!/bin/bash
# Провижининг инстанса Vast.ai (образ vllm/vllm-openai).
# Все тяжёлые данные — на /workspace (переживает stop/start).
set -x
export HF_HOME=/workspace/hf
export COQUI_TOS_AGREED=1

mkdir -p /workspace
cd /workspace
for i in 1 2 3 4 5; do
  [ -d NOVA ] && break
  git clone https://github.com/MrJayfeather/NOVA.git && break
  echo "clone failed, retry $i"; sleep 10
done
if [ ! -d NOVA ]; then
  echo "FATAL: git clone не удался — у хоста нет доступа к github"
  exit 1
fi
cd NOVA && git pull

pip install -e . faster-whisper coqui-tts "nvidia-cudnn-cu12>=9" \
  > /workspace/pip.log 2>&1

# ctranslate2 (whisper) ищет cudnn в pip-пакете (namespace-пакет: путь через __path__)
export LD_LIBRARY_PATH="$(python3 -c 'import nvidia.cudnn; print(list(nvidia.cudnn.__path__)[0] + "/lib")'):$LD_LIBRARY_PATH"

# 1) vLLM с мозгом (Qwen3-VL). Первый старт качает ~31 ГБ весов.
nohup vllm serve Qwen/Qwen3-VL-30B-A3B-Instruct-FP8 \
  --host 127.0.0.1 --port 5000 \
  --max-model-len 16384 \
  --gpu-memory-utilization 0.75 \
  --limit-mm-per-prompt '{"image":12}' \
  > /workspace/vllm.log 2>&1 &

# ждём готовности vLLM (максимум ~45 минут), потом поднимаем оркестратор
for i in $(seq 1 270); do
  curl -s http://127.0.0.1:5000/v1/models > /dev/null && break
  sleep 10
done
if ! curl -s http://127.0.0.1:5000/v1/models > /dev/null; then
  echo "FATAL: vLLM не поднялся за 45 минут — смотри /workspace/vllm.log"
  exit 1
fi

cd /workspace/NOVA
nohup python3 -m nova.server.main > /workspace/nova.log 2>&1 &
nohup python3 deploy/idle_watchdog.py > /workspace/watchdog.log 2>&1 &
