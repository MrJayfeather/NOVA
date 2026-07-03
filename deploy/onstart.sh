#!/bin/bash
# Провижининг инстанса Vast.ai (образ vllm/vllm-openai).
# Все тяжёлые данные — на /workspace (переживает stop/start).
set -x
export HF_HOME=/workspace/hf
export COQUI_TOS_AGREED=1

cd /workspace
if [ ! -d NOVA ]; then
  git clone https://github.com/MrJayfeather/NOVA.git
fi
cd NOVA && git pull

pip install -e . faster-whisper coqui-tts "nvidia-cudnn-cu12>=9" \
  > /workspace/pip.log 2>&1

# ctranslate2 (whisper) ищет cudnn в pip-пакете
export LD_LIBRARY_PATH="$(python3 -c 'import nvidia.cudnn, os; print(os.path.join(os.path.dirname(nvidia.cudnn.__file__), "lib"))'):$LD_LIBRARY_PATH"

# 1) vLLM с мозгом (Qwen3-VL). Первый старт качает ~31 ГБ весов.
nohup vllm serve Qwen/Qwen3-VL-30B-A3B-Instruct-FP8 \
  --host 127.0.0.1 --port 5000 \
  --max-model-len 16384 \
  --gpu-memory-utilization 0.75 \
  --limit-mm-per-prompt '{"image":12}' \
  > /workspace/vllm.log 2>&1 &

# ждём готовности vLLM, потом поднимаем оркестратор
until curl -s http://127.0.0.1:5000/v1/models > /dev/null; do sleep 10; done

cd /workspace/NOVA
nohup python3 -m nova.server.main > /workspace/nova.log 2>&1 &
nohup python3 deploy/idle_watchdog.py > /workspace/watchdog.log 2>&1 &
