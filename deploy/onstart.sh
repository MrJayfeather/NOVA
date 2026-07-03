#!/bin/bash
# Провижининг инстанса Vast.ai (образ vllm/vllm-openai): код, зависимости,
# веса. Идемпотентен — можно перезапускать. Сервисы стартует runner.sh.
set -x
export HF_HOME=/workspace/hf
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

# fish-speech: отдельный venv (его зависимости конфликтуют с vLLM-образом)
if [ ! -d /workspace/fishenv ]; then
  python3 -m venv /workspace/fishenv
  git clone https://github.com/fishaudio/fish-speech /workspace/fish-speech
  /workspace/fishenv/bin/pip install -e /workspace/fish-speech "huggingface_hub[cli]" \
    > /workspace/fishpip.log 2>&1
fi
if [ ! -f /workspace/checkpoints/openaudio-s1-mini/codec.pth ]; then
  [ -f /workspace/hf_token ] && export HF_TOKEN=$(cat /workspace/hf_token)
  /workspace/fishenv/bin/hf download fishaudio/openaudio-s1-mini \
    --local-dir /workspace/checkpoints/openaudio-s1-mini > /workspace/hfdl.log 2>&1
fi

bash /workspace/NOVA/deploy/runner.sh
