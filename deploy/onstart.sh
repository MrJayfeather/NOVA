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

# fish-speech: отдельный venv (его зависимости конфликтуют с vLLM-образом).
# pyaudio требует portaudio-заголовки; индекс pypi.org явно — зеркала хостов
# бывают неполными, и pip откатывается к древним версиям без готовых колёс.
if [ ! -d /workspace/fishenv ]; then
  export DEBIAN_FRONTEND=noninteractive
  apt-get update -qq > /workspace/apt.log 2>&1
  apt-get install -y -qq portaudio19-dev >> /workspace/apt.log 2>&1
  python3 -m venv /workspace/fishenv
  git clone https://github.com/fishaudio/fish-speech /workspace/fish-speech
  /workspace/fishenv/bin/pip install -e /workspace/fish-speech "huggingface_hub[cli]" \
    -i https://pypi.org/simple > /workspace/fishpip.log 2>&1
fi
if [ ! -f /workspace/checkpoints/openaudio-s1-mini/codec.pth ]; then
  [ -f /workspace/hf_token ] && export HF_TOKEN=$(cat /workspace/hf_token)
  # через python-api: имя cli-команды меняется между версиями huggingface_hub
  /workspace/fishenv/bin/python -c "import os; from huggingface_hub import snapshot_download; snapshot_download('fishaudio/openaudio-s1-mini', local_dir='/workspace/checkpoints/openaudio-s1-mini', token=os.environ.get('HF_TOKEN') or None)" \
    > /workspace/hfdl.log 2>&1
fi

bash /workspace/NOVA/deploy/runner.sh
