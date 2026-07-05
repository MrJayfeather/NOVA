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
  # пин на последний S1-совместимый коммит: дальше main переписан под S2
  # (токенизатор через AutoTokenizer не читает tiktoken-чекпоинт S1-mini)
  git -C /workspace/fish-speech checkout 781bf1c
  # pip зацикливается на переборе версий (transformers без нижней границы
  # + datasets==2.18.0) — прижимаем констрейнтом; uv не подходит: не доверяет
  # TLS-сертификатам некоторых хостов (UnknownIssuer)
  echo "transformers>=4.45" > /workspace/constraints.txt
  /workspace/fishenv/bin/pip install -e /workspace/fish-speech "huggingface_hub[cli]" \
    -c /workspace/constraints.txt -i https://pypi.org/simple > /workspace/fishpip.log 2>&1
fi
# голос 3.0 (VoxCPM2) + сервер NOVA: venv поверх системных пакетов —
# системный python не трогаем (в нём живёт vLLM и его зависимости)
if [ ! -d /workspace/vox ]; then
  python3 -m venv --system-site-packages /workspace/vox
  /workspace/vox/bin/pip install voxcpm ruaccent -i https://pypi.org/simple \
    > /workspace/voxpip.log 2>&1
  /workspace/vox/bin/pip install -e /workspace/NOVA >> /workspace/voxpip.log 2>&1
fi
# DFN-полировка выхода: колёс под py3.12 нет — собирается через Rust
if ! /workspace/vox/bin/python -c 'import df' 2>/dev/null; then
  curl -sSf https://sh.rustup.rs | sh -s -- -y -q > /workspace/rust.log 2>&1
  source "$HOME/.cargo/env"
  /workspace/vox/bin/pip install deepfilternet -i https://pypi.org/simple \
    >> /workspace/voxpip.log 2>&1
fi

# веса VoxCPM2 — с повторами (сети хостов рвут долгие скачивания)
cat > /workspace/dl_vox.py <<'PY'
import time
from huggingface_hub import snapshot_download
for attempt in range(12):
    try:
        snapshot_download('openbmb/VoxCPM2')
        print('VOX_DONE')
        break
    except Exception as e:
        print(f'attempt {attempt}: {type(e).__name__}: {e}')
        time.sleep(5)
else:
    raise SystemExit(1)
PY
HF_HOME=/workspace/hf /workspace/vox/bin/python /workspace/dl_vox.py \
  > /workspace/voxdl.log 2>&1

if [ ! -f /workspace/checkpoints/openaudio-s1-mini/codec.pth ]; then
  [ -f /workspace/hf_token ] && export HF_TOKEN=$(cat /workspace/hf_token)
  # через python-api (имя cli-команды меняется между версиями huggingface_hub);
  # цикл повторов: у хостов рвутся долгие соединения, докачка идёт с места обрыва
  cat > /workspace/dl_s1.py <<'PY'
import os, time
from huggingface_hub import snapshot_download
for attempt in range(12):
    try:
        snapshot_download('fishaudio/openaudio-s1-mini',
                          local_dir='/workspace/checkpoints/openaudio-s1-mini',
                          token=os.environ.get('HF_TOKEN') or None)
        print('S1_DONE')
        break
    except Exception as e:
        print(f'attempt {attempt}: {type(e).__name__}: {e}')
        time.sleep(5)
else:
    raise SystemExit(1)
PY
  /workspace/fishenv/bin/python /workspace/dl_s1.py > /workspace/hfdl.log 2>&1
fi

bash /workspace/NOVA/deploy/runner.sh
