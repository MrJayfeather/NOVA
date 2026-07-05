#!/bin/bash
# Идемпотентный запуск сервисов NOVA. Вызывается из onstart.sh при старте
# инстанса и вручную по ssh после git pull (перезапуск с новым кодом).
set -x
export $(tr '\0' '\n' < /proc/1/environ | grep -E '^(NOVA_MOCK|NOVA_TOKEN|NOVA_TTS|NOVA_FISH_CKPT|NOVA_FISH_KEY|NOVA_FISH_REF_ID|NOVA_FISH_TEMP|NOVA_FISH_TOP_P|NOVA_MODEL|NOVA_IDLE_LIMIT|NOVA_EYES|GEMINI_KEY|NOVA_GEMINI_MODEL|NOVA_VOX_TAG|NOVA_VOX_STRESS|NOVA_VOX_SEED|NOVA_GPU_UTIL|VAST_API_KEY|VAST_CONTAINERLABEL|HF_TOKEN)=' | tr '\n' ' ')
# файл-переопределение лимита простоя (сек) — удобно менять без пересоздания
[ -f /workspace/idle_limit ] && export NOVA_IDLE_LIMIT=$(cat /workspace/idle_limit)
export HF_HOME=/workspace/hf
export COQUI_TOS_AGREED=1
export NOVA_TTS=${NOVA_TTS:-fish}
# каталог голосовой модели: базовый s1-mini или дообученный (mita)
export NOVA_FISH_CKPT=${NOVA_FISH_CKPT:-/workspace/checkpoints/openaudio-s1-mini}
# облачный fish.audio: ключ кладётся в /workspace/fish_key (не в git!)
[ -f /workspace/fish_key ] && export NOVA_FISH_KEY=$(cat /workspace/fish_key)
# облачные глаза: ключ кладётся в /workspace/gemini_key (не в git!)
[ -f /workspace/gemini_key ] && export GEMINI_KEY=$(cat /workspace/gemini_key)
# env запечён при создании инстанса — файлы переопределяют без пересоздания
[ -f /workspace/nova_tts ] && export NOVA_TTS=$(cat /workspace/nova_tts)
[ -f /workspace/nova_eyes ] && export NOVA_EYES=$(cat /workspace/nova_eyes)
[ -f /workspace/hf_token ] && export HF_TOKEN=$(cat /workspace/hf_token)
export LD_LIBRARY_PATH="$(python3 -c 'import nvidia.cudnn; print(list(nvidia.cudnn.__path__)[0] + "/lib")'):$LD_LIBRARY_PATH"

cd /workspace/NOVA && git pull

# мозг (vLLM) — если ещё не поднят (процесс тоже считается: во время
# загрузки весов эндпоинт ещё молчит, второй запуск устроил бы конфликт)
export NOVA_MODEL=${NOVA_MODEL:-Qwen/Qwen3.6-27B-FP8}
if ! curl -s http://127.0.0.1:5000/v1/models > /dev/null \
   && ! pgrep -f '[v]llm serve' > /dev/null; then
  # 27B FP8 весит ~28ГБ; профиль-прогон мультимодалки жрёт активации сверх
  # бюджета — не задирать util и лимит картинок (OOM на старте)
  # enforce-eager: профилирование cuda-графов у 27B-мультимодалки
  # вылетает по памяти на 48ГБ; eager чуть медленнее, но стабильно
  # qwen3.6 токенизирует кадры жирно (~4к/кадр в нативном разрешении):
  # 6 кадров ~24к токенов — влезает в 32к вместе с текстом, 8 — нет
  # облачные глаза: кадры в мозг не ходят, KV-кэшу хватает 0.70 —
  # высвобождаем ~7ГБ под VoxCPM2. Локальные глаза — прежние 0.85.
  if [ "${NOVA_EYES:-gemini}" = "gemini" ] && [ -n "$GEMINI_KEY" ]; then
    export NOVA_GPU_UTIL=${NOVA_GPU_UTIL:-0.70}
  else
    export NOVA_GPU_UTIL=${NOVA_GPU_UTIL:-0.85}
  fi
  export NOVA_IMG_LIMIT=${NOVA_IMG_LIMIT:-6}
  nohup vllm serve "$NOVA_MODEL" \
    --host 127.0.0.1 --port 5000 --max-model-len 32768 \
    --gpu-memory-utilization "$NOVA_GPU_UTIL" \
    --limit-mm-per-prompt "{\"image\":$NOVA_IMG_LIMIT}" \
    --enforce-eager \
    > /workspace/vllm.log 2>&1 &
fi

# голос (fish-speech) — если выбран и ещё не поднят
if [ "$NOVA_TTS" = "fish" ] && ! curl -s -o /dev/null http://127.0.0.1:8081/; then
  cd /workspace/fish-speech
  # --compile разгоняет декодер в разы (14 ток/с -> реального времени мало),
  # ценой ~2 мин компиляции при старте
  nohup /workspace/fishenv/bin/python -m tools.api_server \
    --listen 127.0.0.1:8081 \
    --llama-checkpoint-path "$NOVA_FISH_CKPT" \
    --decoder-checkpoint-path "$NOVA_FISH_CKPT/codec.pth" \
    --decoder-config-name modded_dac_vq \
    --compile \
    > /workspace/fish.log 2>&1 &
  cd /workspace/NOVA
fi

# ждём готовности: vLLM до 45 мин (первая загрузка), fish до 15 мин
for i in $(seq 1 270); do
  curl -s http://127.0.0.1:5000/v1/models > /dev/null && break
  sleep 10
done
curl -s http://127.0.0.1:5000/v1/models > /dev/null || { echo "FATAL: vLLM не поднялся"; exit 1; }
if [ "$NOVA_TTS" = "fish" ]; then
  for i in $(seq 1 90); do
    curl -s -o /dev/null http://127.0.0.1:8081/ && break
    sleep 10
  done
  curl -s -o /dev/null http://127.0.0.1:8081/ || { echo "FATAL: fish-speech не поднялся"; exit 1; }
fi

cd /workspace/NOVA
pkill -f '[n]ova.server.main'
pkill -f '[i]dle_watchdog'
sleep 1
# сервер из vox-venv, если он есть (там voxcpm и ruaccent поверх системных)
PYBIN=python3
[ -x /workspace/vox/bin/python ] && PYBIN=/workspace/vox/bin/python
nohup "$PYBIN" -m nova.server.main > /workspace/nova.log 2>&1 &
nohup python3 deploy/idle_watchdog.py > /workspace/watchdog.log 2>&1 &
echo RUNNER_OK
