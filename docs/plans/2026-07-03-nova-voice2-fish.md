# NOVA «Голос 2.0» — Implementation Plan

> Выполнять задачи строго по порядку; чекбоксы `- [ ]` — для отметки прогресса.
> Контекст и предыстория решения — в `docs/STATUS.md`.

**Goal:** Живой голос Миты через OpenAudio S1-mini (fish-speech) рядом с XTTS, с переключением одним env и честным A/B-сравнением на слух.

**Architecture:** fish-speech ставится на инстанс в ОТДЕЛЬНЫЙ venv (его зависимости конфликтуют с vLLM-образом) и поднимается своим api_server на 127.0.0.1:8081. Оркестратор получает новый `FishTTS(TTSModel)` — шлёт предложения по HTTP (msgpack) с референс-аудио Миты и её транскриптом, получает WAV, отдаёт PCM в существующий стриминговый пайплайн. Выбор движка: env `NOVA_TTS` (`xtts` по умолчанию — обратная совместимость; `fish` включается раннером).

**Tech Stack:** fish-speech (api_server), openaudio-s1-mini (HF, gated, CC-BY-NC-SA), httpx + ormsgpack, стандартный `wave` для разбора ответа.

## Global Constraints

- Никаких упоминаний AI-инструментов в коде/коммитах/доках.
- GPU-зависимости не ставятся на ноут; `FishTTS` использует только httpx/ormsgpack/wave — тестируется локально без сети (чистые функции).
- Порт fish: **8081** (8080 занят прокси Vast).
- Модель gated: перед задачей 4 пользователь должен принять условия на
  https://huggingface.co/fishaudio/openaudio-s1-mini и положить read-токен
  в `.env` как `HF_TOKEN=hf_...`.
- Аудио к клиенту: PCM16 mono; частота из `SpeakStart.sample_rate` (S1 отдаёт 44100).
- Существующий СПЯЩИЙ инстанс 43664615 переиспользуем (кэш моделей цел);
  его сохранённый onstart — старой версии, поэтому дефолт `NOVA_TTS=xtts`
  обязан оставаться рабочим без fish.

## File Structure

```
nova/server/models/fish_tts.py  — FishTTS + чистые build_tts_request/wav_to_pcm (новый)
nova/server/main.py             — ветка NOVA_TTS в build_models
deploy/runner.sh                — запуск сервисов, ВЕРСИОНИРУЕТСЯ в репо (новый)
deploy/onstart.sh               — провижининг (клоны, venv, веса) + вызов runner.sh
scripts/vast.py                 — HF_TOKEN в env новых инстансов
personas/nova/voice_sample.txt  — транскрипт референса (создаётся в задаче 4)
tests/test_fish_tts.py          — тесты чистых функций (новый)
```

---

### Task 1: FishTTS — клиент fish-speech за интерфейсом TTSModel

**Files:**
- Create: `nova/server/models/fish_tts.py`
- Modify: `pyproject.toml` (добавить `ormsgpack`)
- Test: `tests/test_fish_tts.py`

**Interfaces:**
- Consumes: `TTSModel` (base), `split_for_tts` (xtts_tts).
- Produces:
  - `build_tts_request(text: str, ref_audio: bytes, ref_text: str) -> dict`
  - `wav_to_pcm(wav_bytes: bytes) -> tuple[bytes, int]` — (PCM16 mono, частота)
  - `FishTTS(url: str, reference_wav: Path, reference_text: str, timeout: float = 120.0)`,
    `FishTTS.sample_rate == 44100` (уточняется по факту первого ответа).

- [ ] **Step 1: Написать падающий тест**

```python
# tests/test_fish_tts.py
import wave
from io import BytesIO

from nova.server.models.fish_tts import build_tts_request, wav_to_pcm


def make_wav(rate=44100, channels=1, frames=b"\x01\x00" * 200) -> bytes:
    buf = BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(channels)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(frames)
    return buf.getvalue()


def test_build_request_shape():
    req = build_tts_request("привет", b"refbytes", "текст референса")
    assert req["text"] == "привет"
    assert req["format"] == "wav"
    assert req["streaming"] is False
    assert req["references"] == [{"audio": b"refbytes", "text": "текст референса"}]


def test_wav_to_pcm_mono_roundtrip():
    frames = b"\x01\x00\x02\x00" * 100
    pcm, rate = wav_to_pcm(make_wav(rate=44100, frames=frames))
    assert rate == 44100
    assert pcm == frames


def test_wav_to_pcm_downmixes_stereo():
    stereo = (b"\x00\x00\x64\x00") * 50  # L=0, R=100 -> моно=50
    pcm, rate = wav_to_pcm(make_wav(rate=24000, channels=2, frames=stereo))
    assert rate == 24000
    import numpy as np
    assert np.frombuffer(pcm, dtype=np.int16)[0] == 50
```

- [ ] **Step 2: Убедиться, что тест падает**

Run: `uv run pytest tests/test_fish_tts.py -q`
Expected: FAIL — `No module named 'nova.server.models.fish_tts'`

- [ ] **Step 3: Реализовать**

В `pyproject.toml` в dependencies добавить строку `"ormsgpack>=1.5",`, затем `uv sync`.

```python
# nova/server/models/fish_tts.py
import wave
from io import BytesIO
from pathlib import Path
from typing import AsyncIterator

import httpx
import ormsgpack

from nova.server.models.base import TTSModel
from nova.server.models.xtts_tts import split_for_tts


def build_tts_request(text: str, ref_audio: bytes, ref_text: str) -> dict:
    return {
        "text": text,
        "references": [{"audio": ref_audio, "text": ref_text}],
        "format": "wav",
        "streaming": False,
    }


def wav_to_pcm(wav_bytes: bytes) -> tuple[bytes, int]:
    with wave.open(BytesIO(wav_bytes)) as w:
        rate = w.getframerate()
        pcm = w.readframes(w.getnframes())
        if w.getnchannels() == 2:
            import numpy as np

            arr = np.frombuffer(pcm, dtype=np.int16).reshape(-1, 2)
            pcm = arr.mean(axis=1).astype(np.int16).tobytes()
    return pcm, rate


class FishTTS(TTSModel):
    """Голос через fish-speech api_server (OpenAudio S1-mini)."""

    sample_rate = 44100  # DAC-кодек S1; сверяется с реальным ответом

    def __init__(self, url: str, reference_wav: Path, reference_text: str,
                 timeout: float = 120.0):
        self._url = url
        self._ref_audio = Path(reference_wav).read_bytes()
        self._ref_text = reference_text
        self._client = httpx.AsyncClient(timeout=timeout)

    async def _tts_call(self, sentence: str) -> bytes:
        req = build_tts_request(sentence, self._ref_audio, self._ref_text)
        r = await self._client.post(
            self._url,
            content=ormsgpack.packb(req),
            headers={"Content-Type": "application/msgpack"},
        )
        r.raise_for_status()
        return r.content

    async def synthesize(self, text: str) -> AsyncIterator[bytes]:
        for sentence in split_for_tts(text)[:5]:
            try:
                pcm, rate = wav_to_pcm(await self._tts_call(sentence))
            except Exception as exc:
                print(f"[nova] ошибка fish-tts: {exc!r}")
                return
            if rate != self.sample_rate:
                print(f"[nova] fish-tts: частота {rate} (ожидалась {self.sample_rate})")
                self.sample_rate = rate
            yield pcm
```

- [ ] **Step 4: Прогнать тесты**

Run: `uv run pytest -q`
Expected: все passed (+2 skipped)

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "feat: fish-speech tts client behind ttsmodel interface"
```

---

### Task 2: Переключатель движка NOVA_TTS в build_models

**Files:**
- Modify: `nova/server/main.py` (функция `build_models`)

**Interfaces:**
- Produces: env `NOVA_TTS`: `"xtts"` (дефолт) | `"fish"`; env `NOVA_FISH_URL`
  (дефолт `http://127.0.0.1:8081/v1/tts`). Для fish требуется файл
  `personas/<persona>/voice_sample.txt` с транскриптом референса.

- [ ] **Step 1: Заменить блок выбора TTS в build_models**

Было (в конце `build_models`, после создания `llm`):

```python
    persona = os.environ.get("NOVA_PERSONA", "nova")
    tts = XttsTTS(speaker_wav=Path("personas") / persona / "voice_sample.wav")
    return asr, llm, tts
```

Стало:

```python
    persona = os.environ.get("NOVA_PERSONA", "nova")
    ref_dir = Path("personas") / persona
    if os.environ.get("NOVA_TTS", "xtts") == "fish":
        from nova.server.models.fish_tts import FishTTS

        tts = FishTTS(
            url=os.environ.get("NOVA_FISH_URL", "http://127.0.0.1:8081/v1/tts"),
            reference_wav=ref_dir / "voice_sample.wav",
            reference_text=(ref_dir / "voice_sample.txt").read_text(encoding="utf-8").strip(),
        )
    else:
        tts = XttsTTS(speaker_wav=ref_dir / "voice_sample.wav")
    return asr, llm, tts
```

(импорт `XttsTTS` в real-ветке уже есть; строку `tts = XttsTTS(...)` из старого места удалить.)

- [ ] **Step 2: Прогнать тесты (mock-режим не задет)**

Run: `uv run pytest -q`
Expected: все passed

- [ ] **Step 3: Commit**

```bash
git add nova/server/main.py
git commit -m "feat: tts engine switch via nova_tts env"
```

---

### Task 3: runner.sh в репозиторий, onstart.sh с fish-провижинингом, HF_TOKEN

**Files:**
- Create: `deploy/runner.sh`
- Modify: `deploy/onstart.sh` (полная замена), `scripts/vast.py`

**Interfaces:**
- Produces: `deploy/runner.sh` — идемпотентный запуск сервисов (vLLM,
  fish при NOVA_TTS=fish, оркестратор, вотчдог); используется и из onstart,
  и для ручного перезапуска по ssh. HF-токен: новые инстансы получают env
  `HF_TOKEN` из `.env`; на существующих читается файл `/workspace/hf_token`.

- [ ] **Step 1: Создать deploy/runner.sh**

```bash
#!/bin/bash
# Идемпотентный запуск сервисов NOVA. Вызывается из onstart.sh при старте
# инстанса и вручную по ssh после git pull (перезапуск с новым кодом).
set -x
export $(tr '\0' '\n' < /proc/1/environ | grep -E '^(NOVA_MOCK|NOVA_TOKEN|NOVA_TTS|VAST_API_KEY|VAST_CONTAINERLABEL|HF_TOKEN)=' | tr '\n' ' ')
export HF_HOME=/workspace/hf
export COQUI_TOS_AGREED=1
export NOVA_TTS=${NOVA_TTS:-fish}
[ -f /workspace/hf_token ] && export HF_TOKEN=$(cat /workspace/hf_token)
export LD_LIBRARY_PATH="$(python3 -c 'import nvidia.cudnn; print(list(nvidia.cudnn.__path__)[0] + "/lib")'):$LD_LIBRARY_PATH"

cd /workspace/NOVA && git pull

# мозг (vLLM) — если ещё не поднят
if ! curl -s http://127.0.0.1:5000/v1/models > /dev/null; then
  nohup vllm serve Qwen/Qwen3-VL-30B-A3B-Instruct-FP8 \
    --host 127.0.0.1 --port 5000 --max-model-len 16384 \
    --gpu-memory-utilization 0.75 --limit-mm-per-prompt '{"image":12}' \
    > /workspace/vllm.log 2>&1 &
fi

# голос (fish-speech) — если выбран и ещё не поднят
if [ "$NOVA_TTS" = "fish" ] && ! curl -s -o /dev/null http://127.0.0.1:8081/; then
  cd /workspace/fish-speech
  nohup /workspace/fishenv/bin/python -m tools.api_server \
    --listen 127.0.0.1:8081 \
    --llama-checkpoint-path /workspace/checkpoints/openaudio-s1-mini \
    --decoder-checkpoint-path /workspace/checkpoints/openaudio-s1-mini/codec.pth \
    --decoder-config-name modded_dac_vq \
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
nohup python3 -m nova.server.main > /workspace/nova.log 2>&1 &
nohup python3 deploy/idle_watchdog.py > /workspace/watchdog.log 2>&1 &
echo RUNNER_OK
```

- [ ] **Step 2: Переписать deploy/onstart.sh**

```bash
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
```

- [ ] **Step 3: scripts/vast.py — прокинуть HF_TOKEN новым инстансам**

В `create_instance` заменить сигнатуру и env:

```python
def create_instance(key: str, offer_id: int, token: str, hf_token: str = "") -> None:
```

и в словарь `"env"` добавить строку:

```python
            "HF_TOKEN": hf_token,
```

В `cmd_up` заменить вызов:

```python
        create_instance(key, offer["id"], token, hf_token=env.get("HF_TOKEN", ""))
```

- [ ] **Step 4: Прогнать тесты и commit**

Run: `uv run pytest -q`
Expected: все passed

```bash
git add -A
git commit -m "feat: versioned runner, fish provisioning and hf token wiring"
git push origin master
```

---

### Task 4: Пререквизит пользователя — доступ к весам

Ручной шаг ПОЛЬЗОВАТЕЛЯ (модель gated):

- [ ] **Step 1:** Зайти (или зарегистрироваться) на huggingface.co.
- [ ] **Step 2:** Открыть https://huggingface.co/fishaudio/openaudio-s1-mini →
  заполнить форму доступа (страна, дата, галочка «некоммерческое использование»)
  → Accept.
- [ ] **Step 3:** https://huggingface.co/settings/tokens → Create new token →
  тип **Read** → скопировать `hf_...`.
- [ ] **Step 4:** Добавить строку `HF_TOKEN=hf_...` в `D:\AI_LLM\.env`
  (руками или прислать в чат).

---

### Task 5: Живой деплой и A/B-прослушка (существующий инстанс 43664615)

Деньги: ~$0.52/час с момента `up`. Первая установка fish + веса ≈ 10–15 минут.

- [ ] **Step 1:** `uv run python scripts/vast.py up --write-config` — дождаться URL
  (кэш цел, старт 2–3 мин; поднимется по СТАРОМУ onstart в режиме xtts — это норма).
- [ ] **Step 2:** Забрать накопленный фидбек первого дня (data/ в .gitignore):

```powershell
scp -i "$env:USERPROFILE\.ssh\nova_vast" -P <ssh_port> root@<ssh_host>:/workspace/NOVA/data/feedback.jsonl D:\AI_LLM\data\feedback-day1.jsonl
```

- [ ] **Step 3:** Залить HF-токен на инстанс:

```powershell
ssh -i "$env:USERPROFILE\.ssh\nova_vast" -p <ssh_port> root@<ssh_host> "cat > /workspace/hf_token" < токен-через-echo
```

(практично: `ssh ... "echo hf_XXXX > /workspace/hf_token"`)

- [ ] **Step 4:** Сгенерировать транскрипт референса whisper'ом на инстансе:

```powershell
ssh -i "$env:USERPROFILE\.ssh\nova_vast" -p <ssh_port> root@<ssh_host> "cd /workspace/NOVA && export LD_LIBRARY_PATH=\$(python3 -c 'import nvidia.cudnn; print(list(nvidia.cudnn.__path__)[0] + \"/lib\")') && python3 -c \"from faster_whisper import WhisperModel; m = WhisperModel('large-v3-turbo', device='cuda', compute_type='int8_float16'); segs, _ = m.transcribe('personas/nova/voice_sample.wav', language='ru'); print(' '.join(s.text.strip() for s in segs))\""
```

Вывод записать в `personas/nova/voice_sample.txt` (поправив явные ошибки на слух,
если есть), затем:

```bash
git add personas/nova/voice_sample.txt
git commit -m "feat: reference transcript for voice cloning"
git push origin master
```

- [ ] **Step 5:** Провижининг fish и перезапуск в режиме fish:

```powershell
ssh -i "$env:USERPROFILE\.ssh\nova_vast" -p <ssh_port> root@<ssh_host> "cd /workspace/NOVA && git pull -q; (setsid nohup bash deploy/onstart.sh > /var/log/voice2.log 2>&1 < /dev/null &); echo started"
```

Следить: `tail /workspace/fishpip.log /workspace/hfdl.log /workspace/fish.log`,
готовность — `http://<ip>:<port>/health` отвечает.

- [ ] **Step 6:** Клиент → слушать fish-голос: короткие и длинные фразы,
  диалог 5+ реплик. Отметить: живость, акцент, «проглоченные буквы», помехи.
- [ ] **Step 7:** Переключить обратно на XTTS для сравнения:

```powershell
ssh ... "cd /workspace/NOVA && NOVA_TTS=xtts bash deploy/runner.sh"
```

(и назад: `NOVA_TTS=fish bash deploy/runner.sh`). Пользователь выносит вердикт.

- [ ] **Step 8:** `uv run python scripts/vast.py down` после прослушки.

---

### Task 6: Зафиксировать вердикт

- [ ] **Step 1:** В `docs/STATUS.md` обновить раздел «Голос 2.0»: итог A/B,
  выбранный движок по умолчанию (если fish победил — оставить дефолт
  `NOVA_TTS=fish` в runner; если нет — план Б: дообучение S1-mini на
  дорожках Миты или облачный API fish.audio, модель
  `6dc11f3f67a543f6ad4537a4a347e224`).
- [ ] **Step 2:** README: упомянуть переключатель `NOVA_TTS`.
- [ ] **Step 3: Commit**

```bash
git add -A
git commit -m "docs: voice engine verdict and switch documentation"
git push origin master
```

---

## Вне скоупа

- Дообучение S1-mini/XTTS на 35 дорожках — отдельный план, если зеро-шот
  не устроит.
- Этап 3 (память/RAG/веб) — следующий большой план после голоса.
