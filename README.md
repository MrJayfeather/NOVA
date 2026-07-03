# NOVA — персональный ИИ-компаньон

Этап 1: скелет (mock-модели, локально). Спека: `docs/specs/2026-07-03-nova-companion-design.md`.

## Запуск (Windows, два терминала)

Сервер:

    uv run python -m nova.server.main

Клиент:

    uv run python -m nova.client.main

## Что должно происходить (mock-режим)

- Клиент печатает `подключено`, захват экрана: dxcam.
- Резко смени картинку на экране (переключи окно на полноэкранное видео) —
  в консоли клиента появится `[NOVA:proactive] (мок) Заметила событие scene_change...`
  и прозвучит бип (mock-TTS).
- Скажи что-нибудь в микрофон — придёт `[NOVA:reply] (мок) Ты сказал...` + бип.
- Хоткеи: Ctrl+Alt+C — «прокомментируй сейчас», Ctrl+Alt+M — mute,
  Ctrl+Alt+P — пауза наблюдения, Ctrl+Alt+↑/↓ — оценка реплики.
- Метрики задержек: `data/metrics.jsonl`; оценки: `data/feedback.jsonl`.

## Тесты

    uv run pytest

## Запуск с реальными моделями (GPU в облаке)

Один раз: положи `VAST_API_KEY` в `.env` (файл в .gitignore).

    # найти и запустить сервер (первый запуск 15–25 минут: качаются модели)
    uv run python scripts/vast.py up --write-config

    # проверить готовность
    uv run python scripts/vast.py status

    # запустить клиент (client_config.yaml уже настроен командой up)
    uv run python -m nova.client.main

    # закончил — остановить (иначе остановится сам через 15 минут простоя)
    uv run python scripts/vast.py down

`down` останавливает инстанс с сохранением диска (хранение ~$0.1–0.2/день,
зато повторный старт за ~2–3 минуты без скачивания моделей).
Полное удаление: `uv run python scripts/vast.py down --destroy`.

Голос NOVA — движок выбирается переменной `NOVA_TTS` на сервере:

- `fish` (по умолчанию на инстансе) — локальный OpenAudio S1-mini, клонирует
  голос по образцу `personas/nova/voice_sample.wav` + `voice_sample.txt`
  (10–30 с чистой записи и её точный транскрипт);
- `fishcloud` — облачный api.fish.audio: ключ в `/workspace/fish_key`
  на инстансе, модель — `NOVA_FISH_MODEL` (по умолчанию `s2.1-pro-free`),
  голос — `NOVA_FISH_REF_ID` (id готовой модели голоса с fish.audio);
- `xtts` — запасной XTTS-v2 по тому же voice_sample.wav.

## Конфиги

- `client_config.yaml` — адрес сервера, профиль, хоткеи.
- `profiles/*.yaml` — чувствительность детектора, болтливость, кулдаун.
- `personas/nova/system_prompt.md` — характер NOVA (этап 2+).
