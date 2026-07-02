# NOVA — персональный ИИ-компаньон

Этап 1: скелет (mock-модели, локально). Спека: `docs/superpowers/specs/2026-07-03-nova-companion-design.md`.

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

## Конфиги

- `client_config.yaml` — адрес сервера, профиль, хоткеи.
- `profiles/*.yaml` — чувствительность детектора, болтливость, кулдаун.
- `personas/nova/system_prompt.md` — характер NOVA (этап 2+).
