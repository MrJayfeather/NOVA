import base64
import hashlib

import httpx

from nova.server.models.base import VisionLLM

GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta"
BAD_SCREEN = "экран виден плохо"

DESCRIBE_PROMPT = (
    "Это кадры экрана пользователя по порядку, последний — самый свежий. "
    "Опиши каждый кадр одной короткой строкой по-русски, только факты: "
    "какие программы/игра/текст/действие видно. Формат строки: «N: ...». "
    "Если кадр почти не отличается от предыдущего — пиши «N: то же». {prev}"
)

CLIP_PROMPT = (
    "Это видеоклип с экрана пользователя (движуха/кино-режим). Разбери "
    "по таймштампам: кто что сделал, что произошло, ЧТО СКАЗАЛИ (если "
    "есть звук — реплики важны). По-русски, фактами, 3-6 строк.\n"
    "СТРОГО: 1) Иконки и панели интерфейса — ЧИТАЙ как информацию "
    "(составы команд, баны, счёт, киллфид), но персонаж с иконки НЕ "
    "действует в бою, пока не виден в кадре сам. 2) Героев узнавай "
    "по облику УВЕРЕННО и называй конкретно (это известные персонажи "
    "— узнавай их); сомневаешься между двумя — назови наиболее "
    "вероятного; но НЕ упоминай тех, чьего облика в кадре нет вообще. "
    "3) Цифры интерфейса подписывай их смыслом, если он ясен из "
    "контекста, иначе просто «число N у прицела/вверху». {hint}"
)

QUESTION_PROMPT = (
    "Это кадр экрана пользователя, снятый В МОМЕНТ его вопроса — текущее "
    "состояние, прямо сейчас. Пользователь спросил: «{q}». Опиши по-русски "
    "в 1-3 строках, что видно, ОСОБЕННО всё, что относится к вопросу — "
    "даты, время, цифры, надписи, индикаторы читай точно, как написано. "
    "Только факты, без домыслов."
)


class GeminiEyes(VisionLLM):
    """Облачные глаза: Gemini описывает кадры текстом, мозг получает
    описания вместо картинок — контекст в ~20 раз дешевле, KV-кэшу
    хватает util 0.70 (см. спеку eyes-cloud-voice3)."""

    def __init__(self, inner: VisionLLM, api_key: str,
                 model: str = "gemini-3.1-flash-lite",
                 timeout: float = 20.0, max_frames: int = 4,
                 on_seen=None):
        self._inner = inner
        self._model = model
        self._max_frames = max_frames
        self._cache: dict[str, str] = {}   # sha1(jpeg) -> строка описания
        self._last_summary = ""
        # хук летописи: каждое СВЕЖЕЕ описание уходит в память ([видела])
        self.on_seen = on_seen
        self._client = httpx.AsyncClient(
            base_url=GEMINI_URL, timeout=timeout,
            headers={"x-goog-api-key": api_key})

    async def _call_gemini(self, frames: list[bytes], prompt: str,
                           video: bytes | None = None) -> str:
        parts = [{"inline_data": {"mime_type": "image/jpeg",
                                  "data": base64.b64encode(f).decode()}}
                 for f in frames]
        if video is not None:
            parts.append({"inline_data": {
                "mime_type": "video/mp4",
                "data": base64.b64encode(video).decode()}})
        parts.append({"text": prompt})
        r = await self._client.post(
            f"/models/{self._model}:generateContent",
            json={"contents": [{"parts": parts}]})
        r.raise_for_status()
        return r.json()["candidates"][0]["content"]["parts"][0]["text"].strip()

    async def describe(self, frames: list[bytes]) -> str:
        frames = frames[-self._max_frames:]
        if not frames:
            return ""
        keys = [hashlib.sha1(f).hexdigest() for f in frames]
        fresh = [(k, f) for k, f in zip(keys, frames) if k not in self._cache]
        if fresh:
            prev = (f"Прошлая сводка: {self._last_summary}"
                    if self._last_summary else "")
            try:
                text = await self._call_gemini(
                    [f for _, f in fresh], DESCRIBE_PROMPT.format(prev=prev))
            except Exception as exc:
                # цензура/лимит/сеть: честная заглушка, НЕ выдумываем
                print(f"[nova] глаза-облако недоступны: {exc!r}")
                text = ""
            lines = [l.split(":", 1)[-1].strip()
                     for l in text.splitlines() if l.strip()]
            if self.on_seen and lines:
                self.on_seen("; ".join(lines))
            for (k, _), line in zip(fresh, lines):
                self._cache[k] = line
            for k, _ in fresh:
                self._cache.setdefault(k, BAD_SCREEN)
            # кадры уходят из deque клиента — чистим и кэш
            while len(self._cache) > 64:
                del self._cache[next(iter(self._cache))]
        desc = "; ".join(self._cache[k] for k in keys)
        self._last_summary = desc
        return desc

    async def describe_for_question(self, frames: list[bytes],
                                    question: str) -> str:
        """Прицельное описание под вопрос пользователя (мимо кэша: общая
        сводка не заметит дату в углу, если о ней спрашивают)."""
        if not frames:
            return ""
        try:
            desc = await self._call_gemini(
                frames, QUESTION_PROMPT.format(q=question[:300]))
            # диагностика «глаза наврали или мозг?»: вставка — в лог
            print(f"[nova] экран->мозг: {desc[:160]!r}")
            if self.on_seen and desc:
                self.on_seen(desc)
            return desc
        except Exception as exc:
            print(f"[nova] глаза-облако недоступны: {exc!r}")
            return BAD_SCREEN

    async def complete_text(self, prompt: str) -> str:
        """Текстовый вызов без кадров — резервный конденсер памяти."""
        return await self._call_gemini([], prompt)

    async def describe_clip(self, mp4: bytes, hint: str = "") -> str:
        """Видео-взгляд: клип экрана -> сводка с таймштампами (и звуком)."""
        try:
            out = await self._call_gemini(
                [], CLIP_PROMPT.format(hint=hint), video=mp4)
        except Exception as exc:
            print(f"[nova] глаза-видео недоступны: {exc!r}")
            return ""
        if self.on_seen and out:
            self.on_seen(out)
        return out

    async def reply_to_user(
        self, text: str, frames: list[bytes], history: list[dict]
    ) -> str:
        if frames:
            # ТОЛЬКО свежайший кадр (клиент шлёт его вместе с репликой):
            # старый рядом лишь путает, когда на них противоречивые факты
            desc = await self.describe_for_question(frames[-1:], text)
            text = (f"[экран СЕЙЧАС, на момент вопроса — это главнее твоих "
                    f"прошлых ответов: {desc}]\n{text}")
        return await self._inner.reply_to_user(text, [], history)

    async def comment_on_event(
        self, event: str, frames: list[bytes], history: list[dict]
    ) -> str:
        desc = await self.describe(frames) or "кадров нет"
        return await self._inner.comment_on_event(
            f"{event}; на экране: {desc}", [], history)


def wrap_eyes(inner: VisionLLM, env: dict | None = None) -> VisionLLM:
    """NOVA_EYES=gemini (дефолт при наличии ключа) | local (как раньше)."""
    if env is None:
        import os

        env = dict(os.environ)
    if env.get("NOVA_EYES", "gemini") != "gemini" or not env.get("GEMINI_KEY"):
        if env.get("NOVA_EYES", "gemini") == "gemini":
            print("[nova] нет GEMINI_KEY — глаза остаются локальными")
        return inner
    return GeminiEyes(
        inner, api_key=env["GEMINI_KEY"],
        model=env.get("NOVA_GEMINI_MODEL", "gemini-3.1-flash-lite"))
