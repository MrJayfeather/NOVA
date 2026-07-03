import base64

import httpx

from nova.server.models.base import NO_COMMENT, VisionLLM

COMMENT_INSTRUCTION = (
    "Событие на экране: {event}. Посмотри на кадры и, если там есть что-то, "
    "что стоит прокомментировать в твоём стиле, дай короткую живую реплику "
    "(1–2 предложения). Если ничего интересного нет или ты это уже "
    "комментировала — ответь ровно: " + NO_COMMENT
)


def trim_to_sentence(text: str) -> str:
    """Обрезка по лимиту токенов рвёт фразу на полуслове — не озвучиваем огрызок."""
    t = text.strip()
    m = max(t.rfind("."), t.rfind("!"), t.rfind("?"), t.rfind("…"))
    if m >= 20:
        return t[: m + 1]
    return t


def _image_part(jpeg: bytes) -> dict:
    b64 = base64.b64encode(jpeg).decode()
    return {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}


class QwenVLM(VisionLLM):
    def __init__(self, persona_prompt: str, base_url: str, model: str, timeout: float = 60.0):
        self._persona = persona_prompt
        self._model = model
        self._client = httpx.AsyncClient(base_url=base_url, timeout=timeout)

    def build_reply_messages(
        self, text: str, frames: list[bytes], history: list[dict]
    ) -> list[dict]:
        content: object = text
        if frames:
            # свежие кадры экрана, чтобы она отвечала по тому, что реально видит
            parts = [_image_part(f) for f in frames[-2:]]
            parts.append({"type": "text", "text": text})
            content = parts
        return [
            {"role": "system", "content": self._persona},
            *history[-24:],
            {"role": "user", "content": content},
        ]

    def build_comment_messages(
        self, event: str, frames: list[bytes], history: list[dict]
    ) -> list[dict]:
        content = [_image_part(f) for f in frames[-8:]]
        content.append({"type": "text", "text": COMMENT_INSTRUCTION.format(event=event)})
        return [
            {"role": "system", "content": self._persona},
            *history[-24:],
            {"role": "user", "content": content},
        ]

    async def _chat(self, messages: list[dict]) -> str:
        r = await self._client.post(
            "/chat/completions",
            json={
                "model": self._model,
                "messages": messages,
                "max_tokens": 110,
                "temperature": 0.8,
                # штрафы против самокопирования («целая палитра» шесть раз подряд)
                "presence_penalty": 0.8,
                "frequency_penalty": 0.6,
            },
        )
        r.raise_for_status()
        return trim_to_sentence(r.json()["choices"][0]["message"]["content"])

    async def reply_to_user(
        self, text: str, frames: list[bytes], history: list[dict]
    ) -> str:
        return await self._chat(self.build_reply_messages(text, frames, history))

    async def comment_on_event(
        self, event: str, frames: list[bytes], history: list[dict]
    ) -> str:
        return await self._chat(self.build_comment_messages(event, frames, history))
