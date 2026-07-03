import base64

import httpx

from nova.server.models.base import NO_COMMENT, VisionLLM

COMMENT_INSTRUCTION = (
    "РЎРѕР±С‹С‚РёРµ РЅР° СЌРєСЂР°РЅРµ: {event}. РџРѕСЃРјРѕС‚СЂРё РЅР° РєР°РґСЂС‹ Рё, РµСЃР»Рё С‚Р°Рј РµСЃС‚СЊ С‡С‚Рѕ-С‚Рѕ, "
    "С‡С‚Рѕ СЃС‚РѕРёС‚ РїСЂРѕРєРѕРјРјРµРЅС‚РёСЂРѕРІР°С‚СЊ РІ С‚РІРѕС‘Рј СЃС‚РёР»Рµ, РґР°Р№ РєРѕСЂРѕС‚РєСѓСЋ Р¶РёРІСѓСЋ СЂРµРїР»РёРєСѓ "
    "(1вЂ“2 РїСЂРµРґР»РѕР¶РµРЅРёСЏ). Р•СЃР»Рё РЅРёС‡РµРіРѕ РёРЅС‚РµСЂРµСЃРЅРѕРіРѕ РЅРµС‚ РёР»Рё С‚С‹ СЌС‚Рѕ СѓР¶Рµ "
    "РєРѕРјРјРµРЅС‚РёСЂРѕРІР°Р»Р° вЂ” РѕС‚РІРµС‚СЊ СЂРѕРІРЅРѕ: " + NO_COMMENT
)


def trim_to_sentence(text: str) -> str:
    """РћР±СЂРµР·РєР° РїРѕ Р»РёРјРёС‚Сѓ С‚РѕРєРµРЅРѕРІ СЂРІС‘С‚ С„СЂР°Р·Сѓ РЅР° РїРѕР»СѓСЃР»РѕРІРµ вЂ” РЅРµ РѕР·РІСѓС‡РёРІР°РµРј РѕРіСЂС‹Р·РѕРє."""
    t = text.strip()
    m = max(t.rfind("."), t.rfind("!"), t.rfind("?"), t.rfind("вЂ¦"))
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
            # СЃРІРµР¶РёРµ РєР°РґСЂС‹ СЌРєСЂР°РЅР°, С‡С‚РѕР±С‹ РѕРЅР° РѕС‚РІРµС‡Р°Р»Р° РїРѕ С‚РѕРјСѓ, С‡С‚Рѕ СЂРµР°Р»СЊРЅРѕ РІРёРґРёС‚
            parts = [_image_part(f) for f in frames[-2:]]
            parts.append({"type": "text", "text": text})
            content = parts
        return [
            {"role": "system", "content": self._persona},
            *history[-100:],
            {"role": "user", "content": content},
        ]

    def build_comment_messages(
        self, event: str, frames: list[bytes], history: list[dict]
    ) -> list[dict]:
        content = [_image_part(f) for f in frames[-8:]]
        content.append({"type": "text", "text": COMMENT_INSTRUCTION.format(event=event)})
        return [
            {"role": "system", "content": self._persona},
            *history[-100:],
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
                # С€С‚СЂР°С„С‹ РїСЂРѕС‚РёРІ СЃР°РјРѕРєРѕРїРёСЂРѕРІР°РЅРёСЏ (В«С†РµР»Р°СЏ РїР°Р»РёС‚СЂР°В» С€РµСЃС‚СЊ СЂР°Р· РїРѕРґСЂСЏРґ)
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
