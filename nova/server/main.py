import os
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from pydantic import ValidationError

from nova.server.models.mock import MockASR, MockLLM, MockTTS
from nova.server.orchestrator import Session
from nova.server.proactive import ProactiveEngine
from nova.shared.profiles import load_persona_prompt, load_profile
from nova.shared.protocol import (
    PROTOCOL_VERSION, Hello, HelloAck, dump_message, parse_client_message,
)


def create_app(
    mock: bool = True,
    profiles_root: Path = Path("profiles"),
    personas_root: Path = Path("personas"),
    feedback_path: Path = Path("data/feedback.jsonl"),
) -> FastAPI:
    if not mock:
        raise NotImplementedError("Реальные модели — этап 2; сейчас только NOVA_MOCK=1")
    app = FastAPI(title="NOVA server")

    @app.websocket("/ws")
    async def ws_endpoint(ws: WebSocket):
        await ws.accept()
        try:
            first = parse_client_message(await ws.receive_text())
        except ValidationError:
            await ws.close(code=4000)
            return
        if not isinstance(first, Hello):
            await ws.close(code=4000)
            return
        if first.protocol != PROTOCOL_VERSION:
            await ws.close(code=4001)
            return

        profile = load_profile(first.profile, profiles_root)
        persona_prompt = load_persona_prompt(first.persona, personas_root)
        engine = ProactiveEngine(
            cooldown_s=profile.proactive.cooldown_s,
            talkativeness=profile.proactive.talkativeness,
            dedupe_window_s=profile.proactive.dedupe_window_s,
        )

        async def send(msg):
            await ws.send_text(dump_message(msg))

        session = Session(
            send=send,
            engine=engine,
            asr=MockASR(),
            llm=MockLLM(persona_prompt=persona_prompt),
            tts=MockTTS(),
            feedback_path=feedback_path,
        )
        await send(HelloAck(mock=True))
        try:
            while True:
                msg = parse_client_message(await ws.receive_text())
                await session.handle(msg)
        except WebSocketDisconnect:
            pass

    return app


if __name__ == "__main__":
    import uvicorn

    mock = os.environ.get("NOVA_MOCK", "1") == "1"
    uvicorn.run(create_app(mock=mock), host="0.0.0.0", port=8000)
