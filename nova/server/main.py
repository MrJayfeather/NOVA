import os
import time
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


def build_models(mock: bool, persona_prompt: str):
    if mock:
        return MockASR(), MockLLM(persona_prompt=persona_prompt), MockTTS()
    from nova.server.models.gemini_vision import wrap_eyes
    from nova.server.models.qwen_llm import QwenVLM
    from nova.server.models.whisper_asr import WhisperASR
    from nova.server.models.xtts_tts import XttsTTS
    from nova.server.tts_text import speech_matches, strip_markers

    asr = WhisperASR(model_name=os.environ.get("NOVA_WHISPER", "large-v3-turbo"))
    llm = wrap_eyes(QwenVLM(
        persona_prompt=persona_prompt,
        base_url=os.environ.get("NOVA_VLLM_URL", "http://127.0.0.1:5000/v1"),
        model=os.environ.get("NOVA_MODEL", "Qwen/Qwen3.6-27B-FP8"),
    ))
    persona = os.environ.get("NOVA_PERSONA", "nova")
    ref_dir = Path("personas") / persona
    ref_txt = ref_dir / "voice_sample.txt"
    mode = os.environ.get("NOVA_TTS", "xtts")

    async def _guard(sentence: str, pcm: bytes, rate: int) -> bool:
        # СТТ-страж от робо-заскоков: сверяем синтез с текстом; при
        # любой ошибке стража голос важнее — пропускаем как есть
        if os.environ.get("NOVA_TTS_GUARD", "1") != "1":
            return True
        try:
            heard = await asr.transcribe(pcm, rate)
        except Exception as exc:
            print(f"[nova] страж синтеза недоступен: {exc!r}")
            return True
        return speech_matches(strip_markers(sentence), heard)

    if mode == "voxcpm" and ref_txt.exists():
        from nova.server.models.vox_tts import build_vox_tts

        tts = build_vox_tts(asr, ref_dir, validator=_guard)
    elif mode == "fishcloud" and os.environ.get("NOVA_FISH_KEY"):
        from nova.server.models.fish_tts import FishTTS

        tts = FishTTS(
            validator=_guard,
            url="https://api.fish.audio/v1/tts",
            api_key=os.environ["NOVA_FISH_KEY"],
            model=os.environ.get("NOVA_FISH_MODEL", "s2.1-pro-free"),
            reference_id=os.environ.get(
                "NOVA_FISH_REF_ID", "4075192824f64dc6aabbbf70124d6a01"),
            # затор free-очереди: лучше быстро пропустить предложение,
            # чем молчать минуту
            timeout=35.0,
            # ниже температура — стабильнее голос (меньше «выпадений из
            # роли»), живость добирается эмоциональными ремарками
            temperature=float(os.environ.get("NOVA_FISH_TEMP", "0.5")),
            top_p=float(os.environ.get("NOVA_FISH_TOP_P", "0.6")),
        )
    elif mode in ("fish", "fishcloud", "voxcpm") and ref_txt.exists():
        if mode != "fish":
            print(f"[nova] {mode} недоступен — откатываюсь на локальный fish")
        from nova.server.models.fish_tts import FishTTS

        tts = FishTTS(
            url=os.environ.get("NOVA_FISH_URL", "http://127.0.0.1:8081/v1/tts"),
            reference_wav=ref_dir / "voice_sample.wav",
            reference_text=ref_txt.read_text(encoding="utf-8").strip(),
        )
    else:
        if mode != "xtts":
            print("[nova] нет voice_sample.txt — откатываюсь на xtts")
        tts = XttsTTS(speaker_wav=ref_dir / "voice_sample.wav")
    return asr, llm, tts


def create_app(
    mock: bool = True,
    profiles_root: Path = Path("profiles"),
    personas_root: Path = Path("personas"),
    feedback_path: Path = Path("data/feedback.jsonl"),
    token: str = "",
) -> FastAPI:
    app = FastAPI(title="NOVA server")
    persona = os.environ.get("NOVA_PERSONA", "nova")
    persona_prompt = load_persona_prompt(persona, personas_root)
    asr, llm, tts = build_models(mock, persona_prompt)
    app.state.clients = 0
    app.state.last_activity = time.time()

    @app.get("/health")
    def health():
        return {
            "clients": app.state.clients,
            "idle_s": round(time.time() - app.state.last_activity, 1),
        }

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
        if token and first.token != token:
            await ws.close(code=4002)
            return

        profile = load_profile(first.profile, profiles_root)
        engine = ProactiveEngine(
            cooldown_s=profile.proactive.cooldown_s,
            talkativeness=profile.proactive.talkativeness,
            dedupe_window_s=profile.proactive.dedupe_window_s,
        )

        async def send(msg):
            await ws.send_text(dump_message(msg))

        session = Session(
            send=send, engine=engine, asr=asr, llm=llm, tts=tts,
            feedback_path=feedback_path,
        )
        await send(HelloAck(mock=mock))
        app.state.clients += 1
        app.state.last_activity = time.time()
        try:
            while True:
                msg = parse_client_message(await ws.receive_text())
                app.state.last_activity = time.time()
                await session.handle(msg)
        except WebSocketDisconnect:
            pass
        finally:
            app.state.clients -= 1
            app.state.last_activity = time.time()

    return app


if __name__ == "__main__":
    import uvicorn

    mock = os.environ.get("NOVA_MOCK", "1") == "1"
    uvicorn.run(
        create_app(mock=mock, token=os.environ.get("NOVA_TOKEN", "")),
        host="0.0.0.0", port=8000,
        ws_ping_interval=20, ws_ping_timeout=60,
    )
