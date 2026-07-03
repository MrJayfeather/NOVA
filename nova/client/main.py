import asyncio
import base64
import os
import time
from pathlib import Path

from nova.client.audio_out import Player, SounddeviceStreamSink
from nova.client.audio_worker import ProcessAudioSource
from nova.client.capture_worker import ProcessFrameSource
from nova.client.config import ClientConfig, load_config
from nova.client.connection import Connection
from nova.client.detector import BurstCollector, FrameDetector
from nova.client.metrics import Metrics
from nova.shared.profiles import load_profile
from nova.shared.protocol import (
    AudioSegment, DetectorEvent, Frame, Hello, Hotkey, SpeakStart,
)


async def capture_loop(source, detector, burst, conn, cfg: ClientConfig,
                       iterations: int | None = None):
    period = 1.0 / cfg.periodic_fps
    last_periodic = 0.0
    i = 0
    while iterations is None or i < iterations:
        i += 1
        item = await asyncio.to_thread(source.get)
        if item is None:
            continue
        ts, jpeg, gray_small, (cursor_x, cursor_y) = item
        event = detector.process(gray_small, ts)
        if event and not burst.active:
            conn.send(DetectorEvent(ts=ts, event=event))
            burst.start()
        if burst.active:
            done = burst.add(jpeg)
            if done is not None:
                for seq, j in enumerate(done):
                    conn.send(Frame(
                        ts=ts, jpeg_b64=base64.b64encode(j).decode(),
                        kind="burst", burst_id=burst.burst_id, seq=seq,
                    ))
        elif ts - last_periodic >= period:
            conn.send_frame(Frame(
                ts=ts, jpeg_b64=base64.b64encode(jpeg).decode(),
                cursor_x=cursor_x, cursor_y=cursor_y,
            ))
            last_periodic = ts


def make_on_message(player: Player, metrics: Metrics, state: dict):
    def on_message(msg) -> None:
        if isinstance(msg, SpeakStart):
            latency = time.time() - state.get("last_event_ts", time.time())
            metrics.log("speak_latency", latency_s=round(latency, 3), reason=msg.reason)
            print(f"[NOVA:{msg.reason}] {msg.text}")
        player.handle(msg)

    return on_message


async def audio_in_loop(conn, source, state: dict):
    while True:
        segment = await asyncio.to_thread(source.get)
        if segment is not None:
            state["last_event_ts"] = time.time()
            conn.send(AudioSegment(
                ts=time.time(), pcm_b64=base64.b64encode(segment).decode(), sample_rate=16000,
            ))


async def hotkey_loop(conn, player: Player, actions: asyncio.Queue, state: dict):
    # имена биндов из конфига → действия протокола
    action_map = {"pause": "toggle_pause"}
    while True:
        action = await actions.get()
        if action == "mute":
            player.muted = not player.muted
            print(f"[nova] mute: {player.muted}")
        else:
            if action == "comment_now":
                state["last_event_ts"] = time.time()
            conn.send(Hotkey(action=action_map.get(action, action)))


def to_pynput_combo(combo: str) -> str:
    """"ctrl+alt+m" -> "<ctrl>+<alt>+m" (формат pynput GlobalHotKeys)."""
    parts = []
    for token in combo.lower().split("+"):
        token = token.strip()
        parts.append(token if len(token) == 1 else f"<{token}>")
    return "+".join(parts)


def register_hotkeys(cfg: ClientConfig, loop, actions: asyncio.Queue) -> None:
    from pynput import keyboard as pk

    mapping = {}
    for action, combo in cfg.hotkeys.items():
        mapping[to_pynput_combo(combo)] = (
            lambda a=action: loop.call_soon_threadsafe(actions.put_nowait, a)
        )
    listener = pk.GlobalHotKeys(mapping)
    listener.daemon = True
    listener.start()


async def amain() -> None:
    cfg = load_config(Path("client_config.yaml"))
    profile = load_profile(cfg.profile, Path("profiles"))
    state: dict = {}
    metrics = Metrics(Path("data/metrics.jsonl"))
    player = Player(SounddeviceStreamSink())
    conn = Connection(
        cfg.server_url,
        on_message=make_on_message(player, metrics, state),
        hello=Hello(profile=cfg.profile, persona=cfg.persona, token=cfg.token),
    )
    detector = FrameDetector(
        motion_threshold=profile.detector.motion_threshold,
        scene_threshold=profile.detector.scene_threshold,
    )

    def sending_conn_send(msg):
        if isinstance(msg, DetectorEvent):
            state["last_event_ts"] = time.time()
        conn.send(msg)

    class ConnAdapter:
        send = staticmethod(sending_conn_send)
        send_frame = staticmethod(conn.send_frame)

    frame_source = ProcessFrameSource(jpeg_quality=cfg.jpeg_quality)
    loop = asyncio.get_running_loop()
    if os.environ.get("NOVA_NO_MIC") == "1":
        audio_source = None
        print("[nova] микрофон ОТКЛЮЧЁН (NOVA_NO_MIC=1)")
    else:
        audio_source = ProcessAudioSource()
    actions: asyncio.Queue = asyncio.Queue()
    if os.environ.get("NOVA_NO_HOTKEYS") == "1":
        print("[nova] клиент запущен, хоткеи ОТКЛЮЧЕНЫ (NOVA_NO_HOTKEYS=1)")
    else:
        register_hotkeys(cfg, loop, actions)
        print("[nova] клиент запущен, хоткеи активны")

    coros = [
        conn.run(),
        capture_loop(frame_source, detector, BurstCollector(cfg.burst_frames), ConnAdapter, cfg),
        hotkey_loop(ConnAdapter, player, actions, state),
    ]
    if audio_source is not None:
        coros.append(audio_in_loop(ConnAdapter, audio_source, state))
    await asyncio.gather(*coros)


def main() -> None:
    asyncio.run(amain())


if __name__ == "__main__":
    main()
