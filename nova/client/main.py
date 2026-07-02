import asyncio
import base64
import time
from pathlib import Path

from nova.client.audio_in import Microphone, SileroVAD, VADSegmenter
from nova.client.audio_out import Player, SounddeviceSink
from nova.client.capture import Grabber, cursor_pos, encode_jpeg, to_gray_small
from nova.client.config import ClientConfig, load_config
from nova.client.connection import Connection
from nova.client.detector import BurstCollector, FrameDetector
from nova.client.metrics import Metrics
from nova.shared.profiles import load_profile
from nova.shared.protocol import (
    AudioSegment, DetectorEvent, Frame, Hello, Hotkey, SpeakStart,
)


async def capture_loop(grabber, detector, burst, conn, cfg: ClientConfig,
                       iterations: int | None = None, sleep_s: float = 1 / 15):
    period = 1.0 / cfg.periodic_fps
    last_periodic = 0.0
    i = 0
    while iterations is None or i < iterations:
        i += 1
        frame = grabber.grab()
        if frame is None:
            await asyncio.sleep(sleep_s)
            continue
        ts = time.time()
        event = detector.process(to_gray_small(frame), ts)
        if event and not burst.active:
            conn.send(DetectorEvent(ts=ts, event=event))
            burst.start()
        if burst.active:
            done = burst.add(encode_jpeg(frame, cfg.jpeg_quality))
            if done is not None:
                for seq, jpeg in enumerate(done):
                    conn.send(Frame(
                        ts=ts, jpeg_b64=base64.b64encode(jpeg).decode(),
                        kind="burst", burst_id=burst.burst_id, seq=seq,
                    ))
        elif ts - last_periodic >= period:
            x, y = cursor_pos()
            conn.send_frame(Frame(
                ts=ts, jpeg_b64=base64.b64encode(encode_jpeg(frame, cfg.jpeg_quality)).decode(),
                cursor_x=x, cursor_y=y,
            ))
            last_periodic = ts
        await asyncio.sleep(sleep_s)


def make_on_message(player: Player, metrics: Metrics, state: dict):
    def on_message(msg) -> None:
        if isinstance(msg, SpeakStart):
            latency = time.time() - state.get("last_event_ts", time.time())
            metrics.log("speak_latency", latency_s=round(latency, 3), reason=msg.reason)
            print(f"[NOVA:{msg.reason}] {msg.text}")
        player.handle(msg)

    return on_message


async def audio_in_loop(conn, segmenter, mic_queue: asyncio.Queue, state: dict):
    while True:
        chunk = await mic_queue.get()
        segment = segmenter.feed(chunk)
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


def register_hotkeys(cfg: ClientConfig, loop, actions: asyncio.Queue) -> None:
    import keyboard

    for action, combo in cfg.hotkeys.items():
        keyboard.add_hotkey(
            combo,
            lambda a=action: loop.call_soon_threadsafe(actions.put_nowait, a),
        )


async def amain() -> None:
    cfg = load_config(Path("client_config.yaml"))
    profile = load_profile(cfg.profile, Path("profiles"))
    state: dict = {}
    metrics = Metrics(Path("data/metrics.jsonl"))
    player = Player(SounddeviceSink())
    conn = Connection(
        cfg.server_url,
        on_message=make_on_message(player, metrics, state),
        hello=Hello(profile=cfg.profile, persona=cfg.persona),
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

    loop = asyncio.get_running_loop()
    mic_queue: asyncio.Queue = asyncio.Queue()
    Microphone().start(loop, mic_queue)
    actions: asyncio.Queue = asyncio.Queue()
    register_hotkeys(cfg, loop, actions)
    print("[nova] клиент запущен, хоткеи активны")

    await asyncio.gather(
        conn.run(),
        capture_loop(Grabber(), detector, BurstCollector(cfg.burst_frames), ConnAdapter, cfg),
        audio_in_loop(ConnAdapter, VADSegmenter(SileroVAD()), mic_queue, state),
        hotkey_loop(ConnAdapter, player, actions, state),
    )


def main() -> None:
    asyncio.run(amain())


if __name__ == "__main__":
    main()
