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
    AudioSegment, CinemaMode, Clip, DetectorEvent, Frame, Hello, Hotkey,
    SpeakEnd, SpeakStart,
)


async def capture_loop(source, detector, burst, conn, cfg: ClientConfig,
                       iterations: int | None = None,
                       state: dict | None = None,
                       gate=None):
    period = 1.0 / cfg.periodic_fps
    last_periodic = 0.0
    last_event = 0.0
    last_clip_frame = 0.0
    i = 0
    while iterations is None or i < iterations:
        i += 1
        item = await asyncio.to_thread(source.get)
        if item is None:
            continue
        ts, jpeg, gray_small, (cursor_x, cursor_y) = item
        if state is not None:
            # свежайший кадр всегда под рукой: уйдёт на сервер вместе с
            # репликой пользователя (глаза видят экран НА МОМЕНТ вопроса,
            # а не последнее событие детектора)
            state["last_frame"] = (ts, jpeg, cursor_x, cursor_y)
        event = detector.process(gray_small, ts)
        if event and gate is not None:
            gate.note_event(ts)
        motion = gate is not None and gate.is_motion(ts) and state is not None
        if motion and ts - last_clip_frame >= 1.0 / cfg.clip_fps:
            # движуха: кадры копятся в клип (jpeg-пачка для mjpeg-пайпа)
            state.setdefault("clip_frames", []).append(jpeg)
            last_clip_frame = ts
        if event and not burst.active and ts - last_event >= cfg.event_cooldown_s:
            last_event = ts
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
        elif not motion and ts - last_periodic >= period:
            # периодика молчит в MOTION — её заменяют клипы
            conn.send_frame(Frame(
                ts=ts, jpeg_b64=base64.b64encode(jpeg).decode(),
                cursor_x=cursor_x, cursor_y=cursor_y,
            ))
            last_periodic = ts


def apply_cinema(gate, recorder, on: bool, via: str, audio: bool) -> None:
    """Кино-режим: принудительный видео-взгляд (+звук системы)."""
    gate.set_cinema(on)
    print(f"[nova] кино-режим: {'ВКЛ' if on else 'ВЫКЛ'} ({via})")
    if on and audio:
        recorder.start()
    elif not on:
        recorder.stop()


async def clip_loop(state: dict, gate, recorder, sender, conn,
                    cfg: ClientConfig, iterations: int | None = None):
    """Каждые clip_s секунд собирает накопленные кадры в mp4 и отдаёт
    отправщику; при STILL чистит буфер."""
    import tempfile
    from pathlib import Path

    from nova.client.clip import encode_clip

    i = 0
    while iterations is None or i < iterations:
        i += 1
        await asyncio.sleep(cfg.clip_s)
        frames = state.pop("clip_frames", [])
        now = time.time()
        if not gate.is_motion(now) or len(frames) < 2:
            continue
        wav = recorder.drain() if (gate.cinema and cfg.clip_audio) else None
        out = tempfile.mktemp(suffix=".mp4")
        ok = await asyncio.to_thread(
            encode_clip, frames, cfg.clip_fps, wav, out,
            cfg.clip_max_w, cfg.clip_crf)
        if wav:
            Path(wav).unlink(missing_ok=True)
        if not ok:
            continue
        mp4 = Path(out).read_bytes()
        Path(out).unlink(missing_ok=True)
        if len(mp4) > 19_000_000:
            # inline-лимит Gemini ~20МБ: редкий сверхжирный клип пропускаем
            print(f"[nova] клип {len(mp4)//1_000_000}МБ — слишком жирный, пропуск")
            continue
        sender.offer(Clip(ts=now, mp4_b64=base64.b64encode(mp4).decode(),
                          dur_s=cfg.clip_s, audio=wav is not None))


def make_on_message(player: Player, metrics: Metrics, state: dict):
    def on_message(msg) -> None:
        if isinstance(msg, CinemaMode):
            fn = state.get("apply_cinema")
            if fn:
                fn(msg.on, "голосом")
            return
        if isinstance(msg, SpeakStart):
            latency = time.time() - state.get("last_event_ts", time.time())
            metrics.log("speak_latency", latency_s=round(latency, 3), reason=msg.reason)
            if msg.heard:
                print(f"[ты (как она услышала)]: {msg.heard}")
            print(f"[NOVA:{msg.reason}] {msg.text}")
            state["speaking"] = True
        elif isinstance(msg, SpeakEnd):
            state["speaking"] = False
            # ещё немного не слушаем — хвост из колонок затихает
            state["deaf_until"] = time.time() + 1.0
        player.handle(msg)

    return on_message


async def audio_in_loop(conn, source, state: dict,
                        iterations: int | None = None):
    i = 0
    while iterations is None or i < iterations:
        i += 1
        segment = await asyncio.to_thread(source.get)
        if segment is None:
            continue
        # не слушаем себя, пока NOVA говорит (иначе звуковая петля)
        if state.get("speaking") or time.time() < state.get("deaf_until", 0):
            continue
        state["last_event_ts"] = time.time()
        last = state.get("last_frame")
        if last:
            # свежий кадр — ПЕРЕД репликой (websocket сохраняет порядок):
            # ответ мозга опирается на экран в момент вопроса
            _, jpeg, cx, cy = last
            conn.send_frame(Frame(
                ts=time.time(), jpeg_b64=base64.b64encode(jpeg).decode(),
                cursor_x=cx, cursor_y=cy,
            ))
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
        elif action == "cinema":
            fn = state.get("apply_cinema")
            if fn:
                gate = state.get("gate")
                fn(not (gate.cinema if gate else False), "хоткей")
        else:
            if action == "comment_now":
                state["last_event_ts"] = time.time()
            if action == "feedback_up":
                print("[nova] фидбек: 👍 записан (за последнюю её реплику)")
            elif action == "feedback_down":
                print("[nova] фидбек: 👎 записан (за последнюю её реплику)")
            conn.send(Hotkey(action=action_map.get(action, action)))


def to_pynput_combo(combo: str) -> str:
    """"ctrl+alt+m" -> "<ctrl>+<alt>+m" (формат pynput GlobalHotKeys)."""
    parts = []
    for token in combo.lower().split("+"):
        token = token.strip()
        parts.append(token if len(token) == 1 else f"<{token}>")
    return "+".join(parts)


# на русской раскладке клавиша M отдаёт «ь» — латинский бинд глохнет;
# регистрируем буквенные хоткеи в обеих раскладках (QWERTY -> ЙЦУКЕН)
_RU_KEYS = dict(zip("qwertyuiopasdfghjklzxcvbnm", "йцукенгшщзфывапролдячсмить"))


def layout_variants(combo: str) -> list[str]:
    latin = to_pynput_combo(combo)
    ru_parts = []
    changed = False
    for token in combo.lower().split("+"):
        token = token.strip()
        if len(token) == 1 and token in _RU_KEYS:
            ru_parts.append(_RU_KEYS[token])
            changed = True
        else:
            ru_parts.append(token if len(token) == 1 else f"<{token}>")
    return [latin, "+".join(ru_parts)] if changed else [latin]


def register_hotkeys(cfg: ClientConfig, loop, actions: asyncio.Queue) -> None:
    from pynput import keyboard as pk

    mapping = {}
    for action, combo in cfg.hotkeys.items():
        for variant in layout_variants(combo):
            mapping[variant] = (
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
    def on_disconnect():
        # обрыв мог съесть SpeakEnd — размораживаем микрофон
        state["speaking"] = False
        state["deaf_until"] = 0.0

    conn = Connection(
        cfg.server_url,
        on_message=make_on_message(player, metrics, state),
        hello=Hello(profile=cfg.profile, persona=cfg.persona, token=cfg.token),
        on_disconnect=on_disconnect,
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

    # со-просмотр: движуха -> клипы (рубильник cfg.cowatch)
    gate = None
    if cfg.cowatch:
        from nova.client.clip import ClipSender, LoopbackRecorder
        from nova.client.motion import MotionGate

        gate = MotionGate(on_events=cfg.motion_on,
                          off_silence_s=cfg.motion_off)
        recorder = LoopbackRecorder()
        sender = ClipSender(ConnAdapter, kbps=cfg.clip_kbps)
        state["gate"] = gate
        state["apply_cinema"] = (
            lambda on, via: apply_cinema(gate, recorder, on, via,
                                         cfg.clip_audio))

    coros = [
        conn.run(),
        capture_loop(frame_source, detector, BurstCollector(cfg.burst_frames), ConnAdapter, cfg,
                     state=state, gate=gate),
        hotkey_loop(ConnAdapter, player, actions, state),
    ]
    if gate is not None:
        coros.append(clip_loop(state, gate, recorder, sender, ConnAdapter, cfg))
        coros.append(sender.pump_loop())
    if audio_source is not None:
        coros.append(audio_in_loop(ConnAdapter, audio_source, state))
    await asyncio.gather(*coros)


def main() -> None:
    asyncio.run(amain())


if __name__ == "__main__":
    main()
