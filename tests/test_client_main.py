import json
import time

import numpy as np

from nova.client.audio_out import Player, StreamSink
from nova.client.config import ClientConfig
from nova.client.detector import BurstCollector, FrameDetector
from nova.client.main import Metrics, capture_loop, make_on_message, to_pynput_combo
from nova.shared.protocol import DetectorEvent, Frame, SpeakStart


class FakeSource:
    """Отдаёт кортежи кадров: чёрные, затем белые (смена сцены)."""

    def __init__(self):
        self.items = []
        for i in range(8):
            value = 0 if i < 3 else 255
            gray = np.full((90, 160), value, dtype=np.uint8)
            self.items.append((time.time(), f"jpg{i}".encode(), gray, (5, 6)))

    def get(self):
        return self.items.pop(0) if self.items else None


class FakeConn:
    def __init__(self):
        self.sent = []
        self.frames = []

    def send(self, msg):
        self.sent.append(msg)

    def send_frame(self, msg):
        self.frames.append(msg)


async def test_capture_loop_sends_periodic_event_and_burst():
    cfg = ClientConfig(server_url="ws://x", periodic_fps=100.0, burst_frames=2)
    conn = FakeConn()
    await capture_loop(
        source=FakeSource(),
        detector=FrameDetector(motion_threshold=12.0, scene_threshold=40.0),
        burst=BurstCollector(size=cfg.burst_frames),
        conn=conn,
        cfg=cfg,
        iterations=8,
    )
    periodic = [m for m in conn.frames if isinstance(m, Frame) and m.kind == "periodic"]
    assert periodic and periodic[0].cursor_x == 5 and periodic[0].cursor_y == 6
    events = [m for m in conn.sent if isinstance(m, DetectorEvent)]
    assert any(e.event == "scene_change" for e in events)
    bursts = [m for m in conn.sent if isinstance(m, Frame) and m.kind == "burst"]
    assert len(bursts) == cfg.burst_frames
    assert all(b.burst_id == bursts[0].burst_id for b in bursts)


async def test_capture_loop_tracks_last_frame_in_state():
    cfg = ClientConfig(server_url="ws://x", periodic_fps=100.0, burst_frames=2)
    state = {}
    await capture_loop(
        source=FakeSource(),
        detector=FrameDetector(motion_threshold=12.0, scene_threshold=40.0),
        burst=BurstCollector(size=cfg.burst_frames),
        conn=FakeConn(),
        cfg=cfg,
        iterations=3,
        state=state,
    )
    # в state всегда лежит самый свежий кадр — для «глаз в реальном времени»
    ts, jpeg, cx, cy = state["last_frame"]
    assert jpeg == b"jpg2"
    assert (cx, cy) == (5, 6)


async def test_audio_sends_fresh_frame_before_speech():
    from nova.client.main import audio_in_loop
    from nova.shared.protocol import AudioSegment

    class OneShotAudio:
        def __init__(self):
            self.items = [b"\x01\x00" * 100]

        def get(self):
            return self.items.pop(0) if self.items else None

    conn = FakeConn()
    state = {"last_frame": (123.0, b"freshjpg", 7, 8)}
    await audio_in_loop(conn, OneShotAudio(), state, iterations=2)
    # свежий кадр уходит ПЕРЕД репликой: мозг видит экран на момент вопроса
    assert conn.frames, "кадр не отправлен"
    speech = [m for m in conn.sent if isinstance(m, AudioSegment)]
    assert speech, "реплика не отправлена"


async def test_event_cooldown_limits_detector_storm():
    # видео = смена сцены каждый кадр; кулдаун держит один всплеск
    class StormSource:
        def __init__(self):
            self.i = 0

        def get(self):
            self.i += 1
            if self.i > 8:
                return None
            value = 0 if self.i % 2 else 255   # мигаем чёрное/белое
            gray = np.full((90, 160), value, dtype=np.uint8)
            return (time.time(), b"jpg", gray, (0, 0))

    cfg = ClientConfig(server_url="ws://x", periodic_fps=0.001,
                       burst_frames=1, event_cooldown_s=999.0)
    conn = FakeConn()
    await capture_loop(
        source=StormSource(),
        detector=FrameDetector(motion_threshold=12.0, scene_threshold=40.0),
        burst=BurstCollector(size=1),
        conn=conn,
        cfg=cfg,
        iterations=9,
    )
    events = [m for m in conn.sent if isinstance(m, DetectorEvent)]
    assert len(events) == 1        # шторм ужат до одного события


def test_disconnect_resets_speaking_flag():
    from nova.client.connection import Connection
    from nova.shared.protocol import Hello

    state = {"speaking": True, "deaf_until": time.time() + 99}

    def on_disconnect():
        state["speaking"] = False
        state["deaf_until"] = 0.0

    conn = Connection("ws://x", on_message=lambda m: None,
                      hello=Hello(profile="p", persona="n", token=""),
                      on_disconnect=on_disconnect)
    conn._on_disconnect()           # как в except-ветке run()
    assert state["speaking"] is False
    assert state["deaf_until"] == 0.0


def test_to_pynput_combo():
    assert to_pynput_combo("ctrl+alt+m") == "<ctrl>+<alt>+m"
    assert to_pynput_combo("ctrl+alt+up") == "<ctrl>+<alt>+<up>"
    assert to_pynput_combo("F9") == "<f9>"


def test_on_message_logs_latency_and_prints(tmp_path, capsys):
    class NullSink(StreamSink):
        def start(self, sample_rate):
            pass

        def write(self, pcm):
            pass

        def stop(self):
            pass

    metrics = Metrics(tmp_path / "metrics.jsonl")
    state = {"last_event_ts": time.time() - 0.5}
    handler = make_on_message(Player(NullSink()), metrics, state)
    handler(SpeakStart(utterance_id="u1", text="о, смена сцены", reason="proactive", sample_rate=16000))
    out = capsys.readouterr().out
    assert "о, смена сцены" in out
    rec = json.loads((tmp_path / "metrics.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert rec["kind"] == "speak_latency"
    assert rec["latency_s"] >= 0.5
