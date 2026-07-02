import json
import time

import numpy as np

from nova.client.audio_out import AudioSink, Player
from nova.client.config import ClientConfig
from nova.client.detector import BurstCollector, FrameDetector
from nova.client.main import Metrics, capture_loop, make_on_message
from nova.shared.protocol import DetectorEvent, Frame, SpeakStart


class FakeGrabber:
    """Отдаёт чёрные кадры, затем белые (смена сцены)."""

    def __init__(self):
        self.count = 0

    def grab(self):
        self.count += 1
        value = 0 if self.count <= 3 else 255
        return np.full((90, 160, 3), value, dtype=np.uint8)


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
        grabber=FakeGrabber(),
        detector=FrameDetector(motion_threshold=12.0, scene_threshold=40.0),
        burst=BurstCollector(size=cfg.burst_frames),
        conn=conn,
        cfg=cfg,
        iterations=8,
        sleep_s=0.0,
    )
    assert any(isinstance(m, Frame) and m.kind == "periodic" for m in conn.frames)
    events = [m for m in conn.sent if isinstance(m, DetectorEvent)]
    assert any(e.event == "scene_change" for e in events)
    bursts = [m for m in conn.sent if isinstance(m, Frame) and m.kind == "burst"]
    assert len(bursts) == cfg.burst_frames
    assert all(b.burst_id == bursts[0].burst_id for b in bursts)


def test_on_message_logs_latency_and_prints(tmp_path, capsys):
    class NullSink(AudioSink):
        def play(self, pcm, sample_rate):
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
