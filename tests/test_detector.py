import numpy as np

from nova.client.detector import BurstCollector, FrameDetector


def frame(value: int) -> np.ndarray:
    return np.full((90, 160), value, dtype=np.uint8)


def test_first_frame_no_event():
    d = FrameDetector(motion_threshold=12.0, scene_threshold=40.0)
    assert d.process(frame(0), ts=0.0) is None


def test_identical_frames_no_event():
    d = FrameDetector(motion_threshold=12.0, scene_threshold=40.0)
    d.process(frame(100), ts=0.0)
    assert d.process(frame(100), ts=1.0) is None


def test_big_change_is_scene_change():
    d = FrameDetector(motion_threshold=12.0, scene_threshold=40.0)
    d.process(frame(0), ts=0.0)
    assert d.process(frame(255), ts=1.0) == "scene_change"


def test_medium_change_is_motion_burst():
    d = FrameDetector(motion_threshold=12.0, scene_threshold=40.0)
    d.process(frame(0), ts=0.0)
    assert d.process(frame(20), ts=1.0) == "motion_burst"


def test_burst_collector_lifecycle():
    b = BurstCollector(size=3)
    assert not b.active
    burst_id = b.start()
    assert b.active and burst_id
    assert b.add(b"f1") is None
    assert b.add(b"f2") is None
    result = b.add(b"f3")
    assert result == [b"f1", b"f2", b"f3"]
    assert not b.active
