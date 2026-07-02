import uuid

import numpy as np


class FrameDetector:
    def __init__(self, motion_threshold: float, scene_threshold: float):
        self._motion_threshold = motion_threshold
        self._scene_threshold = scene_threshold
        self._prev: np.ndarray | None = None

    def process(self, gray_small: np.ndarray, ts: float) -> str | None:
        prev, self._prev = self._prev, gray_small
        if prev is None:
            return None
        diff = float(np.mean(np.abs(gray_small.astype(np.int16) - prev.astype(np.int16))))
        if diff >= self._scene_threshold:
            return "scene_change"
        if diff >= self._motion_threshold:
            return "motion_burst"
        return None


class BurstCollector:
    def __init__(self, size: int):
        self._size = size
        self._frames: list[bytes] | None = None
        self.burst_id: str = ""

    @property
    def active(self) -> bool:
        return self._frames is not None

    def start(self) -> str:
        self._frames = []
        self.burst_id = uuid.uuid4().hex[:8]
        return self.burst_id

    def add(self, jpeg: bytes) -> list[bytes] | None:
        if self._frames is None:
            return None
        self._frames.append(jpeg)
        if len(self._frames) >= self._size:
            result, self._frames = self._frames, None
            return result
        return None
