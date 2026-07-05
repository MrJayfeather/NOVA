class MotionGate:
    """Автомат взгляда: статика -> кадры (STILL), движуха -> клипы
    (MOTION). Кино-режим (хоткей/голос) — принудительный MOTION."""

    def __init__(self, on_events: int = 3, on_window_s: float = 30.0,
                 off_silence_s: float = 60.0):
        self._on_events = on_events
        self._on_window_s = on_window_s
        self._off_silence_s = off_silence_s
        self._events: list[float] = []
        self._motion_since: float | None = None
        self._last_event = 0.0
        self.cinema = False

    def set_cinema(self, on: bool) -> None:
        self.cinema = on

    def note_event(self, ts: float) -> None:
        self._last_event = ts
        self._events = [t for t in self._events
                        if ts - t <= self._on_window_s]
        self._events.append(ts)
        if len(self._events) >= self._on_events:
            self._motion_since = ts

    def is_motion(self, ts: float) -> bool:
        if self.cinema:
            return True
        if self._motion_since is None:
            return False
        if ts - self._last_event > self._off_silence_s:
            self._motion_since = None
            self._events.clear()
            return False
        return True
