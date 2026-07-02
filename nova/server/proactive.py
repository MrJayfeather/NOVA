from dataclasses import dataclass


@dataclass
class Decision:
    speak: bool
    reason: str


class ProactiveEngine:
    """Анти-спам фильтр проактивных комментариев. НЕ цензура инициативы:
    блокирует только повторы (dedupe), слишком частые реплики (cooldown)
    и режим паузы. forced (запрос пользователя) обходит всё."""

    def __init__(self, cooldown_s: float, talkativeness: float, dedupe_window_s: float):
        self._cooldown_s = cooldown_s
        self._talkativeness = max(0.0, min(1.0, talkativeness))
        self._dedupe_window_s = dedupe_window_s
        self._paused = False
        self._last_spoke_at: float | None = None
        self._last_event_times: dict[str, float] = {}

    def set_talkativeness(self, value: float) -> None:
        self._talkativeness = max(0.0, min(1.0, value))

    def toggle_pause(self) -> bool:
        self._paused = not self._paused
        return self._paused

    def _effective_cooldown(self) -> float:
        return self._cooldown_s * (1.75 - 1.5 * self._talkativeness)

    def on_event(self, event: str, now: float, forced: bool = False) -> Decision:
        if forced:
            self._mark(event, now)
            return Decision(True, "forced")
        if self._paused:
            return Decision(False, "paused")
        if (
            self._last_spoke_at is not None
            and now - self._last_spoke_at < self._effective_cooldown()
        ):
            return Decision(False, "cooldown")
        last_same = self._last_event_times.get(event)
        if last_same is not None and now - last_same < self._dedupe_window_s:
            return Decision(False, "dedupe")
        self._mark(event, now)
        return Decision(True, "ok")

    def _mark(self, event: str, now: float) -> None:
        self._last_spoke_at = now
        self._last_event_times[event] = now
