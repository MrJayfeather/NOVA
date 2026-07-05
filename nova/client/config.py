from pathlib import Path

import yaml
from pydantic import BaseModel


DEFAULT_HOTKEYS = {
    "mute": "ctrl+alt+m",
    "comment_now": "ctrl+alt+c",
    "pause": "ctrl+alt+p",
    "feedback_up": "ctrl+alt+up",
    "feedback_down": "ctrl+alt+down",
    "cinema": "ctrl+alt+v",
}


class ClientConfig(BaseModel):
    server_url: str
    profile: str = "desktop"
    persona: str = "nova"
    token: str = ""
    periodic_fps: float = 1.0
    burst_frames: int = 6
    jpeg_quality: int = 85
    # минимум секунд между событиями детектора: видео на экране иначе
    # устраивает шторм кадров (обрывы keepalive, очередь реплик)
    event_cooldown_s: float = 4.0
    # со-просмотр (этап 3В): движуха -> клипы вместо кадров
    cowatch: bool = True
    clip_s: float = 15.0
    clip_fps: int = 8
    clip_audio: bool = True
    clip_kbps: int = 1500
    motion_on: int = 3
    motion_off: float = 60.0
    hotkeys: dict[str, str] = DEFAULT_HOTKEYS


def load_config(path: Path) -> ClientConfig:
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return ClientConfig.model_validate(data)
