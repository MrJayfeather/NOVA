from pathlib import Path

import yaml
from pydantic import BaseModel


DEFAULT_HOTKEYS = {
    "mute": "ctrl+alt+m",
    "comment_now": "ctrl+alt+c",
    "pause": "ctrl+alt+p",
    "feedback_up": "ctrl+alt+up",
    "feedback_down": "ctrl+alt+down",
}


class ClientConfig(BaseModel):
    server_url: str
    profile: str = "desktop"
    persona: str = "nova"
    periodic_fps: float = 1.0
    burst_frames: int = 6
    jpeg_quality: int = 85
    hotkeys: dict[str, str] = DEFAULT_HOTKEYS


def load_config(path: Path) -> ClientConfig:
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return ClientConfig.model_validate(data)
