from pathlib import Path

import yaml
from pydantic import BaseModel


class DetectorConfig(BaseModel):
    motion_threshold: float = 12.0
    scene_threshold: float = 40.0


class ProactiveConfig(BaseModel):
    cooldown_s: float = 20.0
    talkativeness: float = 0.6
    dedupe_window_s: float = 45.0


class ProfileConfig(BaseModel):
    detector: DetectorConfig = DetectorConfig()
    proactive: ProactiveConfig = ProactiveConfig()


def load_profile(name: str, root: Path) -> ProfileConfig:
    data = yaml.safe_load((root / f"{name}.yaml").read_text(encoding="utf-8"))
    return ProfileConfig.model_validate(data or {})


def load_persona_prompt(name: str, root: Path) -> str:
    return (root / name / "system_prompt.md").read_text(encoding="utf-8")
