from pathlib import Path

from nova.shared.profiles import load_persona_prompt, load_profile

ROOT = Path(__file__).parent.parent


def test_load_desktop_profile():
    p = load_profile("desktop", ROOT / "profiles")
    assert p.detector.motion_threshold > 0
    assert p.detector.scene_threshold > p.detector.motion_threshold
    assert 0.0 <= p.proactive.talkativeness <= 1.0
    assert p.proactive.cooldown_s > 0


def test_load_anime_profile():
    p = load_profile("anime", ROOT / "profiles")
    assert p.proactive.cooldown_s > 0


def test_load_persona_prompt():
    text = load_persona_prompt("nova", ROOT / "personas")
    assert "NOVA" in text
    assert len(text) > 100
