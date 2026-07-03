from pathlib import Path

from nova.client.config import load_config

ROOT = Path(__file__).parent.parent


def test_load_repo_config():
    cfg = load_config(ROOT / "client_config.yaml")
    assert cfg.server_url.startswith("ws://")
    assert cfg.profile == "desktop"
    assert cfg.persona == "nova"
    assert cfg.periodic_fps > 0
    assert set(cfg.hotkeys) >= {"mute", "comment_now", "pause", "feedback_up", "feedback_down"}


def test_defaults_applied(tmp_path):
    p = tmp_path / "min.yaml"
    p.write_text("server_url: ws://localhost:8000/ws\n", encoding="utf-8")
    cfg = load_config(p)
    assert cfg.profile == "desktop"
    assert cfg.burst_frames == 6
    assert cfg.jpeg_quality == 85
    assert cfg.token == ""
