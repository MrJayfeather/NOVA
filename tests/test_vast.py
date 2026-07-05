import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from vast import load_env, pick_offer, ws_url


def offer(**kw):
    base = dict(id=1, gpu_ram=49152, reliability2=0.99, inet_down=500,
                disk_space=100, dph_total=0.40, gpu_name="A40",
                geolocation="Sweden, SE", compute_cap=860)
    base.update(kw)
    return base


def test_pick_offer_rejects_old_gpu_arch():
    offers = [offer(id=1, dph_total=0.20, compute_cap=750),  # Turing: нет FP8
              offer(id=2, dph_total=0.40)]
    assert pick_offer(offers)["id"] == 2


def test_pick_offer_rejects_arm_hosts():
    offers = [offer(id=1, dph_total=0.20, cpu_arch="arm64"),
              offer(id=2, dph_total=0.40, cpu_arch="amd64")]
    assert pick_offer(offers)["id"] == 2


def test_pick_offer_rejects_china():
    offers = [offer(id=1, dph_total=0.20, geolocation="China, CN"),
              offer(id=2, dph_total=0.40)]
    assert pick_offer(offers)["id"] == 2


def test_pick_offer_cheapest_valid():
    offers = [offer(id=1, dph_total=0.50), offer(id=2, dph_total=0.30),
              offer(id=3, dph_total=0.20, gpu_ram=24000)]  # мало VRAM
    assert pick_offer(offers)["id"] == 2


def test_pick_offer_rejects_unreliable_and_slow():
    offers = [offer(reliability2=0.90), offer(inet_down=50), offer(disk_space=40)]
    assert pick_offer(offers) is None


def test_ws_url_from_instance():
    inst = {"public_ipaddr": "1.2.3.4 ", "ports": {"8000/tcp": [{"HostPort": "41234"}]}}
    assert ws_url(inst) == "ws://1.2.3.4:41234/ws"
    assert ws_url({"public_ipaddr": "", "ports": {}}) is None


def test_load_env(tmp_path):
    p = tmp_path / ".env"
    p.write_text("# comment\nVAST_API_KEY=abc\nNOVA_TOKEN = xyz\n", encoding="utf-8")
    env = load_env(p)
    assert env["VAST_API_KEY"] == "abc"
    assert env["NOVA_TOKEN"] == "xyz"


def test_create_env_passes_eyes_and_voice(monkeypatch):
    import vast

    sent = {}

    def fake_put(url, headers=None, json=None, timeout=None):
        sent.update(json["env"])

        class R:
            def raise_for_status(self):
                pass

        return R()

    monkeypatch.setattr(vast.httpx, "put", fake_put)
    vast.create_instance("key", 1, "token", env={
        "GEMINI_KEY": "gk", "NOVA_TTS": "voxcpm", "HF_TOKEN": "hf",
        "NOVA_MEMORY_TOKEN": "mt",
    })
    assert sent["GEMINI_KEY"] == "gk"
    assert sent["NOVA_EYES"] == "gemini"
    assert sent["NOVA_TTS"] == "voxcpm"
    assert sent["NOVA_MEMORY_TOKEN"] == "mt"
    assert "nova-memory" in sent["NOVA_MEMORY_REPO"]
