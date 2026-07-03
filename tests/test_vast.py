import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from vast import load_env, pick_offer, ws_url


def offer(**kw):
    base = dict(id=1, gpu_ram=49152, reliability2=0.99, inet_down=500,
                disk_space=100, dph_total=0.40, gpu_name="A40",
                geolocation="Sweden, SE")
    base.update(kw)
    return base


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
