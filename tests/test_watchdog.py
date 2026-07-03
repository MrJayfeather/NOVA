import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "deploy"))

from idle_watchdog import instance_id, should_stop


def test_should_stop_only_when_idle_and_empty():
    assert should_stop(clients=0, idle_s=901)
    assert not should_stop(clients=1, idle_s=901)
    assert not should_stop(clients=0, idle_s=100)


def test_instance_id_parsed_from_label(monkeypatch):
    monkeypatch.setenv("VAST_CONTAINERLABEL", "C.12345")
    assert instance_id() == "12345"
    monkeypatch.delenv("VAST_CONTAINERLABEL")
    assert instance_id() is None
