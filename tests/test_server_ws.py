import json
from pathlib import Path

from fastapi.testclient import TestClient

from nova.server.main import create_app
from nova.shared.protocol import DetectorEvent, Hello, dump_message

ROOT = Path(__file__).parent.parent


def make_client(tmp_path):
    app = create_app(
        mock=True,
        profiles_root=ROOT / "profiles",
        personas_root=ROOT / "personas",
        feedback_path=tmp_path / "feedback.jsonl",
    )
    return TestClient(app)


def test_hello_then_event_flow(tmp_path):
    client = make_client(tmp_path)
    with client.websocket_connect("/ws") as ws:
        ws.send_text(dump_message(Hello(profile="desktop", persona="nova")))
        ack = json.loads(ws.receive_text())
        assert ack["type"] == "hello_ack" and ack["mock"] is True

        ws.send_text(dump_message(DetectorEvent(ts=1.0, event="scene_change")))
        start = json.loads(ws.receive_text())
        assert start["type"] == "speak_start" and start["reason"] == "proactive"
        msg = json.loads(ws.receive_text())
        while msg["type"] == "audio_chunk":
            msg = json.loads(ws.receive_text())
        assert msg["type"] == "speak_end"


def test_health_reports_clients_and_idle(tmp_path):
    client = make_client(tmp_path)
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["clients"] == 0
    assert body["idle_s"] >= 0

    with client.websocket_connect("/ws") as ws:
        ws.send_text(dump_message(Hello(profile="desktop", persona="nova")))
        ws.receive_text()
        assert client.get("/health").json()["clients"] == 1


def test_wrong_token_closes_4002(tmp_path):
    app = create_app(
        mock=True,
        profiles_root=ROOT / "profiles",
        personas_root=ROOT / "personas",
        feedback_path=tmp_path / "feedback.jsonl",
        token="secret123",
    )
    client = TestClient(app)
    with client.websocket_connect("/ws") as ws:
        ws.send_text(dump_message(Hello(profile="desktop", persona="nova", token="wrong")))
        data = ws.receive()
        assert data["type"] == "websocket.close"
        assert data["code"] == 4002


def test_correct_token_accepted(tmp_path):
    app = create_app(
        mock=True,
        profiles_root=ROOT / "profiles",
        personas_root=ROOT / "personas",
        feedback_path=tmp_path / "feedback.jsonl",
        token="secret123",
    )
    client = TestClient(app)
    with client.websocket_connect("/ws") as ws:
        ws.send_text(dump_message(Hello(profile="desktop", persona="nova", token="secret123")))
        ack = json.loads(ws.receive_text())
        assert ack["type"] == "hello_ack"


def test_non_hello_first_message_closes(tmp_path):
    client = make_client(tmp_path)
    with client.websocket_connect("/ws") as ws:
        ws.send_text(dump_message(DetectorEvent(ts=1.0, event="scene_change")))
        data = ws.receive()
        assert data["type"] == "websocket.close"
        assert data["code"] == 4000
