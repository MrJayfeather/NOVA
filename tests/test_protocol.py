import pytest
from pydantic import ValidationError

from nova.shared.protocol import (
    Hello, Frame, DetectorEvent, AudioSegment, Hotkey,
    HelloAck, SpeakStart, AudioChunk, SpeakEnd,
    parse_client_message, parse_server_message, dump_message,
)


def test_client_messages_roundtrip():
    msgs = [
        Hello(profile="desktop", persona="nova"),
        Frame(ts=1.0, jpeg_b64="aGk=", kind="burst", burst_id="b1", seq=2, cursor_x=10, cursor_y=20),
        DetectorEvent(ts=2.0, event="scene_change"),
        AudioSegment(ts=3.0, pcm_b64="aGk=", sample_rate=16000),
        Hotkey(action="comment_now"),
    ]
    for msg in msgs:
        parsed = parse_client_message(dump_message(msg))
        assert parsed == msg


def test_server_messages_roundtrip():
    msgs = [
        HelloAck(mock=True),
        SpeakStart(utterance_id="u1", text="привет", reason="proactive", sample_rate=16000),
        AudioChunk(utterance_id="u1", seq=0, pcm_b64="aGk="),
        SpeakEnd(utterance_id="u1"),
    ]
    for msg in msgs:
        assert parse_server_message(dump_message(msg)) == msg


def test_unknown_type_rejected():
    with pytest.raises(ValidationError):
        parse_client_message('{"type": "hack"}')


def test_clip_roundtrip():
    from nova.shared.protocol import Clip, dump_message, parse_client_message

    msg = Clip(ts=1.0, mp4_b64="QUJD", dur_s=15.0, audio=True)
    back = parse_client_message(dump_message(msg))
    assert isinstance(back, Clip)
    assert back.dur_s == 15.0 and back.audio is True


def test_cinema_mode_roundtrip():
    from nova.shared.protocol import (
        CinemaMode, dump_message, parse_server_message,
    )

    back = parse_server_message(dump_message(CinemaMode(on=True)))
    assert isinstance(back, CinemaMode) and back.on is True


def test_hotkey_cinema_allowed():
    from nova.shared.protocol import Hotkey

    assert Hotkey(action="cinema").action == "cinema"
