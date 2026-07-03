import base64

from nova.client.audio_out import Player, StreamSink
from nova.shared.protocol import AudioChunk, SpeakEnd, SpeakStart


class FakeStreamSink(StreamSink):
    def __init__(self):
        self.events = []

    def start(self, sample_rate: int) -> None:
        self.events.append(("start", sample_rate))

    def write(self, pcm: bytes) -> None:
        self.events.append(("write", pcm))

    def stop(self) -> None:
        self.events.append(("stop",))


def utterance(uid="u1", parts=(b"aa", b"bb")):
    msgs = [SpeakStart(utterance_id=uid, text="т", reason="reply", sample_rate=24000)]
    for i, p in enumerate(parts):
        msgs.append(AudioChunk(utterance_id=uid, seq=i, pcm_b64=base64.b64encode(p).decode()))
    msgs.append(SpeakEnd(utterance_id=uid))
    return msgs


def test_chunks_streamed_in_order():
    sink = FakeStreamSink()
    player = Player(sink)
    for msg in utterance():
        player.handle(msg)
    player.drain()
    assert sink.events == [
        ("start", 24000), ("write", b"aa"), ("write", b"bb"), ("stop",),
    ]


def test_muted_drops_utterance():
    sink = FakeStreamSink()
    player = Player(sink)
    player.muted = True
    for msg in utterance():
        player.handle(msg)
    player.drain()
    assert sink.events == []


def test_chunk_for_unknown_utterance_ignored():
    sink = FakeStreamSink()
    player = Player(sink)
    player.handle(AudioChunk(utterance_id="ghost", seq=0, pcm_b64="YWE="))
    player.handle(SpeakEnd(utterance_id="ghost"))
    player.drain()
    assert sink.events == []
