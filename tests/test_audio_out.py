import base64

from nova.client.audio_out import AudioSink, Player
from nova.shared.protocol import AudioChunk, SpeakEnd, SpeakStart


class FakeSink(AudioSink):
    def __init__(self):
        self.played = []

    def play(self, pcm: bytes, sample_rate: int) -> None:
        self.played.append((pcm, sample_rate))


def utterance(uid="u1", parts=(b"aa", b"bb")):
    msgs = [SpeakStart(utterance_id=uid, text="т", reason="reply", sample_rate=16000)]
    for i, p in enumerate(parts):
        msgs.append(AudioChunk(utterance_id=uid, seq=i, pcm_b64=base64.b64encode(p).decode()))
    msgs.append(SpeakEnd(utterance_id=uid))
    return msgs


def test_chunks_assembled_in_order_and_played():
    sink = FakeSink()
    player = Player(sink)
    for msg in utterance():
        player.handle(msg)
    assert sink.played == [(b"aabb", 16000)]


def test_muted_drops_utterance():
    sink = FakeSink()
    player = Player(sink)
    player.muted = True
    for msg in utterance():
        player.handle(msg)
    assert sink.played == []


def test_chunk_for_unknown_utterance_ignored():
    sink = FakeSink()
    player = Player(sink)
    player.handle(AudioChunk(utterance_id="ghost", seq=0, pcm_b64="YWE="))
    player.handle(SpeakEnd(utterance_id="ghost"))
    assert sink.played == []
