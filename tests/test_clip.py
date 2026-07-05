import shutil

import numpy as np
import pytest

from nova.client.clip import encode_clip

needs_ffmpeg = pytest.mark.skipif(shutil.which("ffmpeg") is None,
                                  reason="нет ffmpeg")


@needs_ffmpeg
def test_encode_clip_makes_playable_mp4(tmp_path):
    frames = [np.full((90, 160, 3), i * 16, dtype=np.uint8)
              for i in range(16)]
    out = tmp_path / "clip.mp4"
    assert encode_clip(frames, fps=8, wav_path=None, out_path=str(out))
    data = out.read_bytes()
    assert len(data) > 500
    assert b"ftyp" in data[:64]           # валидная mp4-шапка


@needs_ffmpeg
def test_encode_clip_with_audio(tmp_path):
    import wave

    wav = tmp_path / "a.wav"
    with wave.open(str(wav), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes(b"\x00\x01" * 16000)
    frames = [np.zeros((90, 160, 3), dtype=np.uint8) for _ in range(8)]
    out = tmp_path / "clip.mp4"
    assert encode_clip(frames, fps=8, wav_path=str(wav), out_path=str(out))
    assert out.stat().st_size > 500


def test_encode_clip_no_frames_returns_false(tmp_path):
    assert not encode_clip([], fps=8, wav_path=None,
                           out_path=str(tmp_path / "x.mp4"))


async def test_clip_sender_keeps_only_latest():
    from nova.client.clip import ClipSender

    sent = []

    class FakeConn:
        def send(self, msg):
            sent.append(msg)

    s = ClipSender(FakeConn(), kbps=10 ** 9)   # без троттла в тесте
    s.offer("клип1")
    s.offer("клип2")                            # первый ещё не ушёл — вытеснен
    await s.pump_once()
    assert sent == ["клип2"]
    await s.pump_once()                         # пусто — ничего не шлёт
    assert sent == ["клип2"]
