import wave
from io import BytesIO

from nova.server.models.fish_tts import build_tts_request, wav_to_pcm


def make_wav(rate=44100, channels=1, frames=b"\x01\x00" * 200) -> bytes:
    buf = BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(channels)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(frames)
    return buf.getvalue()


def test_build_request_shape():
    req = build_tts_request("привет", b"refbytes", "текст референса")
    assert req["text"] == "привет"
    assert req["format"] == "wav"
    assert req["streaming"] is False
    assert req["use_memory_cache"] == "on"
    assert req["references"] == [{"audio": b"refbytes", "text": "текст референса"}]


def test_build_request_cloud_reference_id():
    req = build_tts_request("привет", reference_id="abc123")
    assert req["reference_id"] == "abc123"
    assert req["temperature"] == 0.7
    assert "references" not in req
    assert "use_memory_cache" not in req


async def test_failed_sentence_skipped_not_fatal():
    from nova.server.models.fish_tts import FishTTS

    tts = FishTTS.__new__(FishTTS)  # без сети и файлов
    tts._reference_id = "x"
    tts._validator = None
    tts._temperature = 0.5
    tts.sample_rate = 44100
    calls = []

    async def fake_call(sentence, attempts=2):
        calls.append(sentence)
        if len(calls) == 1:
            raise RuntimeError("облако икнуло")
        return make_wav(frames=b"\x10\x00" * 100)

    tts._tts_call = fake_call
    chunks = [c async for c in tts.synthesize("Первое предложение. Второе предложение.")]
    # первое упало и пропущено, второе прозвучало
    assert len(calls) == 2
    assert len(chunks) == 1


def test_wav_to_pcm_mono_normalizes_gain():
    import numpy as np
    frames = b"\x01\x00\x02\x00" * 100  # тихий сигнал, пик = 2
    pcm, rate = wav_to_pcm(make_wav(rate=44100, frames=frames))
    assert rate == 44100
    arr = np.frombuffer(pcm, dtype=np.int16)
    assert arr.max() == 23000  # пик подтянут к целевому уровню


def test_wav_to_pcm_loud_signal_untouched():
    import numpy as np
    loud = (np.ones(200, dtype=np.int16) * 30000).tobytes()
    pcm, _ = wav_to_pcm(make_wav(rate=44100, frames=loud))
    assert np.frombuffer(pcm, dtype=np.int16).max() == 30000


def test_wav_to_pcm_downmixes_stereo():
    stereo = (b"\x00\x00\x64\x00") * 50  # L=0, R=100 -> моно=50 (до нормализации)
    pcm, rate = wav_to_pcm(make_wav(rate=24000, channels=2, frames=stereo))
    assert rate == 24000
    import numpy as np
    arr = np.frombuffer(pcm, dtype=np.int16)
    # ровный сигнал остаётся ровным, пик нормализован
    assert arr[0] == arr[-1] == 23000
