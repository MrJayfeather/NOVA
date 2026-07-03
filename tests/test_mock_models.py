import struct

from nova.server.models.mock import MockASR, MockLLM, MockTTS


async def test_mock_asr_reports_duration():
    pcm = b"\x00\x00" * 16000  # 1 секунда тишины PCM16 @16kHz
    text = await MockASR().transcribe(pcm, sample_rate=16000)
    assert "1.0" in text


async def test_mock_llm_replies_and_comments():
    llm = MockLLM(persona_prompt="Ты — NOVA.")
    reply = await llm.reply_to_user("привет", frames=[], history=[])
    assert "привет" in reply
    comment = await llm.comment_on_event("scene_change", frames=[b"jpg"], history=[])
    assert "scene_change" in comment
    assert "1" in comment  # количество кадров


async def test_mock_tts_yields_pcm16_chunks():
    tts = MockTTS()
    chunks = [c async for c in tts.synthesize("привет мир")]
    assert len(chunks) >= 1
    total = b"".join(chunks)
    assert len(total) % 2 == 0 and len(total) > 0
    # валидный int16
    struct.unpack(f"<{len(total)//2}h", total)
