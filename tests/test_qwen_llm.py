from nova.server.models.base import NO_COMMENT
from nova.server.models.qwen_llm import QwenVLM


def make_llm():
    return QwenVLM(persona_prompt="Ты — NOVA.", base_url="http://x/v1", model="test-model")


def test_reply_messages_structure():
    history = [
        {"role": "user", "content": "раньше"},
        {"role": "assistant", "content": "ответ"},
    ]
    msgs = make_llm().build_reply_messages("привет", history)
    assert msgs[0] == {"role": "system", "content": "Ты — NOVA."}
    assert msgs[1:3] == history
    assert msgs[-1] == {"role": "user", "content": "привет"}


def test_comment_messages_have_images_and_pass_instruction():
    frames = [b"jpg1", b"jpg2"]
    msgs = make_llm().build_comment_messages("scene_change", frames, history=[])
    content = msgs[-1]["content"]
    images = [c for c in content if c["type"] == "image_url"]
    assert len(images) == 2
    assert images[0]["image_url"]["url"].startswith("data:image/jpeg;base64,")
    text = [c for c in content if c["type"] == "text"][0]["text"]
    assert "scene_change" in text
    assert NO_COMMENT in text


def test_comment_frames_capped_at_eight():
    frames = [b"x"] * 20
    msgs = make_llm().build_comment_messages("motion_burst", frames, history=[])
    images = [c for c in msgs[-1]["content"] if c["type"] == "image_url"]
    assert len(images) == 8
