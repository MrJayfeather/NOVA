from nova.server.models.gemini_vision import GeminiEyes, wrap_eyes


class FakeInner:
    def __init__(self):
        self.calls = []

    async def reply_to_user(self, text, frames, history):
        self.calls.append(("reply", text, frames))
        return "ответ"

    async def comment_on_event(self, event, frames, history):
        self.calls.append(("comment", event, frames))
        return "коммент"


def make_eyes(inner, describe_text="1: рабочий стол"):
    eyes = GeminiEyes.__new__(GeminiEyes)
    eyes._inner = inner
    eyes._model = "m"
    eyes._max_frames = 4
    eyes._cache = {}
    eyes._last_summary = ""
    eyes.gemini_calls = 0

    async def fake_call(frames, prompt):
        eyes.gemini_calls += 1
        return describe_text

    eyes._call_gemini = fake_call
    return eyes


async def test_reply_injects_description_and_drops_frames():
    inner = FakeInner()
    eyes = make_eyes(inner)
    out = await eyes.reply_to_user("что видишь?", [b"jpeg1"], [])
    assert out == "ответ"
    kind, text, frames = inner.calls[0]
    assert frames == []                      # мозг больше не получает картинок
    assert "рабочий стол" in text
    assert "что видишь?" in text


async def test_describe_cached_by_frame_bytes():
    eyes = make_eyes(FakeInner())
    await eyes.describe([b"jpeg1"])
    await eyes.describe([b"jpeg1"])          # тот же кадр — без запроса
    assert eyes.gemini_calls == 1


async def test_describe_failure_honest_stub():
    eyes = make_eyes(FakeInner())

    async def boom(frames, prompt):
        raise RuntimeError("облако закрылось")

    eyes._call_gemini = boom
    desc = await eyes.describe([b"jpeg1"])
    assert desc == "экран виден плохо"       # честно, без выдумок


async def test_comment_includes_description():
    inner = FakeInner()
    eyes = make_eyes(inner)
    await eyes.comment_on_event("окно сменилось", [b"jpeg1"], [])
    kind, event, frames = inner.calls[0]
    assert frames == []
    assert "рабочий стол" in event


async def test_no_frames_reply_passthrough():
    inner = FakeInner()
    eyes = make_eyes(inner)
    await eyes.reply_to_user("привет", [], [])
    assert inner.calls[0][1] == "привет"     # без вставки [экран: ]


async def test_reply_describe_targets_question():
    inner = FakeInner()
    eyes = make_eyes(inner)
    prompts = []

    async def fake_call(frames, prompt):
        prompts.append(prompt)
        return "в правом углу дата 5 июля"

    eyes._call_gemini = fake_call
    await eyes.reply_to_user("какая дата?", [b"jpeg1"], [])
    assert "какая дата?" in prompts[0]        # вопрос уехал в промпт глаз
    # прицельное описание мимо кэша: тот же кадр, другой вопрос — новый вызов
    await eyes.reply_to_user("который час?", [b"jpeg1"], [])
    assert "который час?" in prompts[1]


def test_wrap_eyes_modes():
    inner = FakeInner()
    assert wrap_eyes(inner, {}) is inner                      # нет ключа
    assert wrap_eyes(inner, {"GEMINI_KEY": "k",
                             "NOVA_EYES": "local"}) is inner  # выключено
    wrapped = wrap_eyes(inner, {"GEMINI_KEY": "k"})
    assert isinstance(wrapped, GeminiEyes)                    # дефолт gemini
