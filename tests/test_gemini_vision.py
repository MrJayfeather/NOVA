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
    eyes.on_seen = None
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
    sent_frames = []

    async def fake_call(frames, prompt):
        prompts.append(prompt)
        sent_frames.append(frames)
        return "в правом углу дата 5 июля"

    eyes._call_gemini = fake_call
    await eyes.reply_to_user("какая дата?", [b"old", b"fresh"], [])
    assert "какая дата?" in prompts[0]        # вопрос уехал в промпт глаз
    # только СВЕЖАЙШИЙ кадр: старый рядом путает при противоречивых фактах
    assert sent_frames[0] == [b"fresh"]
    # вставка мозгу помечена «СЕЙЧАС» — против инерции прошлых ответов
    assert "СЕЙЧАС" in inner.calls[0][1]
    # прицельное описание мимо кэша: тот же кадр, другой вопрос — новый вызов
    await eyes.reply_to_user("который час?", [b"old", b"fresh"], [])
    assert "который час?" in prompts[1]


async def test_on_seen_gets_fresh_descriptions():
    eyes = make_eyes(FakeInner(), describe_text="1: замес у точки")
    seen = []
    eyes.on_seen = seen.append
    await eyes.describe([b"j1"])
    await eyes.describe([b"j1"])           # кэш — хук молчит
    assert seen == ["замес у точки"]
    await eyes.reply_to_user("что там?", [b"j2"], [])
    assert len(seen) == 2                  # прицельное описание тоже пишется


def test_wrap_eyes_modes():
    inner = FakeInner()
    assert wrap_eyes(inner, {}) is inner                      # нет ключа
    assert wrap_eyes(inner, {"GEMINI_KEY": "k",
                             "NOVA_EYES": "local"}) is inner  # выключено
    wrapped = wrap_eyes(inner, {"GEMINI_KEY": "k"})
    assert isinstance(wrapped, GeminiEyes)                    # дефолт gemini


async def test_describe_clip_sends_video_and_logs_seen():
    eyes = make_eyes(FakeInner())
    parts_seen = {}

    async def fake_call(frames, prompt, video=None):
        parts_seen["video"] = video
        parts_seen["prompt"] = prompt
        return "0:03 Джефф съел троих"

    eyes._call_gemini = fake_call
    seen = []
    eyes.on_seen = seen.append
    out = await eyes.describe_clip(b"MP4DATA", hint="матч Rivals")
    assert "Джефф" in out
    assert parts_seen["video"] == b"MP4DATA"
    assert "Rivals" in parts_seen["prompt"]
    assert seen == ["0:03 Джефф съел троих"]
