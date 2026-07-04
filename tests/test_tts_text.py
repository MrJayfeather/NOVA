from nova.server.tts_text import (
    drop_leading_sounds, normalize_for_tts, strip_markers,
)


def test_integers_become_words():
    assert normalize_for_tts("у тебя 25 монет") == "у тебя двадцать пять монет"


def test_decimal_number():
    out = normalize_for_tts("осталось 3.5 часа")
    assert "три" in out and "пять" in out and "3" not in out


def test_time_pattern():
    out = normalize_for_tts("встреча в 2:30")
    assert "два" in out and "тридцать" in out and ":" not in out


def test_known_english_word():
    assert "дискорд" in normalize_for_tts("зайди в Discord")


def test_unknown_english_transliterated():
    out = normalize_for_tts("режим stealth")
    assert "стилт" in out
    assert not any(c.isascii() and c.isalpha() for c in out)


def test_symbols_spoken():
    out = normalize_for_tts("загрузка 100%")
    assert out == "загрузка сто процентов"


def test_symbols_dropped_or_spaced():
    out = normalize_for_tts("путь C:/games/mita")
    assert "/" not in out


def test_plain_russian_untouched():
    s = "Привет, как дела? Всё отлично!"
    assert normalize_for_tts(s) == s


def test_emotion_markers_survive_normalization():
    out = normalize_for_tts("[sarcastic] Ну у тебя и 25 фпс. [laughing]")
    assert "[sarcastic]" in out and "[laughing]" in out
    assert "двадцать пять" in out


def test_strip_markers_for_display():
    s = "[sarcastic] Ну ты гений. [laughing] Прям вау."
    assert strip_markers(s) == "Ну ты гений. Прям вау."


def test_marker_with_space_and_hyphen():
    out = normalize_for_tts("[soft tone] тише. [long-break] дальше")
    assert "[soft tone]" in out and "[long-break]" in out


def test_leading_laugh_dropped():
    assert drop_leading_sounds("[laughing] Ну конечно, с тобой.") == "Ну конечно, с тобой."
    assert drop_leading_sounds("[chuckling] [sighing] Привет.") == "Привет."


def test_leading_emotion_kept():
    s = "[sarcastic] Ну ты гений."
    assert drop_leading_sounds(s) == s


def test_mid_text_laugh_kept():
    s = "Ну ты дал. [laughing] Ладно, живи."
    assert drop_leading_sounds(s) == s


def test_asr_garbage_detected():
    from nova.server.tts_text import asr_garbage
    assert asr_garbage("Субтитры создавал DimaTorzok")
    assert asr_garbage("Продолжение следует...")
    assert asr_garbage("   ")
    assert not asr_garbage("Нова, привет, как дела?")
