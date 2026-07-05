import numpy as np

from nova.server.models.vox_tts import (
    cut_spoken_head, norm_word, normalize_peak, stress_to_acute,
)


def test_stress_plus_to_acute():
    # RUAccent пишет «пр+омах», VoxCPM2 понимает «про́мах» (U+0301 после гласной)
    assert stress_to_acute("каждый твой пр+омах") == "каждый твой про́мах"
    assert stress_to_acute("молок+о и хл+еб") == "молоко́ и хле́б"
    assert stress_to_acute("без ударений") == "без ударений"


def test_norm_word_strips_punct_and_yo():
    assert norm_word("Слушай,") == "слушай"
    assert norm_word("ЕЩЁ!") == "еще"


def test_cut_spoken_head_cuts_before_first_word():
    rate = 100  # секунда = 100 сэмплов, удобно считать
    pcm = np.arange(500, dtype=np.int16)  # 5 «секунд»
    words = [("and", 0.5), ("speaking", 1.0), ("Слушай,", 3.0), ("я", 3.5)]
    out = cut_spoken_head(pcm, rate, words, "Слушай", margin=0.12, fade=0.0)
    # срез на 3.0 - 0.12 = 2.88с -> сэмпл 288
    assert len(out) == 500 - 288
    assert out[0] == 288


def test_cut_spoken_head_fade_in():
    rate = 100
    pcm = np.full(500, 1000, dtype=np.int16)
    words = [("Слушай", 1.0)]
    out = cut_spoken_head(pcm, rate, words, "Слушай", margin=0.0, fade=0.1)
    assert out[0] == 0            # начало фейда — тишина
    assert out[20] == 1000        # после 10 сэмплов фейда — полная громкость


def test_cut_spoken_head_word_missing_returns_all():
    pcm = np.arange(100, dtype=np.int16)
    out = cut_spoken_head(pcm, 100, [("другое", 0.1)], "Слушай")
    assert len(out) == 100


def test_normalize_peak_boosts_quiet():
    pcm = np.array([0, 100, -100], dtype=np.int16)
    out = normalize_peak(pcm)
    assert int(np.abs(out).max()) == 23000


def test_normalize_peak_keeps_loud():
    pcm = np.array([0, 30000], dtype=np.int16)
    assert normalize_peak(pcm)[1] == 30000
