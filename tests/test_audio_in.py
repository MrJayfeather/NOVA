from nova.client.audio_in import VAD, VADSegmenter

CHUNK = b"\x01\x00" * 512  # 32 мс PCM16
SILENT = b"\x00\x00" * 512


class ScriptedVAD(VAD):
    """is_speech по заранее заданному сценарию."""

    def __init__(self, flags):
        self._flags = list(flags)

    def is_speech(self, chunk: bytes) -> bool:
        return self._flags.pop(0)


def run(flags, chunks=None):
    seg = VADSegmenter(ScriptedVAD(flags), silence_end_ms=96)  # 3 чанка тишины = конец
    out = []
    for i in range(len(flags)):
        chunk = (chunks or [CHUNK] * len(flags))[i]
        r = seg.feed(chunk)
        if r is not None:
            out.append(r)
    return out


def test_silence_only_no_segments():
    assert run([False] * 10) == []


def test_speech_then_silence_emits_one_segment():
    segments = run([False, True, True, True, False, False, False])
    assert len(segments) == 1
    # пре-ролл + 3 речи + 3 тишины — сегмент содержит всё от пре-ролла
    assert len(segments[0]) >= 4 * len(CHUNK)


def test_pre_roll_included():
    quiet = b"\x02\x00" * 512
    segments = run(
        [False, True, True, False, False, False],
        chunks=[quiet, CHUNK, CHUNK, SILENT, SILENT, SILENT],
    )
    assert len(segments) == 1
    assert segments[0].startswith(quiet)  # пре-ролл попал в сегмент


def test_max_segment_forces_cut():
    seg = VADSegmenter(ScriptedVAD([True] * 100), silence_end_ms=96, max_segment_s=0.1)
    results = [seg.feed(CHUNK) for _ in range(100)]
    emitted = [r for r in results if r is not None]
    assert len(emitted) >= 1
