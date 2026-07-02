import numpy as np

from nova.client.capture import cursor_pos, encode_jpeg, to_gray_small


def test_to_gray_small_shape_and_dtype():
    frame = np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8)
    small = to_gray_small(frame)
    assert small.shape == (90, 160)
    assert small.dtype == np.uint8


def test_encode_jpeg_roundtrip():
    import cv2

    frame = np.full((100, 100, 3), 128, dtype=np.uint8)
    data = encode_jpeg(frame, quality=85)
    assert data[:2] == b"\xff\xd8"  # JPEG magic
    decoded = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    assert decoded.shape == (100, 100, 3)


def test_cursor_pos_returns_ints():
    x, y = cursor_pos()
    assert isinstance(x, int) and isinstance(y, int)
