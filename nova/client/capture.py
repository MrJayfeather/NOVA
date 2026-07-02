import numpy as np


def to_gray_small(frame_bgr: np.ndarray, width: int = 160, height: int = 90) -> np.ndarray:
    import cv2

    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    return cv2.resize(gray, (width, height), interpolation=cv2.INTER_AREA)


def encode_jpeg(frame_bgr: np.ndarray, quality: int = 85) -> bytes:
    import cv2

    ok, buf = cv2.imencode(".jpg", frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise RuntimeError("jpeg encode failed")
    return buf.tobytes()


def cursor_pos() -> tuple[int, int]:
    try:
        import ctypes
        from ctypes import wintypes

        pt = wintypes.POINT()
        ctypes.windll.user32.GetCursorPos(ctypes.byref(pt))
        return int(pt.x), int(pt.y)
    except Exception:
        return (0, 0)


class Grabber:
    """dxcam (DXGI duplication) с fallback на mss."""

    def __init__(self):
        self._backend = "none"
        try:
            import dxcam

            self._cam = dxcam.create(output_color="BGR")
            if self._cam is not None:
                self._backend = "dxcam"
        except Exception:
            pass
        if self._backend == "none":
            import mss

            self._sct = mss.mss()
            self._backend = "mss"
        print(f"[nova] захват экрана: {self._backend}")

    def grab(self) -> np.ndarray | None:
        if self._backend == "dxcam":
            return self._cam.grab()  # None, если кадр не менялся
        raw = self._sct.grab(self._sct.monitors[1])
        return np.asarray(raw)[:, :, :3].copy()
