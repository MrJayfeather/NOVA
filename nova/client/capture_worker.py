"""Захват экрана в отдельном процессе.

dxcam (DXGI duplication) в одном процессе с PortAudio (sounddevice)
и onnxruntime (pysilero-vad) падает с access violation при grab —
конфликт нативных библиотек. Изоляция захвата в дочерний процесс
убирает весь класс таких конфликтов и уносит JPEG-кодирование
с главного цикла клиента.
"""
import time


def run_capture(out_queue, jpeg_quality: int = 85, fps: float = 15.0) -> None:
    from nova.client.capture import Grabber, cursor_pos, encode_jpeg, to_gray_small

    grabber = Grabber()
    period = 1.0 / fps
    while True:
        started = time.time()
        frame = grabber.grab()
        if frame is not None:
            item = (
                time.time(),
                encode_jpeg(frame, jpeg_quality),
                to_gray_small(frame),
                cursor_pos(),
            )
            if out_queue.full():
                try:
                    out_queue.get_nowait()
                except Exception:
                    pass
            out_queue.put(item)
        time.sleep(max(0.0, period - (time.time() - started)))


class ProcessFrameSource:
    """Источник кадров из дочернего процесса захвата.

    get() -> (ts, jpeg_bytes, gray_small, (cursor_x, cursor_y)) | None
    """

    def __init__(self, jpeg_quality: int = 85, fps: float = 15.0):
        import multiprocessing as mp

        self._queue = mp.Queue(maxsize=2)
        self._proc = mp.Process(
            target=run_capture, args=(self._queue, jpeg_quality, fps), daemon=True
        )
        self._proc.start()

    def get(self):
        import queue

        try:
            return self._queue.get(timeout=1.0)
        except queue.Empty:
            return None
