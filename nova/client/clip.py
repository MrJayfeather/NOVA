import asyncio
import subprocess
import tempfile
from pathlib import Path

import numpy as np


def encode_clip(frames: list, fps: int, wav_path: str | None,
                out_path: str, max_w: int = 1920, crf: int = 26) -> bool:
    """Кадры -> h264 mp4 (ширина до max_w); wav_path муксится в звук.
    Кадры — либо numpy BGR, либо готовые JPEG-байты (конвейер клиента
    отдаёт jpeg). ffmpeg обязателен; любая ошибка -> False и печать."""
    if not frames:
        return False
    jpeg_input = isinstance(frames[0], (bytes, bytearray))
    cmd = ["ffmpeg", "-y", "-v", "error"]
    if jpeg_input:
        cmd += ["-f", "image2pipe", "-c:v", "mjpeg", "-r", str(fps), "-i", "-"]
        vf = f"scale='min({max_w},iw)':-2"
    else:
        h, w = frames[0].shape[:2]
        # даунскейл к max_w по ширине, чётные размеры для h264
        scale = min(1.0, max_w / w)
        out_w, out_h = int(w * scale) // 2 * 2, int(h * scale) // 2 * 2
        cmd += ["-f", "rawvideo", "-pix_fmt", "bgr24",
                "-s", f"{w}x{h}", "-r", str(fps), "-i", "-"]
        vf = f"scale={out_w}:{out_h}"
    if wav_path:
        cmd += ["-i", wav_path, "-c:a", "aac", "-shortest"]
    cmd += ["-vf", vf,
            "-c:v", "libx264", "-preset", "veryfast", "-crf", str(crf),
            "-pix_fmt", "yuv420p", "-movflags", "+faststart", out_path]
    p = None
    try:
        p = subprocess.Popen(cmd, stdin=subprocess.PIPE,
                             stderr=subprocess.PIPE)
        for f in frames:
            p.stdin.write(bytes(f) if jpeg_input
                          else np.ascontiguousarray(f).tobytes())
        p.stdin.close()
        p.wait(timeout=60)
        if p.returncode != 0:
            err = (p.stderr.read() or b"")[-300:].decode("utf-8", "replace")
            print(f"[nova] клип: ffmpeg код {p.returncode}: {err}")
        return p.returncode == 0 and Path(out_path).exists()
    except Exception as exc:
        # BrokenPipe = ffmpeg умер раньше кадров; его stderr — причина
        err = ""
        if p is not None and p.stderr is not None:
            try:
                err = (p.stderr.read() or b"")[-300:].decode("utf-8", "replace")
            except Exception:
                pass
        print(f"[nova] клип: ffmpeg не собрал ({exc!r}) {err}")
        return False


class LoopbackRecorder:
    """Системный звук (WASAPI loopback) для кино-режима. Нет
    pyaudiowpatch/устройства — работаем без звука, не падаем."""

    def __init__(self):
        self._stream = None
        self._pa = None
        self._frames: list[bytes] = []
        self._rate = 48000
        self._channels = 2

    def start(self) -> None:
        if self._stream is not None:
            return
        try:
            import pyaudiowpatch as pa

            self._pa = pa.PyAudio()
            wasapi = self._pa.get_host_api_info_by_type(pa.paWASAPI)
            out = self._pa.get_device_info_by_index(
                wasapi["defaultOutputDevice"])
            # loopback-двойник дефолтного вывода
            for i in range(self._pa.get_device_count()):
                dev = self._pa.get_device_info_by_index(i)
                if dev.get("isLoopbackDevice") and out["name"] in dev["name"]:
                    # loopback открывается ТОЛЬКО с родными параметрами
                    # устройства: моно даёт OSError -9996 Invalid device
                    self._rate = int(dev["defaultSampleRate"])
                    self._channels = max(1, int(dev["maxInputChannels"]))
                    self._frames = []
                    self._stream = self._pa.open(
                        format=pa.paInt16, channels=self._channels,
                        rate=self._rate,
                        input=True, input_device_index=i,
                        frames_per_buffer=4096,
                        stream_callback=self._cb)
                    return
            print("[nova] клип: loopback-устройство не найдено — без звука")
        except Exception as exc:
            print(f"[nova] клип: звук недоступен ({exc!r}) — без звука")
            self._stream = None

    def _cb(self, in_data, *_):
        import pyaudiowpatch as pa

        self._frames.append(in_data)
        return (None, pa.paContinue)

    def stop(self) -> str | None:
        """Останавливает запись; возвращает путь wav или None."""
        if self._stream is None:
            return None
        import wave

        try:
            self._stream.stop_stream()
            self._stream.close()
        except Exception:
            pass
        self._stream = None
        if not self._frames:
            return None
        path = tempfile.mktemp(suffix=".wav")
        with wave.open(path, "wb") as w:
            w.setnchannels(self._channels)
            w.setsampwidth(2)
            w.setframerate(self._rate)
            w.writeframes(b"".join(self._frames))
        self._frames = []
        return path

    def drain(self) -> str | None:
        """Снять накопленный звук и продолжить запись (для потока клипов)."""
        if self._stream is None:
            return None
        path = self.stop()
        self.start()
        return path


class ClipSender:
    """Очередь «только свежий» + троттлинг: клип неделим, но следующий
    не начнёт уходить раньше, чем длина_текущего/kbps секунд — пинг
    онлайн-каток не дёргается бурстами."""

    def __init__(self, conn, kbps: int = 1500):
        self._conn = conn
        self._kbps = max(1, kbps)
        self._latest = None

    def offer(self, msg) -> None:
        self._latest = msg   # старый несданный — вытесняется

    async def pump_once(self) -> None:
        msg = self._latest
        if msg is None:
            return
        self._latest = None
        self._conn.send(msg)
        size_kbit = len(getattr(msg, "mp4_b64", "") or str(msg)) * 8 / 1000
        await asyncio.sleep(size_kbit / self._kbps)

    async def pump_loop(self) -> None:
        while True:
            await self.pump_once()
            await asyncio.sleep(0.5)
