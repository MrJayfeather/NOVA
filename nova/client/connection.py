import asyncio
from typing import Callable, Iterator

import websockets

from nova.shared.protocol import Hello, dump_message, parse_server_message


def backoff_delays(base: float = 1.0, factor: float = 2.0, max_delay: float = 15.0) -> Iterator[float]:
    delay = base
    while True:
        yield delay
        delay = min(delay * factor, max_delay)


class LatestSlot:
    def __init__(self):
        self._q: asyncio.Queue = asyncio.Queue(maxsize=1)

    def put(self, item) -> None:
        if self._q.full():
            self._q.get_nowait()
        self._q.put_nowait(item)

    async def get(self):
        return await self._q.get()


class Connection:
    def __init__(self, url: str, on_message: Callable, hello: Hello,
                 on_disconnect: Callable | None = None):
        self._url = url
        self._on_message = on_message
        self._hello = hello
        self._on_disconnect = on_disconnect
        self._out: asyncio.Queue = asyncio.Queue()
        self._frames = LatestSlot()

    def send(self, msg) -> None:
        self._out.put_nowait(dump_message(msg))

    def send_frame(self, msg) -> None:
        self._frames.put(dump_message(msg))

    async def run(self) -> None:
        delays = backoff_delays()
        while True:
            try:
                # ping_timeout щедрый: сеть хостов vast временами замирает
                # на десятки секунд, не стоит рвать сессию из-за этого
                async with websockets.connect(
                    self._url, max_size=32 * 1024 * 1024,
                    ping_interval=20, ping_timeout=60,
                ) as ws:
                    delays = backoff_delays()  # успешный коннект — сброс backoff
                    await ws.send(dump_message(self._hello))
                    print(f"[nova] подключено к {self._url}")
                    await asyncio.gather(
                        self._pump_queue(ws), self._pump_frames(ws), self._pump_in(ws)
                    )
            except (OSError, websockets.WebSocketException) as exc:
                # обрыв посреди её реплики теряет SpeakEnd: даём клиенту
                # сбросить зависшие флаги (иначе микрофон глохнет навсегда)
                if self._on_disconnect:
                    self._on_disconnect()
                delay = next(delays)
                print(f"[nova] соединение потеряно ({exc!r}), повтор через {delay:.0f}с")
                await asyncio.sleep(delay)

    async def _pump_queue(self, ws) -> None:
        while True:
            await ws.send(await self._out.get())

    async def _pump_frames(self, ws) -> None:
        while True:
            await ws.send(await self._frames.get())

    async def _pump_in(self, ws) -> None:
        async for raw in ws:
            self._on_message(parse_server_message(raw))
