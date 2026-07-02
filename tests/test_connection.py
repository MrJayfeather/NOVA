import asyncio
from itertools import islice

from nova.client.connection import LatestSlot, backoff_delays


def test_backoff_sequence_caps():
    delays = list(islice(backoff_delays(base=1.0, factor=2.0, max_delay=15.0), 6))
    assert delays == [1.0, 2.0, 4.0, 8.0, 15.0, 15.0]


async def test_latest_slot_keeps_only_freshest():
    slot = LatestSlot()
    slot.put("old")
    slot.put("new")
    assert await slot.get() == "new"


async def test_latest_slot_get_waits_for_put():
    slot = LatestSlot()

    async def putter():
        await asyncio.sleep(0.01)
        slot.put("x")

    task = asyncio.create_task(putter())
    assert await asyncio.wait_for(slot.get(), timeout=1.0) == "x"
    await task
