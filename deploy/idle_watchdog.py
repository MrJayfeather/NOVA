"""Автостоп: если к NOVA 15 минут никто не подключён — остановить инстанс,
чтобы не жечь деньги. Работает на самом инстансе."""
import os
import re
import time

import httpx

API = "https://console.vast.ai/api/v1"  # управление инстансами — на v1
IDLE_LIMIT_S = 900.0


def should_stop(clients: int, idle_s: float, limit_s: float = IDLE_LIMIT_S) -> bool:
    return clients == 0 and idle_s >= limit_s


def instance_id() -> str | None:
    m = re.search(r"(\d+)", os.environ.get("VAST_CONTAINERLABEL", ""))
    return m.group(1) if m else None


def main() -> None:
    key = os.environ.get("VAST_API_KEY", "")
    iid = instance_id()
    if not key or not iid:
        print("[watchdog] нет VAST_API_KEY или id инстанса — автостоп выключен")
        return
    print(f"[watchdog] слежу за простоем инстанса {iid} (лимит {IDLE_LIMIT_S:.0f}с)")
    while True:
        time.sleep(60)
        try:
            h = httpx.get("http://127.0.0.1:8000/health", timeout=5).json()
        except Exception:
            continue  # оркестратор ещё грузится
        if should_stop(h.get("clients", 0), h.get("idle_s", 0.0)):
            print("[watchdog] 15 минут простоя — останавливаю инстанс")
            httpx.put(
                f"{API}/instances/{iid}/",
                headers={"Authorization": f"Bearer {key}"},
                json={"state": "stopped"},
                timeout=30,
            )
            return


if __name__ == "__main__":
    main()
