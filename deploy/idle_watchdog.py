"""Автостоп: если к NOVA 15 минут никто не подключён — остановить инстанс,
чтобы не жечь деньги. Работает на самом инстансе."""
import os
import re
import subprocess
import time

import httpx

API = "https://console.vast.ai/api/v0"  # смена состояния инстанса — на v0
IDLE_LIMIT_S = float(os.environ.get("NOVA_IDLE_LIMIT", "900"))

# тяжёлая GPU-работа без подключённых клиентов — тоже активность:
# однажды вачдог усыпил инстанс посреди дообучения голоса
BUSY_PATTERNS = ("fish_speech/train.py", "merge_lora", "extract_vq",
                 "build_dataset", "finetune_prep", "rvc_bench", "uv venv",
                 "finetune_cli", "pip install")


def should_stop(clients: int, idle_s: float, limit_s: float = IDLE_LIMIT_S) -> bool:
    return clients == 0 and idle_s >= limit_s


def gpu_work_running() -> bool:
    for pat in BUSY_PATTERNS:
        if subprocess.run(["pgrep", "-f", pat], capture_output=True).returncode == 0:
            return True
    return False


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
    started = time.time()
    while True:
        time.sleep(60)
        # льготный период: не стрелять раньше лимита с СОБСТВЕННОГО старта —
        # унаследованный многочасовой idle не повод для мгновенной казни
        if time.time() - started < IDLE_LIMIT_S:
            continue
        try:
            h = httpx.get("http://127.0.0.1:8000/health", timeout=5).json()
        except Exception:
            continue  # оркестратор ещё грузится
        if gpu_work_running():
            continue
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
