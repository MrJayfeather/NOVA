"""Запуск NOVA одним кликом: разбудить сервер, дождаться, открыть клиент.

Вызывается из NOVA_START.bat. Понимает все состояния: сервер спит,
сервер уже работает, модели греются.
"""

import subprocess
import sys
import time
from pathlib import Path

import httpx
import yaml

ROOT = Path(__file__).parent.parent


def health_url() -> str:
    cfg = yaml.safe_load((ROOT / "client_config.yaml").read_text(encoding="utf-8"))
    return cfg["server_url"].replace("ws://", "http://").replace("/ws", "/health")


def orchestrator_ready() -> bool:
    try:
        return "clients" in httpx.get(health_url(), timeout=5).text
    except Exception:
        return False


def main() -> None:
    print("=" * 46)
    print("  NOVA — запуск")
    print("=" * 46)

    if orchestrator_ready():
        print("Сервер уже готов — открываю клиента!")
    else:
        print("Бужу сервер (обычно 1–3 минуты, после долгого сна — до 10)...")
        r = subprocess.run(
            [sys.executable, str(ROOT / "scripts" / "vast.py"), "up", "--write-config"],
            cwd=ROOT,
        )
        if r.returncode != 0:
            print("Не удалось поднять сервер — смотри сообщения выше.")
            input("Enter — закрыть...")
            sys.exit(1)

        print("Сервер работает, жду прогрева моделей", end="", flush=True)
        deadline = time.time() + 40 * 60
        while time.time() < deadline:
            if orchestrator_ready():
                print("\nГотово!")
                break
            print(".", end="", flush=True)
            time.sleep(10)
        else:
            print("\nМодели так и не прогрелись за 40 минут — что-то не так.")
            input("Enter — закрыть...")
            sys.exit(1)

    print("Открываю клиента. Говори с NOVA! (закрыть — Ctrl+C или крестик)")
    print("-" * 46)
    subprocess.run([sys.executable, "-m", "nova.client.main"], cwd=ROOT)


if __name__ == "__main__":
    main()
