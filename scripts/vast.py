"""Управление GPU-инстансом NOVA на Vast.ai.

  python scripts/vast.py search           — топ дешёвых подходящих карт
  python scripts/vast.py up [--write-config]  — старт (существующий или новый)
  python scripts/vast.py status           — состояние и баланс
  python scripts/vast.py down [--destroy] — стоп (диск сохраняется) / полное удаление
"""
import argparse
import secrets
import sys
import time
from pathlib import Path

import httpx

API = "https://console.vast.ai/api/v0"
API_V1 = "https://console.vast.ai/api/v1"  # инстансы переехали на v1
LABEL = "nova"
IMAGE = "vllm/vllm-openai:v0.11.0"
DISK_GB = 80
ROOT = Path(__file__).parent.parent


def load_env(path: Path) -> dict:
    env = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if "=" in line and not line.lstrip().startswith("#"):
                k, v = line.split("=", 1)
                env[k.strip()] = v.strip()
    return env


def pick_offer(offers: list[dict], min_disk: float = 80.0) -> dict | None:
    ok = [
        o for o in offers
        if o.get("gpu_ram", 0) >= 47000
        and o.get("reliability2", 0) >= 0.98
        and o.get("inet_down", 0) >= 200
        and o.get("disk_space", 0) >= min_disk
        # хосты из Китая не достучатся до huggingface/github
        and not any(cc in (o.get("geolocation") or "") for cc in ("CN", "China"))
    ]
    return min(ok, key=lambda o: o.get("dph_total", 9e9)) if ok else None


def ws_url(instance: dict) -> str | None:
    ports = (instance.get("ports") or {}).get("8000/tcp") or []
    ip = (instance.get("public_ipaddr") or "").strip()
    if ip and ports:
        return f"ws://{ip}:{ports[0]['HostPort']}/ws"
    return None


# ---- REST-обвязка ----

def _headers(key: str) -> dict:
    return {"Authorization": f"Bearer {key}"}


def search_offers(key: str) -> list[dict]:
    q = {
        "verified": {"eq": True}, "rentable": {"eq": True},
        "num_gpus": {"eq": 1}, "gpu_ram": {"gte": 47000},
        "order": [["dph_total", "asc"]], "type": "on-demand", "limit": 40,
    }
    r = httpx.post(f"{API}/bundles/", headers=_headers(key), json=q, timeout=60)
    r.raise_for_status()
    return r.json().get("offers", [])


def my_instances(key: str) -> list[dict]:
    r = httpx.get(f"{API_V1}/instances/", headers=_headers(key), timeout=60)
    r.raise_for_status()
    return [i for i in r.json().get("instances", []) if i.get("label") == LABEL]


def create_instance(key: str, offer_id: int, token: str) -> None:
    # CRLF из windows-копии ломает bash на инстансе
    onstart = (ROOT / "deploy" / "onstart.sh").read_text(encoding="utf-8").replace("\r\n", "\n")
    body = {
        "client_id": "me",
        "image": IMAGE,
        "disk": DISK_GB,
        "label": LABEL,
        "onstart": onstart,
        "runtype": "ssh",
        "env": {
            "-p 8000:8000": "1",
            "NOVA_MOCK": "0",
            "NOVA_TOKEN": token,
            "VAST_API_KEY": key,
            "HF_HOME": "/workspace/hf",
            "COQUI_TOS_AGREED": "1",
        },
    }
    r = httpx.put(f"{API}/asks/{offer_id}/", headers=_headers(key), json=body, timeout=60)
    r.raise_for_status()


def set_state(key: str, instance_id: int, state: str) -> None:
    r = httpx.put(f"{API_V1}/instances/{instance_id}/", headers=_headers(key),
                  json={"state": state}, timeout=60)
    r.raise_for_status()


def destroy(key: str, instance_id: int) -> None:
    r = httpx.delete(f"{API_V1}/instances/{instance_id}/", headers=_headers(key), timeout=60)
    r.raise_for_status()


def credit(key: str) -> float:
    r = httpx.get(f"{API}/users/current/", headers=_headers(key), timeout=60)
    r.raise_for_status()
    return round(r.json().get("credit", 0.0), 2)


# ---- команды ----

def ensure_token(env_path: Path, env: dict) -> str:
    token = env.get("NOVA_TOKEN", "")
    if not token:
        token = secrets.token_hex(16)
        with env_path.open("a", encoding="utf-8") as f:
            f.write(f"NOVA_TOKEN={token}\n")
        print("[vast] сгенерирован NOVA_TOKEN и добавлен в .env")
    return token


def write_client_config(url: str, token: str) -> None:
    import yaml

    path = ROOT / "client_config.yaml"
    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    cfg["server_url"] = url
    cfg["token"] = token
    path.write_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False),
                    encoding="utf-8")
    print(f"[vast] client_config.yaml обновлён: {url}")


def cmd_search(key: str) -> None:
    for o in sorted(search_offers(key), key=lambda o: o["dph_total"])[:10]:
        print(f"  {o['gpu_name']:<14} {o['gpu_ram'] / 1024:.0f}ГБ  "
              f"${o['dph_total']:.3f}/ч  надёжн.{o.get('reliability2', 0):.2f}  "
              f"↓{o.get('inet_down', 0):.0f}Мбит  {o.get('geolocation') or '?':<16} id={o['id']}")


def cmd_up(key: str, env_path: Path, env: dict, write_config: bool) -> None:
    token = ensure_token(env_path, env)
    existing = my_instances(key)
    if existing:
        inst = existing[0]
        if inst.get("actual_status") != "running":
            print(f"[vast] запускаю существующий инстанс {inst['id']}...")
            set_state(key, inst["id"], "running")
    else:
        offer = pick_offer(search_offers(key), min_disk=DISK_GB)
        if offer is None:
            print("[vast] нет подходящих карт — попробуй позже")
            sys.exit(1)
        print(f"[vast] арендую {offer['gpu_name']} за ${offer['dph_total']:.3f}/ч...")
        create_instance(key, offer["id"], token)

    print("[vast] жду запуска (первый старт с загрузкой моделей — 15–25 минут)...")
    while True:
        time.sleep(15)
        insts = my_instances(key)
        if not insts:
            continue
        inst = insts[0]
        url = ws_url(inst)
        status = inst.get("actual_status")
        print(f"  статус: {status}")
        if status == "running" and url:
            print(f"[vast] инстанс работает: {url}")
            print(f"[vast] цена: ${inst.get('dph_total', 0):.3f}/ч | баланс: ${credit(key)}")
            if write_config:
                write_client_config(url, token)
            print("[vast] дождись, пока прогреются модели (см. status), затем запускай клиент")
            return


def cmd_status(key: str) -> None:
    insts = my_instances(key)
    if not insts:
        print("[vast] инстансов нет")
    for i in insts:
        print(f"  id={i['id']}  {i.get('gpu_name')}  {i.get('actual_status')}  "
              f"${i.get('dph_total', 0):.3f}/ч  {ws_url(i) or 'ещё нет адреса'}")
        url = ws_url(i)
        if url:
            health = url.replace("ws://", "http://").replace("/ws", "/health")
            try:
                h = httpx.get(health, timeout=5).json()
                print(f"    NOVA готова: клиентов {h['clients']}, простой {h['idle_s']}с")
            except Exception:
                print("    NOVA ещё грузит модели (или порт не открылся)")
    print(f"  баланс: ${credit(key)}")


def cmd_down(key: str, destroy_it: bool) -> None:
    for i in my_instances(key):
        if destroy_it:
            destroy(key, i["id"])
            print(f"[vast] инстанс {i['id']} УДАЛЁН (диск и кэш моделей стёрты)")
        else:
            set_state(key, i["id"], "stopped")
            print(f"[vast] инстанс {i['id']} остановлен "
                  f"(диск сохранён, ~$0.1-0.2/день за хранение)")
    print(f"  баланс: ${credit(key)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["search", "up", "status", "down"])
    parser.add_argument("--destroy", action="store_true")
    parser.add_argument("--write-config", action="store_true")
    args = parser.parse_args()

    env_path = ROOT / ".env"
    env = load_env(env_path)
    key = env.get("VAST_API_KEY", "")
    if not key:
        print("Нет VAST_API_KEY в .env")
        sys.exit(1)

    if args.command == "search":
        cmd_search(key)
    elif args.command == "up":
        cmd_up(key, env_path, env, args.write_config)
    elif args.command == "status":
        cmd_status(key)
    elif args.command == "down":
        cmd_down(key, args.destroy)


if __name__ == "__main__":
    main()
