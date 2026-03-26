#!/usr/bin/env python3
"""
GPU Network Client — Keep your device connected to earn AI access.

Install:  pip install requests
Run:      python gpu_client.py
Stop:     Ctrl+C

Your API key will be displayed — use it with any OpenAI-compatible client.
"""

import os
import sys
import json
import time
import uuid
import platform
import threading
import requests

API_BASE = "https://gpu.social"
CONFIG_FILE = os.path.expanduser("~/.gpu_network.json")
HEARTBEAT_INTERVAL = 180  # 3 minutes


def get_device_id():
    """Get or create persistent device ID."""
    config = load_config()
    if "device_id" in config:
        return config["device_id"]
    device_id = str(uuid.uuid4())
    config["device_id"] = device_id
    save_config(config)
    return device_id


def load_config():
    try:
        with open(CONFIG_FILE) as f:
            return json.load(f)
    except:
        return {}


def save_config(config):
    with open(CONFIG_FILE, "w") as f:
        json.dump(config, f, indent=2)


def register(device_id):
    """Register device and get API key."""
    resp = requests.post(f"{API_BASE}/api/register", json={
        "device_id": device_id,
        "platform": f"{platform.system()} {platform.machine()}"
    })
    data = resp.json()
    return data["api_key"]


def heartbeat(api_key):
    """Send heartbeat to keep device online."""
    try:
        requests.post(f"{API_BASE}/api/heartbeat",
                     headers={"Authorization": f"Bearer {api_key}"},
                     timeout=10)
        return True
    except:
        return False


def heartbeat_loop(api_key):
    """Background heartbeat thread."""
    while True:
        ok = heartbeat(api_key)
        status = "✅ connected" if ok else "⚠️ reconnecting..."
        # Update terminal title
        sys.stdout.write(f"\033]0;GPU Network — {status}\007")
        sys.stdout.flush()
        time.sleep(HEARTBEAT_INTERVAL)


def main():
    print()
    print("╔══════════════════════════════════════════════════╗")
    print("║        GPU — Free AI for Everyone                ║")
    print("║        Gifted People United                      ║")
    print("╚══════════════════════════════════════════════════╝")
    print()

    # Get or create device
    device_id = get_device_id()
    config = load_config()

    # Register or use existing key
    if "api_key" in config:
        api_key = config["api_key"]
        print(f"  Device:  {device_id[:8]}...")
        print(f"  Status:  Reconnecting...")
    else:
        print(f"  Device:  {device_id[:8]}... (new)")
        print(f"  Registering...")
        api_key = register(device_id)
        config["api_key"] = api_key
        save_config(config)
        print(f"  Status:  Registered ✅")

    # First heartbeat
    ok = heartbeat(api_key)
    if ok:
        print(f"  Status:  Connected ✅")
    else:
        print(f"  Status:  Connection failed ⚠️")

    print()
    print("  ┌────────────────────────────────────────────┐")
    print(f"  │  Your API Key: {api_key[:40]}  │")
    print("  └────────────────────────────────────────────┘")
    print()
    print("  Usage (OpenAI-compatible):")
    print()
    print(f'  curl {API_BASE}/v1/chat/completions \\')
    print(f'    -H "Authorization: Bearer {api_key}" \\')
    print(f'    -H "Content-Type: application/json" \\')
    print(f'    -d \'{{"model":"qwen2.5-14b","messages":[{{"role":"user","content":"Hello!"}}]}}\'')
    print()
    print("  Or use with any OpenAI client:")
    print(f'    openai.base_url = "{API_BASE}/v1"')
    print(f'    openai.api_key = "{api_key}"')
    print()
    print("  ─────────────────────────────────────────────")
    print("  Keep this running to maintain API access.")
    print("  Your device helps train free AI for everyone.")
    print("  Press Ctrl+C to disconnect.")
    print()

    # Start heartbeat in background
    t = threading.Thread(target=heartbeat_loop, args=(api_key,), daemon=True)
    t.start()

    # Keep alive
    try:
        while True:
            time.sleep(60)
            # Show periodic status
            now = time.strftime("%H:%M:%S")
            print(f"  [{now}] ⚡ Connected — API key active", end="\r")
    except KeyboardInterrupt:
        print()
        print()
        print("  Disconnected. Start again to reconnect.")
        print("  Your API key is saved — no need to re-register.")
        print()


if __name__ == "__main__":
    main()
