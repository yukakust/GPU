#!/usr/bin/env python3
"""
Probe a Fediverse instance for what the GPU bot needs to federate with it.

Usage: python experiments/activitypub/check_instance.py [defcon.social] [user]

Reports the server software, post length limit, whether authorized fetch
("secure mode") is on — then the bot MUST sign its GETs, which pubby does —
and resolves a user via WebFinger.
"""

import sys
import requests

domain = sys.argv[1] if len(sys.argv) > 1 else "defcon.social"
user = sys.argv[2] if len(sys.argv) > 2 else "defcon"
AP = {"Accept": "application/activity+json"}

links = requests.get(f"https://{domain}/.well-known/nodeinfo", timeout=10).json()["links"]
info = requests.get(links[-1]["href"], timeout=10).json()
print(f"software:        {info['software']['name']} {info['software']['version']}")

try:
    inst = requests.get(f"https://{domain}/api/v2/instance", timeout=10).json()
    print(f"max characters:  {inst['configuration']['statuses']['max_characters']}")
except Exception as e:
    print(f"max characters:  unknown ({e})")

wf = requests.get(f"https://{domain}/.well-known/webfinger",
                  params={"resource": f"acct:{user}@{domain}"}, timeout=10).json()
actor = next(link["href"] for link in wf["links"]
             if link.get("rel") == "self" and "json" in link.get("type", ""))
print(f"webfinger:       @{user}@{domain} -> {actor}")

code = requests.get(actor, headers=AP, timeout=10).status_code
print(f"unsigned GET:    HTTP {code} -> authorized fetch "
      f"{'ON (signed requests required)' if code == 401 else 'off'}")
