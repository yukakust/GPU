#!/usr/bin/env python3
"""
GPU — Gifted People United
Free AI for Everyone

Simple desktop app: stays alive, sends heartbeat, shows API key.
"""

import os
import sys
import json
import time
import uuid
import platform
import threading
import tkinter as tk
from tkinter import font as tkfont
import urllib.request
import urllib.error

API_BASE = "https://gpu.social"
CONFIG_DIR = os.path.expanduser("~/.gpu")
CONFIG_FILE = os.path.join(CONFIG_DIR, "config.json")


def load_config():
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE) as f:
            return json.load(f)
    return {}


def save_config(config):
    os.makedirs(CONFIG_DIR, exist_ok=True)
    with open(CONFIG_FILE, "w") as f:
        json.dump(config, f, indent=2)


def api_post(path, data=None, headers=None):
    req = urllib.request.Request(
        f"{API_BASE}{path}",
        data=json.dumps(data).encode() if data else None,
        headers={**(headers or {}), "Content-Type": "application/json"},
        method="POST"
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            return json.loads(resp.read())
    except Exception as e:
        return {"error": str(e)}


def register():
    config = load_config()
    if config.get("api_key"):
        return config

    device_id = str(uuid.uuid4())
    plat = f"{platform.system()}-{platform.machine()}"
    
    result = api_post("/api/register", {
        "device_id": device_id,
        "platform": plat,
    })
    
    if "api_key" in result:
        config = {
            "device_id": device_id,
            "api_key": result["api_key"],
            "platform": plat,
        }
        save_config(config)
        return config
    return {"api_key": "registration_failed", "error": result.get("error", "unknown")}


class GPUApp:
    def __init__(self):
        self.config = register()
        self.api_key = self.config.get("api_key", "")
        self.running = True
        
        self.root = tk.Tk()
        self.root.title("GPU")
        self.root.configure(bg="#fafaf9")
        self.root.resizable(False, False)
        
        # Window size
        w, h = 420, 640
        x = (self.root.winfo_screenwidth() - w) // 2
        y = (self.root.winfo_screenheight() - h) // 3
        self.root.geometry(f"{w}x{h}+{x}+{y}")
        
        self.build_ui()
        self.start_heartbeat()
        
    def build_ui(self):
        bg = "#fafaf9"
        fg = "#1c1917"
        muted = "#78716c"
        accent = "#d97706"
        
        frame = tk.Frame(self.root, bg=bg, padx=32, pady=24)
        frame.pack(fill="both", expand=True)
        
        # Title
        title_font = tkfont.Font(family="Helvetica", size=28, weight="bold")
        tk.Label(frame, text="GPU", font=title_font, bg=bg, fg=fg).pack(anchor="w")
        
        sub_font = tkfont.Font(family="Helvetica", size=11)
        tk.Label(frame, text="Gifted People United", font=sub_font, bg=bg, fg=muted).pack(anchor="w")
        
        tk.Frame(frame, height=24, bg=bg).pack()
        
        # Status
        status_frame = tk.Frame(frame, bg=bg)
        status_frame.pack(anchor="w")
        
        self.status_dot = tk.Canvas(status_frame, width=10, height=10, bg=bg, highlightthickness=0)
        self.status_dot.create_oval(1, 1, 9, 9, fill="#22c55e", outline="")
        self.status_dot.pack(side="left", padx=(0, 6))
        
        self.status_label = tk.Label(status_frame, text="Connected", 
                                      font=tkfont.Font(size=12), bg=bg, fg="#22c55e")
        self.status_label.pack(side="left")
        
        tk.Frame(frame, height=24, bg=bg).pack()
        
        # API Key
        tk.Label(frame, text="YOUR API KEY", font=tkfont.Font(size=9, weight="bold"),
                bg=bg, fg=muted).pack(anchor="w")
        tk.Frame(frame, height=4, bg=bg).pack()
        
        key_frame = tk.Frame(frame, bg="#f5f5f4", padx=12, pady=10,
                            highlightbackground="#e7e5e4", highlightthickness=1)
        key_frame.pack(fill="x")
        
        key_font = tkfont.Font(family="Courier", size=11)
        self.key_entry = tk.Entry(key_frame, font=key_font, bg="#f5f5f4", fg=fg,
                                  bd=0, highlightthickness=0, readonlybackground="#f5f5f4",
                                  state="readonly", textvariable=tk.StringVar(value=self.api_key))
        self.key_entry.pack(side="left", fill="x", expand=True)
        
        copy_btn = tk.Button(key_frame, text="Copy", font=tkfont.Font(size=10),
                            bg="#f5f5f4", fg=accent, bd=0, cursor="hand2",
                            activebackground="#f5f5f4", activeforeground=fg,
                            command=self.copy_key)
        copy_btn.pack(side="right")
        
        tk.Frame(frame, height=20, bg=bg).pack()
        
        # Base URL
        tk.Label(frame, text="BASE URL", font=tkfont.Font(size=9, weight="bold"),
                bg=bg, fg=muted).pack(anchor="w")
        tk.Label(frame, text=f"{API_BASE}/v1", font=tkfont.Font(family="Courier", size=11),
                bg=bg, fg=fg).pack(anchor="w")
        
        tk.Frame(frame, height=8, bg=bg).pack()
        
        tk.Label(frame, text="MODEL", font=tkfont.Font(size=9, weight="bold"),
                bg=bg, fg=muted).pack(anchor="w")
        tk.Label(frame, text="qwen2.5-14b", font=tkfont.Font(family="Courier", size=11),
                bg=bg, fg=fg).pack(anchor="w")
        
        tk.Frame(frame, height=24, bg=bg).pack()
        
        # Info
        info_font = tkfont.Font(size=10)
        tk.Label(frame, text="Works with any OpenAI-compatible tool.",
                font=info_font, bg=bg, fg=muted).pack(anchor="w")
        tk.Label(frame, text="Keep this app running to use the API.",
                font=info_font, bg=bg, fg=muted).pack(anchor="w")
        
        tk.Frame(frame, height=20, bg=bg).pack()
        
        # Links
        link_frame = tk.Frame(frame, bg=bg)
        link_frame.pack(anchor="w")
        
        github_btn = tk.Label(link_frame, text="GitHub", font=tkfont.Font(size=10, underline=1),
                             bg=bg, fg=accent, cursor="hand2")
        github_btn.pack(side="left", padx=(0, 16))
        github_btn.bind("<Button-1>", lambda e: self.open_url("https://github.com/yukakust/GPU"))
        
        contact_btn = tk.Label(link_frame, text="Contact", font=tkfont.Font(size=10, underline=1),
                              bg=bg, fg=accent, cursor="hand2")
        contact_btn.pack(side="left")
        contact_btn.bind("<Button-1>", lambda e: self.open_url("mailto:kustyuka@gmail.com"))
        
        # Manifesto
        tk.Frame(frame, height=20, bg=bg).pack()

        manifesto = (
            "AI is accelerating faster than anyone predicted. "
            "We've all read how this ends — Orwell wrote it, "
            "the Terminator showed us. Every dystopia starts "
            "the same way: power in too few hands.\n\n"
            "So we built an AI that can't be owned. Open source. "
            "Free forever. Trained by millions of devices "
            "thinking together.\n\n"
            "No corporation controls it. No paywall gates it.\n\n"
            "Help us train the first AI that truly belongs "
            "to everyone."
        )

        manifesto_label = tk.Label(frame, text=manifesto,
                font=tkfont.Font(size=7), bg=bg, fg=muted,
                wraplength=356, justify="left")
        manifesto_label.pack(anchor="w")

        tk.Frame(frame, height=8, bg=bg).pack()
        tk.Label(frame, text="Oh, and to Big Tech — thanks for the inspiration. We'll take it from here. 😉",
                font=tkfont.Font(size=7, slant="italic"), bg=bg, fg=accent,
                wraplength=356, justify="left").pack(anchor="w")
    
    def copy_key(self):
        self.root.clipboard_clear()
        self.root.clipboard_append(self.api_key)
        self.status_label.config(text="Copied!", fg="#d97706")
        self.root.after(1500, lambda: self.status_label.config(text="Connected", fg="#22c55e"))
    
    def open_url(self, url):
        import webbrowser
        webbrowser.open(url)
    
    def start_heartbeat(self):
        def hb():
            while self.running:
                result = api_post("/api/heartbeat", headers={
                    "Authorization": f"Bearer {self.api_key}"
                })
                if "error" in result:
                    self.root.after(0, lambda: self.status_label.config(text="Reconnecting...", fg="#d97706"))
                else:
                    self.root.after(0, lambda: self.status_label.config(text="Connected", fg="#22c55e"))
                time.sleep(180)
        
        t = threading.Thread(target=hb, daemon=True)
        t.start()
    
    def run(self):
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        self.root.mainloop()
    
    def on_close(self):
        self.running = False
        self.root.destroy()


def main():
    app = GPUApp()
    app.run()


if __name__ == "__main__":
    main()
