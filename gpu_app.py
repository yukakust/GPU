#!/usr/bin/env python3
"""
GPU — Gifted People United
Free AI for Everyone

Desktop app: connects to GPU network, shows API key, model selector.
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
    result = api_post("/api/register", {"device_id": device_id, "platform": plat, "mode": "new"})
    if "api_key" in result:
        config = {"device_id": device_id, "api_key": result["api_key"], "platform": plat, "model": "smart"}
        save_config(config)
        return config
    return {"api_key": "registration_failed", "model": "smart"}


class GPUApp:
    def __init__(self):
        self.config = register()
        self.api_key = self.config.get("api_key", "")
        self.current_model = self.config.get("model", "smart")
        self.running = True

        self.root = tk.Tk()
        self.root.title("GPU")
        self.root.configure(bg="#fafaf9")
        self.root.resizable(False, False)

        w, h = 420, 680
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

        frame = tk.Frame(self.root, bg=bg, padx=28, pady=20)
        frame.pack(fill="both", expand=True)

        # Title (triple-click easter egg)
        self.title_label = tk.Label(frame, text="GPU", font=tkfont.Font(size=28, weight="bold"),
                                    bg=bg, fg=fg)
        self.title_label.pack(anchor="w")
        self.title_clicks = 0
        self.title_label.bind("<Button-1>", self._title_click)

        tk.Label(frame, text="Gifted People United", font=tkfont.Font(size=11),
                bg=bg, fg=muted).pack(anchor="w")

        tk.Frame(frame, height=16, bg=bg).pack()

        # Status
        sf = tk.Frame(frame, bg=bg)
        sf.pack(anchor="w")
        self.status_dot = tk.Canvas(sf, width=10, height=10, bg=bg, highlightthickness=0)
        self.status_dot.create_oval(1, 1, 9, 9, fill="#22c55e", outline="")
        self.status_dot.pack(side="left", padx=(0, 6))
        self.status_label = tk.Label(sf, text="Connected", font=tkfont.Font(size=11),
                                      bg=bg, fg="#22c55e")
        self.status_label.pack(side="left")

        tk.Frame(frame, height=16, bg=bg).pack()

        # API Key
        tk.Label(frame, text="YOUR API KEY", font=tkfont.Font(size=9, weight="bold"),
                bg=bg, fg=muted).pack(anchor="w")
        tk.Frame(frame, height=4, bg=bg).pack()

        kf = tk.Frame(frame, bg="#f5f5f4", padx=12, pady=10,
                     highlightbackground="#e7e5e4", highlightthickness=1)
        kf.pack(fill="x")
        self.key_var = tk.StringVar(value=self.api_key)
        tk.Entry(kf, font=tkfont.Font(family="Courier", size=10), bg="#f5f5f4", fg=fg,
                bd=0, highlightthickness=0, readonlybackground="#f5f5f4",
                state="readonly", textvariable=self.key_var).pack(side="left", fill="x", expand=True)
        tk.Button(kf, text="Copy", font=tkfont.Font(size=10), bg="#f5f5f4", fg=accent,
                 bd=0, cursor="hand2", command=self.copy_key).pack(side="right")

        tk.Frame(frame, height=16, bg=bg).pack()

        # Model selector
        tk.Label(frame, text="MODEL", font=tkfont.Font(size=9, weight="bold"),
                bg=bg, fg=muted).pack(anchor="w")
        tk.Frame(frame, height=6, bg=bg).pack()

        mf = tk.Frame(frame, bg=bg)
        mf.pack(fill="x")

        self.smart_btn = tk.Button(mf, text="🧠 Smart", font=tkfont.Font(size=11, weight="bold"),
                                    bg="#1c1917", fg="#fafaf9", bd=0, padx=16, pady=8,
                                    cursor="hand2", command=lambda: self.set_model("smart"))
        self.smart_btn.pack(side="left", padx=(0, 8))

        self.fast_btn = tk.Button(mf, text="⚡ Fast", font=tkfont.Font(size=11, weight="bold"),
                                   bg="#f5f5f4", fg="#1c1917", bd=0, padx=16, pady=8,
                                   cursor="hand2", command=lambda: self.set_model("fast"))
        self.fast_btn.pack(side="left")

        tk.Frame(frame, height=4, bg=bg).pack()
        self.model_desc = tk.Label(frame, text="Qwen3.5-122B · Opus 4+ · ~5 tok/s",
                                    font=tkfont.Font(size=9), bg=bg, fg=muted)
        self.model_desc.pack(anchor="w")

        tk.Frame(frame, height=12, bg=bg).pack()

        # Base URL
        tk.Label(frame, text="BASE URL", font=tkfont.Font(size=9, weight="bold"),
                bg=bg, fg=muted).pack(anchor="w")
        tk.Label(frame, text=f"{API_BASE}/v1", font=tkfont.Font(family="Courier", size=10),
                bg=bg, fg=fg).pack(anchor="w")

        tk.Frame(frame, height=20, bg=bg).pack()

        # Info
        tk.Label(frame, text="Works with any OpenAI-compatible tool.",
                font=tkfont.Font(size=10), bg=bg, fg=muted).pack(anchor="w")
        tk.Label(frame, text="Keep this app running to use the API.",
                font=tkfont.Font(size=10), bg=bg, fg=muted).pack(anchor="w")

        tk.Frame(frame, height=16, bg=bg).pack()

        # Links
        lf = tk.Frame(frame, bg=bg)
        lf.pack(anchor="w")

        self.gh_label = tk.Label(lf, text="GitHub", font=tkfont.Font(size=10, underline=1),
                                 bg=bg, fg=accent, cursor="hand2")
        self.gh_label.pack(side="left", padx=(0, 16))
        self.gh_label.bind("<Button-1>", lambda e: self.open_url("https://github.com/yukakust/GPU"))
        # Rick Rubin tooltip on hover
        self.tooltip = None
        self.hover_timer = None
        self.gh_label.bind("<Enter>", self._gh_enter)
        self.gh_label.bind("<Leave>", self._gh_leave)

        tk.Label(lf, text="Contact", font=tkfont.Font(size=10, underline=1),
                bg=bg, fg=accent, cursor="hand2").pack(side="left")

        tk.Frame(frame, height=16, bg=bg).pack()

        # Footer
        tk.Label(frame, text="The network is alive — it can't be downloaded.",
                font=tkfont.Font(size=9, slant="italic"), bg=bg, fg=accent).pack(anchor="w")

    def set_model(self, model):
        self.current_model = model
        self.config["model"] = model
        save_config(self.config)

        if model == "smart":
            self.smart_btn.configure(bg="#1c1917", fg="#fafaf9")
            self.fast_btn.configure(bg="#f5f5f4", fg="#1c1917")
            self.model_desc.configure(text="Qwen3.5-122B · Opus 4+ · ~5 tok/s")
        else:
            self.fast_btn.configure(bg="#1c1917", fg="#fafaf9")
            self.smart_btn.configure(bg="#f5f5f4", fg="#1c1917")
            self.model_desc.configure(text="Qwen2.5-14B · GPT-4o-mini · ~20 tok/s")

    def copy_key(self):
        self.root.clipboard_clear()
        self.root.clipboard_append(self.api_key)
        self.status_label.config(text="Copied!", fg="#d97706")
        self.root.after(1500, lambda: self.status_label.config(text="Connected", fg="#22c55e"))

    def open_url(self, url):
        import webbrowser
        webbrowser.open(url)

    def _title_click(self, event):
        self.title_clicks += 1
        if self.title_clicks >= 3:
            self.title_clicks = 0
            self._pied_piper()

    def _pied_piper(self):
        # Easter egg: Pied Piper reference
        self.title_label.config(text="♪ GPU ♪", fg="#22c55e")
        self.root.after(2000, lambda: self.title_label.config(text="GPU", fg="#1c1917"))

    def _gh_enter(self, event):
        self.hover_timer = self.root.after(500, self._show_tooltip)

    def _gh_leave(self, event):
        if self.hover_timer:
            self.root.after_cancel(self.hover_timer)
        if self.tooltip:
            self.tooltip.destroy()
            self.tooltip = None

    def _show_tooltip(self):
        if self.tooltip:
            return
        self.tooltip = tk.Toplevel(self.root)
        self.tooltip.wm_overrideredirect(True)
        x = self.gh_label.winfo_rootx()
        y = self.gh_label.winfo_rooty() - 30
        self.tooltip.wm_geometry(f"+{x}+{y}")
        tk.Label(self.tooltip,
                text='"The best work is made in the service of others." — Rick Rubin',
                font=tkfont.Font(size=9, slant="italic"),
                bg="#1c1917", fg="#d97706", padx=8, pady=4).pack()

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
        threading.Thread(target=hb, daemon=True).start()

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
