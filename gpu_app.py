#!/usr/bin/env python3
"""
GPU — Gifted People United
Free AI for Everyone

Desktop app: stays alive, sends heartbeat, shows API key.
Easter eggs inside. You'll find them.
"""

import os
import sys
import json
import time
import uuid
import math
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
        self.title_clicks = 0
        self.last_title_click = 0
        self.piper_shown = False
        self.github_hover_timer = None

        self.root = tk.Tk()
        self.root.title("GPU")
        self.root.configure(bg="#fafaf9")
        self.root.resizable(False, False)

        # Window size
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

        frame = tk.Frame(self.root, bg=bg, padx=32, pady=24)
        frame.pack(fill="both", expand=True)

        # Title (triple-click = Pied Piper easter egg)
        title_font = tkfont.Font(family="Helvetica", size=28, weight="bold")
        self.title_label = tk.Label(frame, text="GPU", font=title_font, bg=bg, fg=fg, cursor="hand2")
        self.title_label.pack(anchor="w")
        self.title_label.bind("<Button-1>", self.on_title_click)

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
        self.key_var = tk.StringVar(value=self.api_key)
        self.key_entry = tk.Entry(key_frame, font=key_font, bg="#f5f5f4", fg=fg,
                                  bd=0, highlightthickness=0, readonlybackground="#f5f5f4",
                                  state="readonly", textvariable=self.key_var)
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

        # Links with Rick Rubin tooltip
        link_frame = tk.Frame(frame, bg=bg)
        link_frame.pack(anchor="w")

        self.github_btn = tk.Label(link_frame, text="GitHub", font=tkfont.Font(size=10, underline=1),
                             bg=bg, fg=accent, cursor="hand2")
        self.github_btn.pack(side="left", padx=(0, 16))
        self.github_btn.bind("<Button-1>", lambda e: self.open_url("https://github.com/yukakust/GPU"))
        self.github_btn.bind("<Enter>", self.on_github_enter)
        self.github_btn.bind("<Leave>", self.on_github_leave)

        contact_btn = tk.Label(link_frame, text="Contact", font=tkfont.Font(size=10, underline=1),
                              bg=bg, fg=accent, cursor="hand2")
        contact_btn.pack(side="left")
        contact_btn.bind("<Button-1>", lambda e: self.open_url("mailto:kustyuka@gmail.com"))

        # Rick Rubin tooltip (hidden until hover)
        self.rick_label = tk.Label(frame, text="", font=tkfont.Font(size=8, slant="italic"),
                                   bg=bg, fg="#a8a29e")
        self.rick_label.pack(anchor="w")

        tk.Frame(frame, height=12, bg=bg).pack()

        # Pied Piper canvas (hidden, shown on triple-click)
        self.piper_canvas = tk.Canvas(frame, width=356, height=60, bg=bg, highlightthickness=0)
        self.piper_canvas.pack(anchor="w")
        # Initially empty — populated on triple-click

        tk.Frame(frame, height=8, bg=bg).pack()

        # Manifesto
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

        tk.Label(frame, text=manifesto,
                font=tkfont.Font(size=7), bg=bg, fg=muted,
                wraplength=356, justify="left").pack(anchor="w")

        tk.Frame(frame, height=8, bg=bg).pack()
        tk.Label(frame, text="Oh, and to Big Tech — thanks for the inspiration. We'll take it from here. \U0001f609",
                font=tkfont.Font(size=7, slant="italic"), bg=bg, fg=accent,
                wraplength=356, justify="left").pack(anchor="w")

    # --- Easter egg: Triple-click GPU → Pied Piper ---
    def on_title_click(self, event):
        now = time.time()
        if now - self.last_title_click < 0.5:
            self.title_clicks += 1
        else:
            self.title_clicks = 1
        self.last_title_click = now

        if self.title_clicks >= 3 and not self.piper_shown:
            self.piper_shown = True
            self.show_pied_piper()

    def show_pied_piper(self):
        c = self.piper_canvas
        c.delete("all")

        # Draw pixel piper man in green
        green = "#22c55e"
        dark_green = "#16a34a"

        # Head
        c.create_oval(30, 5, 44, 19, fill=green, outline="")
        # Body
        c.create_rectangle(33, 19, 41, 38, fill=dark_green, outline="")
        # Legs
        c.create_rectangle(33, 38, 37, 50, fill=dark_green, outline="")
        c.create_rectangle(37, 38, 41, 50, fill=dark_green, outline="")
        # Arms — one holding flute
        c.create_rectangle(41, 22, 55, 26, fill=dark_green, outline="")  # arm to flute
        # Flute
        c.create_rectangle(55, 20, 80, 24, fill="#d97706", outline="")

        # Musical notes floating away
        self.notes = []
        for i in range(5):
            x = 85 + i * 20
            y = 15 - (i % 3) * 8
            note = c.create_text(x, y, text="♪" if i % 2 == 0 else "♫",
                                font=tkfont.Font(size=10 + i), fill="#d97706")
            self.notes.append({"id": note, "x": x, "y": y, "phase": i * 0.5})

        # Animate notes
        self.animate_notes(0)

        # Text
        c.create_text(178, 45, text="Silicon Valley was a documentary.",
                      font=tkfont.Font(size=8, slant="italic"), fill="#78716c")

    def animate_notes(self, frame):
        if frame > 60 or not self.running:
            return
        c = self.piper_canvas
        for note in self.notes:
            # Float upward and rightward
            dy = -0.5 + math.sin(frame * 0.15 + note["phase"]) * 0.3
            dx = 0.3
            c.move(note["id"], dx, dy)
            # Fade effect via size change
            if frame > 40:
                c.itemconfig(note["id"], fill="#e7e5e4")
        self.root.after(50, lambda: self.animate_notes(frame + 1))

    # --- Easter egg: Rick Rubin tooltip on GitHub hover 500ms ---
    def on_github_enter(self, event):
        self.github_hover_timer = self.root.after(500, self.show_rick_rubin)

    def on_github_leave(self, event):
        if self.github_hover_timer:
            self.root.after_cancel(self.github_hover_timer)
            self.github_hover_timer = None
        self.rick_label.config(text="")

    def show_rick_rubin(self):
        self.rick_label.config(text='"The best work is made in the service of others." — Rick Rubin')

    # --- Core functions ---
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
