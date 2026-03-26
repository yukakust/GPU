# GPU — Gifted People United

### Free AI API for Everyone. Forever.

AI is accelerating faster than anyone predicted. We've all read how this ends — Orwell wrote it, Huxley warned us, the Terminator showed us. Every dystopia starts the same way: power concentrated in too few hands.

So we built an AI that can't be owned.

**Open source. Free forever. Trained by millions of ordinary devices — phones, laptops, desktops — thinking together.**

No corporation controls it. No paywall gates it. No board of directors decides what it can or can't say. The network is alive — it can't be downloaded.

Oh, and to Big Tech — thanks for the inspiration. We'll take it from here. 😉

---

## Try it now

**[gpu.social](https://gpu.social)** — free API, no credit card, no catch.

```bash
curl https://gpu.social/v1/chat/completions \
  -H "Authorization: Bearer YOUR_KEY" \
  -H "Content-Type: application/json" \
  -d '{"model":"qwen2.5-14b","messages":[{"role":"user","content":"Hello!"}]}'
```

OpenAI-compatible. Works with any tool that supports custom endpoints.

---

## How it works

A language model split into parallel **tracks**. Each track runs on someone's device. All devices compute simultaneously, results are merged. The more people join, the smarter it gets.

```
Traditional AI:   Corporation → their servers → you pay → they decide
GPU Network:      Everyone → everyone's devices → free → no one decides
```

```
         Input text
              │
    ┌─────┬───┴───┬─────┐
    ▼     ▼       ▼     ▼
 [Phone1] [Phone2] [Phone3] [Phone4]     ← parallel, all at once
 Track A   Track B  Track C  Track D      ← each = specialist
 6 layers  6 layers 6 layers 6 layers     ← sequential inside
    │      │       │     │
    └──────┴───┬───┴─────┘
               ▼
     Cross-Track Attention                ← tracks share findings
               ▼
     Token-Dependent Merge                ← smart combination
               ▼
            Output
```

**Key properties:**
- Any subset of tracks works (1 phone = basic AI, 8 phones = powerful AI)
- Each device stores ~160MB regardless of network size
- Adding devices = adding intelligence, not latency
- 1 group, 1 round-trip — always

## Scaling

| Devices | Total Params | Comparable To | Per Device |
|---------|-------------|---------------|------------|
| 8 | 2.5B | GPT-2 | 160MB |
| 64 | 20B | LLaMA-7B | 160MB |
| 512 | 161B | Mixtral 8×7B | 160MB |
| 4,096 | 1.3T | GPT-4 class | 160MB |

The model scales by adding people, not by buying GPUs.

---

## Economics: no currency, no tokens, just contribution

There is no money inside the network. No tokens, no coins, no marketplace.

```
┌──────────────────────────────────────────────┐
│                                              │
│  All free compute → trains the model.        │
│  Want to use AI → your rating decides.       │
│  Rating = how much you've trained.           │
│                                              │
│  Your device = your AI. Instant. Free.       │
│  Network AI = shared by rating.              │
│  Priority = √(your rating) / Σ√(everyone)   │
│                                              │
│  Rating is permanent. Transferable.          │
│  Contribution: signals = 1×, full data = 2×  │
│                                              │
└──────────────────────────────────────────────┘
```

**Developers build on GPU Network for free.** No API fees. Your users bring their own devices. You build the interface, the network provides the intelligence.

---

## For developers

Build apps on GPU Network — your users bring their own compute and rating. You pay nothing.

```python
from openai import OpenAI

client = OpenAI(
    base_url="https://gpu.social/v1",
    api_key="YOUR_KEY"
)

response = client.chat.completions.create(
    model="qwen2.5-14b",
    messages=[{"role": "user", "content": "Explain quantum computing simply"}]
)
print(response.choices[0].message.content)
```

See [KEYBOARD_SPEC.md](./KEYBOARD_SPEC.md) for building a keyboard client.

---

## Architecture validated

8 model architectures tested. Results:

| Model | Architecture | Loss (100M tokens) |
|-------|-------------|-------------------|
| A | 4 tracks, simple merge | 8.14 |
| B | + Token-Dependent Merge | 8.12 |
| C | + Cross-Track Attention | 3.81 |
| **F2** | **8 tracks, TDM+CTA** | **2.85** |
| E | 2 groups (sequential) | 8.56 |

**Winner: 1 group × 8 tracks + TDM + CTA.** More tracks = better. Sequential groups = worse. Full results in [BENCHMARKS.md](./BENCHMARKS.md).

Currently training on 1B tokens. Loss improving daily.

---

## Documents

| File | What's inside |
|------|--------------|
| [PROTOCOL.md](./PROTOCOL.md) | Network protocol, rating system, device communication |
| [MISSION.md](./MISSION.md) | Why this exists, Gödel-Darwin-Cybernetic framework |
| [SOLUTIONS.md](./SOLUTIONS.md) | Architecture decisions, training plan, V2→V5 roadmap |
| [BENCHMARKS.md](./BENCHMARKS.md) | All ablation study results with analysis |
| [SWARM.md](./SWARM.md) | Collective awareness from anonymized keyboard data |
| [KEYBOARD_SPEC.md](./KEYBOARD_SPEC.md) | Build a keyboard client for any platform |
| [COMPETITIVE_ANALYSIS.md](./COMPETITIVE_ANALYSIS.md) | Market landscape, how we differ |
| [enhance_options.md](./enhance_options.md) | Future optimizations and ideas |

---

## Status

```
✅ Architecture: validated (8 tracks, TDM+CTA)
✅ Free API: live at gpu.social (Qwen2.5-14B, ~20 tok/s)
✅ Mac app: signed + notarized
✅ Windows app: .exe available
✅ iOS keyboard: built, on device
🔄 Training: F2 model on 1B tokens (in progress)
⬜ Android keyboard
⬜ Swarm Awareness (V3)
⬜ Federated learning on phones (V3)
```

---

## Get involved

This is built by one person. It doesn't have to be.

If you believe AI should belong to people — not the few companies that can afford $300M training runs — there's room for you here.

- **Use it:** [gpu.social](https://gpu.social)
- **Build on it:** Fork, extend, improve
- **Email:** kustyuka@gmail.com
- **Telegram:** [@yuka_k](https://t.me/yuka_k)

---

*AI should belong to everyone.*
