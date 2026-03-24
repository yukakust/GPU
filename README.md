# GPU — Gifted People United

**Decentralized AI that belongs to everyone.**

Split a language model into parallel tracks. Each track runs on someone's phone. Phones compute simultaneously, results are merged. No central server, no corporation in the middle. The more people join, the smarter it gets.

```
Traditional AI:     Corporation owns model → their servers → you pay → they control
GPU:                Everyone owns a piece → everyone's phones → free → no one controls
```

## Core Ideas

- **Parallel Tracks Architecture** — Model split into N independent specialist tracks. Each track runs on one device. All compute in parallel, merge once. 1 round-trip latency regardless of network size.

- **Compute Miles** — Your device processes tokens for others → you earn miles. Spend miles on AI requests. Your own requests on your own device = free. The network runs on reciprocity, not money.

- **Self-Improving System** — Every keystroke teaches the model. Federated learning from millions of devices. The model gets smarter every day without any central training infrastructure.

- **Swarm Awareness** — Anonymized word frequency anomaly detection across the network. Detects earthquakes, floods, epidemics faster than news. A keyboard that tells you what's happening around you.

## Architecture Overview

```
         Input text
              │
    ┌─────┬───┴───┬─────┐
    ▼     ▼       ▼     ▼
 [Phone1] [Phone2] [Phone3] [Phone4]     ← parallel compute
 Track A   Track B  Track C  Track D      ← each track = specialist
 6 layers  6 layers 6 layers 6 layers     ← sequential inside device
    │      │       │     │
    └──────┴───┬───┴─────┘
               ▼
     Cross-Track Attention                ← tracks exchange information
               ▼
     Token-Dependent Merge                ← smart per-token combination
               ▼
            Output
```

**Key properties:**
- Any subset of tracks produces valid output (1, 2, 4, or 32 tracks)
- Weak phone = 1 track (150MB). Powerful phone = 4 tracks. Computer = 8+.
- Adding phones = adding intelligence, not adding latency
- Hierarchical merge for 64+ tracks (each merge layer handles max 8)

## Scaling

| Tracks | Total Params | Comparable To | Per Device |
|--------|-------------|---------------|------------|
| 8 | 2.5B | GPT-2 | 160MB |
| 16 | 5B | — | 160MB |
| 64 | 20B | LLaMA-7B | 160MB |
| 512 | 161B | Mixtral 8x7B | 160MB |

Each device stores the same 160MB regardless of network size. The model scales by adding devices, not by making devices heavier.

## Compute Miles Protocol

Each device earns Compute Miles proportional to tokens processed through its local track layers.

```
1 Compute Mile = 1 token × 1 layer processed on your device
```

Earning multiplier adjusts based on network size:

| Network Size | Multiplier |
|-------------|-----------|
| 0 — 100 | 50.0 |
| 100 — 1K | 24.0 |
| 1K — 10K | 12.0 |
| 10K — 100K | 6.0 |
| 100K — 1M | 3.0 |
| 1M+ | 1.0 |

Device multiplier uses logarithmic scaling to prevent concentration:

```
multiplier = 1 + log2(tracks_on_device)

1 track:   1.0×
2 tracks:  2.0×
4 tracks:  3.0×
8 tracks:  4.0×
```

Miles can be transferred between devices:

```
POST /miles/transfer
{ "from": device_id, "to": device_id, "amount": int, "sig": ed25519_sig }
```

Transfers are irreversible. No fee is applied.

## Data Contribution Tiers

```
Private (default):        Signals only (predicted vs actual token ID)    +0% miles
Help Train:               Signals + full context (token IDs)             +50% miles
Maximum (V3+):            Signals + context + local gradients            +100% miles
```

Users choose their tier. More data contribution = more miles. All tiers get the same free AI.

## Documents

- [MISSION.md](./MISSION.md) — Why this exists
- [SOLUTIONS.md](./SOLUTIONS.md) — Technical architecture decisions
- [PROTOCOL.md](./PROTOCOL.md) — Network protocol specification
- [SWARM.md](./SWARM.md) — Swarm Awareness system design
- [COMPETITIVE_ANALYSIS.md](./COMPETITIVE_ANALYSIS.md) — Landscape and differentiation

## Status

Active research. Architecture validated through ablation studies. Training in progress.

## Contact

Building this solo. Looking for collaborators who believe AI should belong to people, not corporations.

- **Email:** kustyuka@gmail.com
- **Telegram:** @yuka_k
- **GitHub:** [@yukakust](https://github.com/yukakust)
