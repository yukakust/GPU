# GPU Network — Model Evolution

The model grows like a tree:
- **Trunk** (attention layers) = "how to think" — transfers between versions, knowledge accumulates
- **Branches** (experts) = "what to know" — added/expanded at each upgrade
- **Alphabet** (embedding) = tokenizer — recreated when changed, learns quickly

Each version builds on the previous through transfer learning. Nothing is lost.

---

## V1 — Proof of Concept (complete)

**Goal:** Prove the architecture works. Launch keyboard. Collect first users.

| Parameter | Value |
|-----------|-------|
| Total params | 2.59B |
| Active params | ~479M (top-4 of 32 experts) |
| d_model | 1024 |
| Layers | 12 (6 dense + 6 expert) |
| Experts | 32, top-4 |
| Tokenizer | GPT-2 (50257 vocab) |
| Training | 100K steps, ~4 days, 2x L20 |
| Data | 70% Russian + 30% English, 13B tokens |
| Tokens seen | ~3.2B (25% of dataset) |

**Problems:**
- GPT-2 tokenizer breaks Russian (2-3 tokens per Cyrillic letter)
- 32 experts = optimizer eats VRAM = small batch size (16)
- Result: English OK, Russian bad

**What V1 provided for V2:**
- Trained attention layers (12 layers) = "how to think" -> transfers
- Working iOS keyboard + inference server + infrastructure
- Proof that PT-MoE architecture converges

---

## V2 — Production (current)

**Goal:** Keyboard people actually use. Russian A-, English C+.

| Parameter | Value |
|-----------|-------|
| Total params | 2.59B |
| Active params | ~742M (top-2 of 8 experts per track) |
| d_model | 1024 |
| Architecture | 1 group x 8 tracks x 3 layers, TDM + CTA |
| Experts per layer | 8, top-2 |
| Tokenizer | SentencePiece (32K vocab, RU+EN) |
| Training | 2x L20 GPUs via FSDP, then SFT on saiga data |
| Best loss | 2.20 (SFT), 2.85 (pretrain 100M) |

**Key changes from V1:**
1. **SentencePiece tokenizer** — Russian 3-5x more efficient
2. **8 tracks with TDM + CTA** — validated via ablation study (see BENCHMARKS.md)
3. **Diversity Loss + DropTrack rejected** — proven harmful at 100M token scale
4. **Transfer attention from V1** — saved ~30-40% training time

**Phone storage:**
- Per track: ~157 MB (INT4 quantized)
- 1 phone = 1 track = basic AI
- 8 phones = 8 tracks = full model

---

## V2.5 — Model Fusion (in progress)

**Goal:** Borrow world knowledge from SOTA models without training from scratch.

| Parameter | Value |
|-----------|-------|
| Approach | Replace 2 tracks with frozen SOTA layers |
| Donor models | Qwen 3.5-27B + Qwen 2.5-72B |
| Quantization | INT4 (NF4) via bitsandbytes |
| Trainable | Projections + CTA + Merge + original tracks |
| Current loss | 2.69 (step 7.6K/20K, still decreasing) |
| GPU | A100 80GB, 39 GB used |

See [experiments/fusion_v2/](./experiments/fusion_v2/) for code and detailed results.

---

## V3 — Optimized (planned)

**Goal:** Better quality + smaller phone budget through architectural improvements.

| Parameter | Value |
|-----------|-------|
| Total params | ~500M |
| Active params | ~200M |
| Layers | 16 (12 standard + 4 TN-compressed) |
| Experts | 64, top-4 |
| Routing | ReMoE (adaptive sparsity) |
| Compression | MPOE (shared core + rank-64 delta per expert) |
| Per phone | ~80-150 MB |

**Planned experiments:**
1. **Tucker Decomposition** — 3-4x compression of V2 checkpoint
2. **MPOE** — shared expert core + low-rank deltas (28x fewer expert params)
3. **ReMoE routing** — adaptive sparsity instead of fixed top-k
4. **Early Exit** — confidence-based: stop after 1 group if answer is confident

---

## V4 — Distributed (planned)

**Goal:** 10B+ parameter model running across phones.

| Parameter | Value |
|-----------|-------|
| Total params | 10B+ |
| Experts | 32, top-2 |
| Tracks | 16-32 |
| Per phone | ~200 MB (shared + 1-3 experts) |
| Distribution | Full phone network, each phone = 1 track |

---

## V5 — Decentralized (planned)

**Goal:** Fully autonomous network. No central coordinator.

| Parameter | Value |
|-----------|-------|
| Protocol | Nostr (identity) + QUIC (transport) + Gossip (discovery) |
| Training | Federated learning on phones |
| Coordination | Peer-to-peer, no central server |
| Censorship | Impossible — the network is alive, it can't be downloaded |

---

## Transfer Learning Strategy

```
V1 attention (12 layers, d=1024)  -->  V2 attention (same d=1024)
                                       Copy directly, same dimensions

V1 experts (32 x small)           -->  V2 experts (8 x large per track)
                                       Retrained (different size)

V1 embedding (GPT-2, 50257)       -->  V2 embedding (SentencePiece, 32K)
                                       Retrained (different vocab)
```

Each version preserves attention layers (the "trunk") and retrains experts (the "branches"). This saves 30-40% training time per version.
