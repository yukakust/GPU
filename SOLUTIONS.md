# PT-MoE Solutions — Brainstorm Session

## Key Architectural Decision: Minimize Groups, Maximize Tracks

### Core Idea
Group = sequential round-trip over network = +50-100ms latency.
Track = parallel compute unit on a separate phone = free compute.

**Goal: minimize groups (1-2), maximize tracks (4-32+).**

### Architecture: 1-2 Groups x N Tracks

```
Fast (autocomplete):   1 group × 1-2 tracks  = local, ~30-50ms
Normal (suggestions):  1 group × 4 tracks     = 1 round-trip, ~150-200ms
Smart (complex answer): 2 groups × 4+ tracks  = 2 round-trips, ~300-400ms
```

With 1 group:
```
     Input (text)
          │
  ┌───┬───┴───┬───┐
  ▼   ▼       ▼   ▼
[T1] [T2]   [T3] [T4]     ← in parallel on 4 phones
 6L   6L     6L   6L        each computes its layers internally
  │   │       │   │
  └───┴───┬───┴───┘
          ▼
 Cross-Track Attention      ← information exchange between tracks
          ▼
 Token-Dependent Merge      ← smart aggregation
          ▼
       Response
```

1 broadcast → all compute in parallel → 1 gather → CTA + merge → done.

With 2 groups — the same thing twice: first round "what is the question about?", second "here is the answer considering context from all".

### Terminology

**Group** = a round of discussion between phones. All phones return their results → merge → if needed, another round. Each group = 1 network round-trip.

**Track** = one specialist phone. Inside — sequential layers (6-12). The phone computes them quickly, locally, ~50-100ms.

**Variable number of groups:**
```
0 groups:  Local, 1 track, no network.          "Hi" → "there"            ~30ms
1 group:   Asked N phones, merge.                "What's the weather?" → answer  ~150-200ms
2 groups:  Two rounds of discussion.             "Write a function..."     ~300-400ms
3 groups:  Three rounds (max, for complex).      "Analyze..."              ~500ms
```

Number of groups is chosen automatically by entropy after each round:
- entropy < 0.3 after round 1 → stop, answer is confident
- entropy > 0.3 → need another round

OR set by application: keyboard = 0-1, chat = 1-2, API = 2-3.

### Redundancy x2 with Priority Routing

Each track is sent to 2 phones: top-1 expert (best) and top-2 expert (second best).

```
Query: "write a sorting function in Python"

Router:
  Code   → top-1: Masha's phone  |  top-2: Petya's phone    (backup)
  Lang   → top-1: Kolya's phone  |  top-2: Dima's phone     (backup)
  Know   → top-1: Anya's phone   |  top-2: Sasha's phone    (backup)
  Style  → top-1: Lena's phone   |  top-2: Misha's phone    (backup)

Send to all 8. Take the first response from each track type.
Masha is stuck → Petya is already computing → take his. No delay.
```

Token-dependent merge automatically accounts for top-2 arriving instead of top-1 (slightly lower weight).
Cost: x2 compute. Benefit: resilience + speed.

### Justification (Papers)
- **Branch-Train-Merge** (Meta, 2022): 22.4B model without sequential dependency = 2.5x cheaper than standard at the same quality
- **ParaFormer** (2025): quality is determined by inter-branch collaboration, not depth
- **Kraken** (NeurIPS 2024): parallel branches for multi-device inference, +35.6% speedup
- **Transformers & Logarithmic Depth** (2024): logarithmic depth is sufficient for many tasks

---

## Solution 1: Token-Dependent Merge

**Problem:** Current merge = fixed scalar weights. All tokens get the same mix of tracks.

**Solution:** `gate = softmax(Linear(d_model, num_tracks))` — each token chooses which tracks matter more.

```python
# Before: static weights
weights = softmax(self.gate_weights)  # [num_tracks] — same for all tokens
output = sum(w * track_out for w, track_out in zip(weights, track_outputs))

# After: token-dependent
# hidden shape: [batch, seq_len, d_model]
weights = softmax(self.gate_proj(hidden))  # [batch, seq_len, num_tracks]
output = sum(w.unsqueeze(-1) * track_out for w, track_out in zip(weights.unbind(-1), track_outputs))
```

**Overhead:** +768 x num_tracks parameters (negligible).

**Proven:** Branchformer (ICML 2022) — exactly this approach, outperforms both Transformer and Conformer.

---

## Solution 2: Cross-Track Attention

**Problem:** With 1 group, each track works blind — doesn't know what others computed.

**Solution:** Before merge, tracks attend to each other.

```python
# track_outputs: list of [batch, seq_len, d_model], len = num_tracks
stacked = torch.stack(track_outputs, dim=2)  # [batch, seq_len, num_tracks, d_model]

# For each position: 4 vectors (from 4 tracks) attend to each other
# Multi-head attention: Q, K, V all from stacked
# Result: each track "knows" what the others computed

cross_attn_out = self.cross_attention(stacked)  # same shape
track_outputs = cross_attn_out.unbind(dim=2)    # list of [batch, seq_len, d_model]
```

**Overhead:** ~4 x d_model^2 = 4 x 768^2 ~ 2.4M params (with 4 heads). Computed on the coordinator, not on the phone.

**Justification:** ParaFormer showed that inter-branch communication is the key to quality in parallel architectures.

---

## Solution 3: Track Specialization (Diversity Loss + DropTrack)

**Problem:** Tracks see the same input, receive the same gradients → learn the same thing.

**Solution A — Diversity Loss:** Penalty for cosine similarity of track **outputs**.

```python
# IMPORTANT: penalize OUTPUTS (activations), not weights.
# Paper "Geometric Regularization" (2025) proved that on weights this DOES NOT work.
diversity_loss = 0
for i in range(num_tracks):
    for j in range(i+1, num_tracks):
        cos_sim = F.cosine_similarity(track_outputs[i], track_outputs[j], dim=-1)
        diversity_loss += cos_sim.mean()
diversity_loss *= lambda_diversity  # 0.01-0.1
```

**Solution B — DropTrack:** Randomly disable 1+ track during training (like DropPath in vision).

```python
if self.training:
    drop_mask = torch.bernoulli(torch.full((num_tracks,), 1 - drop_rate))
    # Guarantee at least 1 track is alive
    if drop_mask.sum() == 0:
        drop_mask[torch.randint(num_tracks, (1,))] = 1
    track_outputs = [out * mask for out, mask in zip(track_outputs, drop_mask)]
```

**DropTrack Effect:**
1. Each track is forced to be useful on its own
2. Model works with ANY subset of tracks at inference
3. Natural specialization: if track 1 is already good at language, track 2 is "forced" to learn something else

---

## Solution 4: Any Subset of Tracks is Valid (Tiered System)

**Problem:** Different devices have different capabilities.

**Solution:** DropTrack during training guarantees that the model works with any subset of tracks.

### Device Tiers

| Tier | Device | Tracks | RAM | Storage | Capability |
|------|--------|:------:|:---:|:-------:|-----------|
| 1 | Low-end phone | 1 | ~50MB | ~150MB | Autocomplete, simple tasks |
| 2 | Mid-range phone | 2 | ~100MB | ~300MB | Suggestions, medium tasks |
| 3 | High-end phone | 4 | ~200MB | ~600MB | Full model, complex tasks |
| 4 | Desktop / server | 8-32 | 1-4GB | 1.5-5GB | Maximum intelligence |

### Rating Growth by Device Tier

More tracks = more training compute = higher rating = higher priority.

| Device | Tracks | Training compute | Rating growth |
|--------|:------:|:----------------:|:-------------:|
| Tier 1 (low-end phone) | 1 | ~30ms/token | Slow but steady |
| Tier 2 (mid-range) | 2 | ~60ms/token | 2x faster rating growth |
| Tier 3 (high-end) | 4 | ~120ms/token | 4x faster rating growth |
| Tier 4 (desktop/server) | 8-32 | varies | Fastest rating growth |

No multipliers, no bonuses. 1 token trained = 1 rating point (signals mode) or 2 rating points (full data mode). More tracks simply means more tokens processed per unit of time.

### How Token-Dependent Merge Works with Incomplete Track Sets

```python
# available_tracks — mask of available tracks [1, 0, 1, 1] = tracks 0, 2, 3
weights = softmax(self.gate_proj(hidden))  # [batch, seq, num_tracks]
weights = weights * available_tracks       # zero out unavailable
weights = weights / weights.sum(dim=-1, keepdim=True)  # renormalize
```

Merge automatically redistributes weights to available tracks.

---

## Solution 5: Scaling Tracks = Scaling Intelligence

Each new track = a new specialist in the network. Scales linearly:

```
4 tracks:   Language + Knowledge + Domain + Style              → base model
8 tracks:   + Reasoning + Code + Math + Creative               → advanced
16 tracks:  + Medical + Legal + Finance + Science + ...         → expert
32 tracks:  + personal user experts                             → unique
64+ tracks: each user = their own expert                        → infinite growth
```

Token-dependent merge scales: Linear(d_model, N) — adding a track = adding a column to the weight matrix.

### Adding a New Track Without Retraining the Entire Model

1. Initialize new track from nearest existing one
2. Expand merge projection: add column (init ~0)
3. Fine-tune only new track + merge layer
4. Existing tracks are untouched

This is the BTX approach (Meta, 2024): Branch → Train independently → Mix via MoE routing.

---

## Solution 6: Adaptive Depth (Variable Number of Groups)

### Product Tiers (Explicit Choice, Not Automatic)

```
Keyboard:    0-1 group    free (participant contributes compute)
Chat/Babel:  1-2 groups   free (participant contributes compute + translation data)
API Light:   2 groups     free, priority by rating
API Full:    3 groups     free, priority by rating
API Deep:    4 groups     free, priority by rating (API without device = last in queue)
```

Application/user chooses how many rounds. No entropy magic.

### Why NOT Entropy for Choosing Number of Groups

1. Entropy after 1 group != entropy after all groups (shallow model can be "confident" in garbage)
2. Threshold calibration depends on model size, data, task — brittle
3. LM head between groups = blocking +5-10ms

### V3+ Optimization: Confidence Head

Small MLP (768 → 64 → 1) predicts "is another round needed?". Trained jointly with the model.

```python
confidence = sigmoid(self.confidence_head(hidden.mean(dim=1)))
if confidence > 0.8:
    break  # save a round-trip
```

API Full = up to 3 groups, but confidence head may stop after 2. Often delivers results with fewer rounds than the maximum.

### For Training

Loss on the output of EVERY group (intermediate exit heads), not just the final one. The model learns to produce useful output after any number of rounds.

---

## Solution 7: Streaming Between Groups (Kraken-style)

With 2+ groups — overlap compute and network:

```
Standard:    [compute 100ms] → [send 50ms] → [merge] → [compute 100ms] → [send 50ms]
Streaming:   [compute L1-L3] → [send partial] → [compute L4-L6] → [send rest]
                                    ↕ (in parallel)
             Coordinator already starts receiving data while phone finishes computing
```

With 3 groups, savings: ~30-40% network latency (overlap instead of sequential).

Critical for API Full/Deep (3-4 groups). Not needed for keyboard (0-1 group).

---

## Solution 8: Rating — Network Economics

### Formula
```
1 rating point = 1 token trained for the network on your device
```

Rating — permanent, never decreases. It's a total counter of how much you've contributed to the network.

### Rules
- **Everything is free for participants.** If you have the app, you get AI — no spending, no balances.
- **Rating = priority.** Higher rating → higher priority in the queue.
- **Priority formula:** `priority_share = √(your_rating) / Σ√(all_ratings)` — square root ensures diminishing returns for whales.
- **API without device = last in queue.** You can use the API without contributing compute, but you wait behind everyone who does.
- **Developers build free** on the GPU Network — the API is open, priority is the only differentiator.

### How Rating Grows
```
Signals mode (default):       1× rating per token trained
Full data mode (opt-in):      2× rating per token trained
```

### Rating Display
Rating shown as a number with ⚡ in settings. People compare, compete, show off. Transferable to family/friends.

### Battery / Internet Rules

Like sleep for humans — work during the day, brain processes at night. Phone charges → trains on everything from today → wakes up smarter.

```
Charging + WiFi:       training (day's data) + inference for network + data sync
Charging + cellular:   training (day's data) + inference for network (KB only)
Battery + WiFi:        inference for network + sync pending rating
Battery + cellular:    inference for network only (KB per request)
Offline:               own AI only, data accumulates, pending_rating queued
```

Key principles:
- **Training = only while charging.** Plugged in at night → phone learns from everything typed today → smarter by morning.
- **Inference = always** (except offline). Serving network requests = kilobytes of text, negligible traffic/battery.
- **No user settings.** It just works. User never thinks about it.
- **Pending rating:** training done while charging queues up. Syncs to coordinator when WiFi available. Nothing lost.

---

## Summary: What We Adopt for V2

### In Plain Terms

| # | Solution | What it provides |
|---|----------|-----------------|
| 1 | **Token-Dependent Merge** | Each word gets its own mix of experts, not 25%/25%/25%/25% |
| 2 | **Cross-Track Attention** | Experts exchange notes before answering, not working blind |
| 3 | **Diversity Loss** | Tracks learn different things, not duplicating each other. 4 phones = 4x benefit |
| 4 | **DropTrack** | Model works with any number of tracks (1-32). Phone dropped out — OK |
| 5 | **Variable Groups (0-4)** | Simple task = 0 rounds, complex = 4. Application chooses |
| 6 | **Streaming** | Network and processor work in parallel. -30-40% waiting time with 2+ groups |
| 7 | **Redundancy x2** | Each track on 2 phones (top-1 + top-2 expert). One is stuck — backup is already computing |
| 8 | **Rating** | Everything free for participants. Rating = total tokens trained. Higher rating = higher priority in queue |
| 9 | **Tiered Tracks** | Everyone contributes according to their capacity: 1 track (150MB) or 32 (5GB) |
| 10 | **Own requests = 0** | Keyboard works even without internet and points |
| 11 | **Signal-based learning** | Devices send signals (predicted vs typed). Server fine-tunes the model weekly. Flywheel: more users → more data → smarter model |
| 12 | **Opt-in full data sharing** | User chooses: "Private" (signals = 1x rating), "Help train" (full data = 2x rating). More data = better training = more rating. We honestly explain what we see |

### Later (V3+)

| # | Solution | What it provides |
|---|----------|-----------------|
| 13 | **Confidence Head** | Saves rounds: "already confident after 2, 3rd not needed" |
| 14 | **Federated Learning** | Gradients on device, text never leaves the phone |
| 15 | **Adding Tracks** | New specialist without retraining the entire model |
| 16 | **Expertise marketplace** | Professionals train model on domain data → rating grows → higher priority. Contributing, not "earning" |
| 17 | **Speculative Decoding** | Draft model on phone, network verifies. 3-5x speed |
| 18 | **P2P WiFi** | Phones in the same room communicate directly, without internet |

---

## Training Plan

### Phase 0: Preparation (Done Once, Reused Always)

**Day 1: Tokenizer + Data**

```
1. SentencePiece tokenizer (~2-3 hours)
   - Train on RU + EN (70/30 in tokenizer sample)
   - 32K vocab, optimal for both languages
   - Done ONCE, used for ablation → solid → full → production

2. Data tokenization (~1-2 hours)
   - Quick ablation: 100M tokens, 100% Russian (CulturaX/C4 ru)
   - Retokenize with SentencePiece → binary format
   - Then scale up: 100M → 500M → 5B (append, don't redo)

3. Warmup sweep (~1 hour)
   - 3 runs of 1K steps with different LR warmup (500, 1000, 2000)
   - Pick the best by loss reduction speed
```

**Day 1-2: Architecture Code**

```
Implement in models/:
├── Batched expert dispatch (ffn.py) — 5-10x MoE routing speedup
├── Token-Dependent Merge (layers.py) — Linear(d_model, num_tracks) + softmax
├── Cross-Track Attention (layers.py) — MHA between track outputs
├── Diversity Loss (layers.py) — cosine penalty on track OUTPUTS
├── DropTrack (layers.py) — random drop of tracks during training
├── Expert utilization logging (ffn.py) — bincount + log every N steps
├── Track weight logging (layers.py) — TDM weight distribution + cosine between tracks
└── Variable groups (model.py) — num_groups as parameter, exit after any group
```

**Everything is reused:** tokenizer, data, code — done once, used across all stages.

---

### Phase 1: Quick Ablation (100M tokens, ~1 day of training)

**Goal:** Verify architectural decisions. What works, what doesn't.

**Data:** 100M tokens, 100% Russian.
**Why Russian only:** 100M is too little for two languages. Goal is to test architecture, not language.

**Hardware:** 2x NVIDIA L20 48GB. 2 models in parallel (1 per GPU).

**Models:**

| ID | Groups | Tracks | Merge | CTA | Diversity | DropTrack | What we test |
|----|:------:|:-----:|-------|:---:|:---------:|:---------:|--------------|
| A | 1 | 4 | Scalar (baseline) | - | - | - | Baseline: does 1 group x 4 tracks work? |
| B | 1 | 4 | Token-Dependent | - | - | - | Does TDM help? |
| C | 1 | 4 | TDM | + | - | - | Does CTA on top of TDM help? |
| D | 1 | 4 | TDM | + | + | + | Full package: do tracks specialize? |
| E | 2 | 4 | TDM | + | + | + | Is the second round worth it? |
| F | 1 | 8 | TDM | + | + | + | More tracks = better? |
| G | 1 | 2 | TDM | + | + | + | Degradation with 2 tracks? (DropTrack test) |

**Training parameters:**

```
batch_size = 8
seq_len = 1024
grad_accum = 4
tokens_per_step = 32,768
optimizer = AdamW (β1=0.9, β2=0.95, wd=0.1)
precision = bfloat16
patience = 5-7
LR = from warmup sweep
aux_loss_weight = 0.01
diversity_loss_weight = 0.01-0.1 (tuned on model D)
drop_track_rate = 0.25 (on average 1 out of 4 tracks disabled)
```

**Time calculation:**

```
steps_per_epoch = 100M / 32,768 = 3,052
epochs ≈ 5 (patience=5, estimate)
total_steps = 15,260
t_step ≈ 1.40s (with batched dispatch + TDM + CTA)

T_per_model = 15,260 × 1.40 = 21,364s ≈ 5.9 hours

7 models / 2 in parallel = 4 rounds × 5.9h = 23.6h ≈ 1 day
```

**Metrics (automatic after each experiment):**

```
Quantitative:
├── Val PPL (generation quality)
├── Expert utilization (tokens per expert, dead experts)
├── Track cosine similarity (are tracks duplicating each other)
├── TDM weight entropy (does merge use different weights for different tokens)
└── Throughput (tokens/sec — training speed)

Qualitative (human eye):
├── Generation from fixed prompts (6 total, RU + EN + Code)
├── Comparison table: all models side by side, one prompt
└── Track weights heatmap: which track is active on which token
```

**Test prompts (fixed across all models):**

```
1. "Привет, как"                              — Russian conversational
2. "Уважаемый коллега, хочу сообщить что"     — Russian formal
3. "Столица Франции —"                        — knowledge / facts
4. "def fibonacci("                           — code
5. "Напиши функцию которая сортирует"         — mixed (RU + code)
6. "The weather today is"                     — English (tokenizer test)
```

---

### Phase 2: Solid Ablation (500M tokens, ~3 days) — if Quick shows results

**Data:** 500M tokens, 90% RU + 10% EN.
**Models:** Top-2 from Quick ablation + variants.
**Goal:** Confirm results on larger volume, verify that English doesn't break things.

---

### Phase 3: Full Training V2 (1-5B tokens, 1-3 weeks)

**Data:** 1-5B tokens, 70% RU + 30% EN.
**Model:** Best configuration from Solid ablation.
**Goal:** Production quality for keyboard and API.

---

### Time Summary

| Stage | Data | Languages | Training time | Total with preparation |
|-------|:----:|:---------:|:-------------:|:----------------------:|
| Preparation | — | — | — | 1-2 days (code + tokenizer) |
| Quick ablation | 100M | RU | ~1 day | 3-4 days from start |
| Solid ablation | 500M | 90% RU + 10% EN | ~3 days | +4 days |
| Full training V2 | 1-5B | 70% RU + 30% EN | 1-3 weeks | +1-3 weeks |

**Preparation is done once.** Tokenizer, code, data — reused across all stages.
Data scales up: 100M → 500M → 5B (append, don't redo).

---

## Quick Ablation Results (10M tokens, March 2026)

| Model | Architecture | Final Loss | Notes |
|-------|-------------|:----------:|-------|
| **F_8tracks** | 1 group × 8 tracks + TDM + CTA + Diversity + DropTrack | **8.09** | BEST. More tracks = better. |
| C_cta | 1 group × 4 tracks + TDM + CTA | 8.12 | CTA helps slightly |
| B_tdm | 1 group × 4 tracks + TDM | 8.12 | TDM helps vs baseline |
| A_baseline | 1 group × 4 tracks, scalar merge | 8.14 | Baseline |
| G_2tracks | 1 group × 2 tracks + all | 8.29* | Graceful degradation works |
| D_full | 1 group × 4 tracks + TDM + CTA + Div + Drop | 8.30 | Regularization hurts at 10M, may help at 100M+ |
| E_2groups | 2 groups × 4 tracks + all | 8.56 | WORST. Groups add latency AND hurt quality. |

*G was interrupted at step 120/152, may improve slightly.

### Key Conclusions from Ablation
1. **More tracks = better quality.** F (8 tracks) beats all 4-track models.
2. **1 group is optimal.** E (2 groups) is the worst model. Groups add latency without quality gain.
3. **TDM and CTA help.** B and C beat A baseline consistently.
4. **Diversity + DropTrack hurt at 10M.** D worse than C. Expected — regularization needs more data. Must verify on 100M+.
5. **Architecture confirmed: 1 group × N tracks.** Aligns perfectly with our distributed phone vision.

### Technical Notes
- Models A-D trained with standard AdamW on single GPU (38GB VRAM)
- Models E, F trained with AdamW8bit (bitsandbytes) — required for 2.6B param models on 44GB GPU
- Model G trained on single GPU (699M params, 18GB VRAM)
- 8-bit Adam impact on loss: < 0.01 (negligible for architecture comparison)

---

## Scaling Architecture: Glass Ceiling & Solutions

### Problem: CTA Trained on N Tracks Cannot Extrapolate to N+

CTA (Cross-Track Attention) is trained with a fixed number of tracks (e.g., 8).
DropTrack teaches the model to handle FEWER tracks (1-8), but NOT more (9+).
Softmax distribution fundamentally changes: 8 tracks = 12.5% each, 64 tracks = 1.6% each.

### Solution: Three-Phase Scaling

**Phase A: BTX (8→16 tracks, no latency penalty)**
- Train new tracks independently (copy existing + fine-tune on domain data)
- Retrain ONLY merge layer (TDM + CTA) on new track count
- Merge layer = ~10K params, retraining takes minutes
- CTA extrapolation from 8→16 is small step, likely works
- 0 additional round-trips: same ~115ms latency

**Phase B: BTX or Hierarchical (16→64, test both)**
- BTX path: retrain merge layer on 64 inputs (hours)
  - 1 round-trip, ~115ms, but quality uncertain
- Hierarchical path: 8 groups × 8 tracks, two-level CTA
  - 2 round-trips, ~200ms, quality guaranteed
- **Run both, compare perplexity. Pick winner.**

**Phase C: Hierarchical only (64→512)**
- 8 groups × 8 subgroups × 8 tracks = 512
- 3 round-trips, ~285ms
- CTA always sees exactly 8 inputs (trained configuration)
- No glass ceiling at any scale

### Network Transfer Optimization: INT8
- Compute in bfloat16 on device
- Quantize hidden states to INT8 before sending (divide by max, multiply by 127)
- Coordinator dequantizes back to bfloat16
- Bandwidth halved: 100 tracks × 1MB = 100MB (vs 200MB with bfloat16)
- Quality loss minimal (hidden states, not weights)

---

## V3 Plan: Scaling Beyond Pilot

### V2 → V3 Differences

| Dimension | V2 (Pilot) | V3 (Production) |
|-----------|-----------|-----------------|
| Track size | d=1024, 6 layers, 160MB (INT4) | d=2048, 8 layers, 500MB (INT4) |
| Track count | 8 | 16-64 |
| Total params | 2.5B | 24-96B |
| Active params | 700M | 6-25B |
| Comparable to | GPT-2 | LLaMA-7B to LLaMA-70B |
| Training | 2x L20 (have) | 8-16x A100 80GB (rent ~$5-10K) |
| Phone storage | 160MB per track | 500MB per track |
| New tracks via | Training from scratch | BTX (copy + fine-tune) |
| Merge scaling | Single CTA(8) | BTX or Hierarchical |

### V3 Training Plan
1. Train seed model (8 tracks, d=2048, 8 layers) on 10-50B tokens
2. BTX: copy seed tracks → fine-tune 8-56 more on domain data
3. Test BTX merge vs hierarchical merge at 16 and 64 tracks
4. Pick winner, deploy

### V3 Track Specialization (via BTX)
Each new track is a copy fine-tuned on specific domain:
```
Track 1-8:   General (seed model, trained together)
Track 9:     Code (fine-tuned on code corpora)
Track 10:    Medical (fine-tuned on medical text)
Track 11:    Legal (fine-tuned on legal text)
Track 12:    Math/Science (fine-tuned on STEM)
Track 13-16: Language-specific (EN, ZH, ES, AR)
```

### V3 Training Compute Estimate
```
Seed model (8 tracks, d=2048):
  Params: 8 × 1.5B = 12B
  Data: 50B tokens
  Hardware: 8× A100 80GB, ~1 week
  Cost: ~$3,000-5,000 (cloud rental)

BTX tracks (8 additional):
  Each: 1.5B params, 5B tokens fine-tune
  Hardware: 1× A100 per track, ~1 day each
  Cost: ~$500-1,000 total

Merge retraining:
  Params: ~100K (tiny)
  Time: hours
  Cost: negligible
```

### V4 Vision (Future)
```
Track size: d=4096, 12 layers, 1.5GB (INT4)
Track count: 64-512
Total params: 384B-3T
Active params: 100-800B
Comparable to: GPT-4 / GPT-5 territory
Training: Major compute investment ($100K-1M)
Merge: Hierarchical (3 levels)
Latency: ~285ms (3 round-trips between coordinators)
```

---

## Path to GPT-5: Time x Users, Not Money

### Why We Don't Need $300M

OpenAI spent $300M on GPT-5 because they need to buy everything: data, compute, annotations.
The GPU Network gets all three for free from participants. We are a participant in the network, not a central bank — we contribute compute and benefit from the collective just like everyone else.

### Data: Users ARE the Dataset

```
Users        Tokens/year     Cumulative (4 years)
1M           36B             144B
10M          365B            1.4T
100M         3.6T            14.4T  ← GPT-5 scale
```

This is not scraped web data — it's real human language, fresh every day,
in every language, across every domain. No other company has this.

### RLHF: Every Keystroke is an Annotation

```
Prediction accepted → reward +1 (preferred response)
Prediction rejected → reward -1 (dispreferred), user's input = correct response

This is Direct Preference Optimization (DPO) data:
  OpenAI RLHF:     ~1M preference pairs ($10M, paid annotators)
  Our 1M users:     100M preference pairs PER DAY (free)
  Our 100M users:   10B preference pairs PER DAY

For API: explicit thumbs up/down = classic RLHF at massive scale.
```

### Training: Federated Learning on Phones

Pre-training from scratch requires centralized GPU (seed model).
But continuous fine-tuning runs on phones:

```
Each phone:
  1. Runs inference for user (already doing this)
  2. Collects signals (predicted vs actual)
  3. Computes local gradients for its track (LoRA adapter, small)
  4. Sends gradients to coordinator (NOT data — privacy preserved)

Coordinator:
  1. Aggregates gradients from thousands of phones (Federated Averaging)
  2. Updates global model
  3. Pushes updated weights to phones
  4. Repeat daily

Effective compute:
  1 phone ≈ 1/1000 GPU for training
  10M phones = 10,000 GPU-equivalents
  With federated overhead (10-100×) = 100-1000 free GPUs, 24/7, forever
```

Google already does this with Gboard keyboard. Proven at scale.

### Revised Roadmap

```
Year 0:  Seed model on GPU (our contribution as a network participant)
         V2: 8 tracks, GPT-2 level
         Launch to 150 users (TestFlight)

Year 1:  V3 seed on GPU (network grows)
         64 tracks, LLaMA-13B level
         1M users → 36B tokens/year + 100M DPO pairs/day
         Federated fine-tuning begins

Year 2:  Model continuously improved by participant signals
         LLaMA-70B level (from data, not money)
         10M users → 365B tokens/year
         Professional track specialization emerging

Year 3:  V4 seed on GPU (grant/investor/community fund)
         512 tracks, GPT-4 base level
         + 2 years of federated learning from 10M participants
         + trillions of DPO pairs
         = GPT-4+ level

Year 4:  100M participants
         3.6T new tokens/year
         Federated learning = 100-1000 free GPUs 24/7
         Model improves EVERY DAY
         GPT-5 level — not because anyone spent $300M,
         but because 100M people train the model together
```

### Our Advantage Over OpenAI

```
OpenAI:                              GPU Network:
$300M upfront → GPT-5               Seed model → participants grow it
Model frozen between releases        Model improves EVERY DAY
Data = scraped internet (stale)      Data = live human speech (fresh)
RLHF = paid annotators (expensive)  RLHF = user keystrokes (free, massive)
Inference = expensive servers        Inference = participant phones (free)
Controlled by 1 company             Controlled by no one
```

**Our advantage is not money. Our advantage is time x users.**

---

## Key Papers

| Paper | Year | What it proves | Relevance |
|-------|------|---------------|-----------|
| Branchformer | ICML 2022 | Token-dependent merge works | TDM |
| ParaFormer | 2025 | Inter-branch collaboration > depth | CTA |
| Kraken | NeurIPS 2024 | Parallel branches for multi-device | Our use case |
| BTM / BTX | Meta 2022/2024 | Parallel training + merge = 2.5x cheaper | Scaling |
| DEMix | NAACL 2022 | Domain-specific experts, add/remove post-training | Tiered tracks |
| LayerDrop | ICLR 2020 | Drop layers → extract sub-networks | DropTrack |
| Geometric Reg. | 2025 | Diversity on weights DOES NOT work, on activations YES | Diversity loss |
| Mixture-of-Depths | ICML 2024 | Skip computation per-token | Adaptive depth |
| Soft MoE | ICLR 2024 | Soft routing > hard routing | Merge design |
