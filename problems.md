# PT-MoE Research — Problems & Bottlenecks

## 1. ARCHITECTURE: DISTRIBUTED MODEL ON PHONES

### Root problem: PT-MoE lost to dense in the experiment

The entire architecture is built on the hypothesis "parallel tracks = distributable without quality loss". The experiment showed the opposite: the more tracks, the worse the PPL. This must be fixed **before** V2, otherwise the foundation is shaky.

**Why PT-MoE degrades (likely causes):**

**A. Merge Layer does not learn anything useful.** 4-12 scalar weights (softmax) = static weighted average. The model cannot say "for this token track 2 matters, and for that one — track 4". All tracks learn the same thing → duplication instead of specialization.

- **Fix:** Token-dependent merge: `gate = softmax(Linear(d_model, num_tracks))` — each token gets its own weight vector. This is +768×num_tracks parameters (negligible), but gives the track a reason to specialize. Going further: cross-track attention at the merge boundary (expensive, but maximally powerful).

**B. No track specialization mechanism.** Aux loss balances experts **within** a MoE layer, but nothing forces **tracks** to be different. Track 1 and track 4 see the same input, receive the same gradients through merge → converge to the same solution.

- **Fix:** Track diversity loss — penalty for cosine similarity between track outputs. Or: dropout of entire tracks during training (like DropPath in vision) — forces each track to be self-sufficient. Or: different data/tasks for different tracks (one for RU, another for EN, a third for code).

**C. Shallow-wide loses to deep-narrow.** 2 groups × 12 tracks = 2 sequential transformation steps. A transformer with 2 layers is fundamentally weaker than a 24-layer one — no amount of parallel tracks compensates for this. Depth = abstraction and reasoning.

- **Fix:** Do not go below 6 groups. The optimal point is likely 6-8 groups × 2-4 tracks. This is still distributable (4 phones), but preserves sufficient depth. Configurations 2×12 and 3×8 are dead ends.

---

### Coordinator problem

GPU_PLAN Phase 2-4 assumes the coordinator is "lightweight". In reality:

| Operation | Size at V2 (d=1024) | Size at V4 (d=4096) |
|----------|----------------------|----------------------|
| Embedding | 50K × 1024 = 50M params | 50K × 4096 = 200M params |
| LM Head | 50K × 1024 = 50M params | 50K × 4096 = 200M params |
| Merge (per group) | negligible | negligible |
| **Total on coordinator** | **~100M params** | **~400M params** |

The coordinator performs **a third of all computations** and runs on numpy/CPU. This is the throughput ceiling.

- **Fix (V2):** Move the coordinator to PyTorch + GPU (at least server-side). Or: distribute embedding/LM head too — one phone computes embedding, another the LM head.
- **Fix (V4):** If the coordinator = user's device, embedding + LM head must be quantized and run locally on the user's phone. That's ~100-200MB INT4 — fits.

---

### Network protocol

**Float32 over the network — 2× overhead.** Tensor `1 × seq_len × d_model` on each round-trip:
- V2 (d=1024): 1 × 512 × 1024 × 4 = **2MB** per call, float16 = 1MB
- V4 (d=4096): 1 × 512 × 4096 × 4 = **8MB** per call, float16 = 4MB

With 6 groups × 2 directions = 12 transfers per token. V4: **96MB per token** at float32.

- **Fix:** Float16 minimum. Better: INT8 quantization of hidden states. Even better: delta encoding.

**No validation of responses from phones.** NaN/wrong shape = silent garbage generation.

- **Fix:** `assert tensor.shape == expected`, `assert isfinite(tensor).all()`, timeout + fallback.

**No redundancy.** The slowest phone = the group's speed.

- **Fix:** Send the task to N+1 phones, take the first N responses.

---

### What will actually improve distributability

- **Speculative decoding** — a small model on the user's phone generates 4-8 draft tokens, the network verifies in 1 forward pass. ×3-5 speed improvement.
- **Pipeline parallelism** — group 1 processes token N, group 2 simultaneously processes token N-1. Throughput ×num_groups.
- **P2P WiFi** — MultipeerConnectivity iOS, latency <1ms. 5 phones in a room = full model locally.

---

## 2. TRAINING: WHAT WE ARE DOING WRONG

### Batched expert dispatch — bottleneck #1

`models/ffn.py` — Python for-loop over each token × each expert. The main reason MoE is 1.7× slower than dense (5368s vs 3120s).

```python
# CURRENT: O(batch × seq_len × top_k) Python calls
for i in range(batch * seq_len):
    for j, expert_idx in enumerate(top_indices[i]):
        expert_out = self.experts[expert_idx](token)

# NEEDED: O(num_experts) batched calls
for expert_idx in range(num_experts):
    mask = (top_indices == expert_idx).any(dim=-1)
    tokens_for_expert = x[mask]
    out[mask] = expert(tokens_for_expert)
```

Difference: **5-10×** on real sizes.

### 10M tokens for MoE = insufficient

MoE requires significantly more data than dense. 10M = 500-2000× less than needed. The negative result may be entirely due to insufficient data.

- **Fix:** Minimum 100M for ablation, 1B+ for real conclusions.

### Expert utilization — flying blind

No logging of: how many tokens each expert receives, dead experts, router collapse.

- **Fix:** 5 lines of code: `expert_counts = torch.bincount(top_indices.flatten(), minlength=num_experts)`

### Early stopping patience=3

MoE PPL is unstable. Patience=3 can kill training during a temporary spike.

- **Fix:** Patience=5-7 or EMA smoothing on val_ppl.

### LR warmup is not tuned

Fixed `min(2000, max_steps // 10)`.

- **Fix:** 2-3 short runs with different warmup.

### No ablation study

One experiment, one dataset, one size. Impossible to understand what is hurting.

- **Fix:** Minimal ablation:
  1. token-dependent merge vs scalar merge
  2. track diversity loss vs without
  3. batched dispatch vs loop
  4. 10M vs 100M vs 500M tokens

---

## 3. EVERYTHING ELSE

### Security and infrastructure

| Problem | Impact | Fix |
|----------|--------|------|
| Credentials in public repo (IP, SSH paths) | Anyone can find the servers | `.gitignore` + rotate + remove from git history |
| WebSocket without auth | Garbage tensors, DoS | API key in header at handshake |
| No rate limiting | One client clogs the server | `slowapi` or nginx rate limit |
| requirements.txt without pinning | Breaks on updates | `pip freeze > requirements.lock` |
| No tests, no CI | Refactoring = roulette | Smoke test for the model + contract test for WebSocket |

### iOS keyboard

| Problem | Impact | Fix |
|----------|--------|------|
| KeyboardViewController 1800 lines | Impossible to maintain | Split into: InputHandler, LayoutManager, PredictionManager |
| No client-side debounce | 5-10 WS requests/sec | `DispatchWorkItem` with 150ms delay |
| SignalCollector without atomic write | Extension gets killed → corrupt | Temp file → atomic rename or SQLite |
| AutocorrectEngine O(n) fuzzy | Lag on 100K+ dictionary | SymSpell — O(1) lookup |
| RealPredictor greedy only | Repetitive predictions | Connect TOP_K/TEMPERATURE from config.py |

### Privacy

SignalCollector sends `context_ids` — this is encoded text. The server can reconstruct everything.

- **V2:** Encrypt context_ids, server receives only `(predicted_id, actual_id, match: bool)`.
- **V3+:** Gradients on device, context is not sent.

---

## MAPPING: WHAT IS IN THE PLAN, WHAT IS NOT

### Planned to solve (present in GPU_PLAN / GPU_EVOLUTION)

| Problem | Where in the plan |
|----------|-------------|
| GPT-2 tokenizer | V2 → SentencePiece 32K |
| ONNX without quantization | INT4 Quantization (Stage 2) |
| KV-cache | KV-Cache Sharing (Stage 2), but only sharing, not basic |
| ComputeWorker = stub | Phase 2: ONNX Runtime Web |
| Signals not used | Phase 3: federated learning |
| No offline predictions | V3: on-device inference |
| Network latency | Pipeline Parallelism + P2P + Speculative |
| Synchronous barrier | Pipeline Parallelism (Stage 2) |

### NOT planned to solve (missed)

| Problem | Criticality |
|----------|-------------|
| Batched expert dispatch | HIGH — 5-10× training speedup |
| Token-dependent merge | HIGH — root cause of PT-MoE degradation |
| Negative results not analyzed | HIGH — foundational |
| Expert utilization logging | HIGH — 5 lines of code, diagnostics |
| Coordinator on numpy = bottleneck | MEDIUM — throughput ceiling |
| Track diversity / specialization | HIGH — tracks learn the same thing |
| WebSocket without auth | MEDIUM — security |
| Binary protocol without validation | MEDIUM — silent errors |
| Float32 over the network | MEDIUM — 2× data overhead |
| KeyboardViewController God Object | MEDIUM — maintainability |
| SignalCollector without atomic write | MEDIUM — data loss |
| RealPredictor greedy only | LOW — UX |
| No tests / CI | MEDIUM — quality |
| Credentials in public repo | HIGH — security |
| Privacy: context_ids = text | HIGH — reputation risk |
