# GPU Network Protocol Specification

Version 0.2.0 — Draft

---

## The Rules

Everything is free. The model belongs to everyone. The network is alive — you can't download it.

Your device = your AI. Instant. No queue.

Device in the network → trains the model → rating grows.

Want more powerful AI → rating decides.

Rating = your passport. Works in any app built on GPU Network.

**Build with free AI.** If you're a developer — build anything: a chatbot, a translator, a code assistant, a medical app, a game with AI characters. You don't pay for AI. Not a cent. Not ever. Your users already have devices in the GPU Network. When they open your app, their rating gives them priority, their devices provide compute. You write the UI. The network does the thinking. No API keys. No billing. No rate limits. Just build.

API without a device = last in line. We do NOT train on API data.

Code is open. Rules change by vote.

```
╔══════════════════════════════════════════════════════╗
║                                                      ║
║  Your device = your AI. Instant.                     ║
║  Device online → trains model → rating grows.        ║
║                                                      ║
║  Want more powerful AI → rating decides:             ║
║    signals   → 1x rating per token                   ║
║    full data → 2x rating per token                   ║
║  Queue = √rating / Σ√everyone.                       ║
║                                                      ║
║  Rating = permanent contribution. Transferable.      ║
║  Rating = your passport across all apps.             ║
║                                                      ║
║  Developers: build anything with free AI.             ║
║  No API keys. No billing. No rate limits.            ║
║  Your users bring compute. You build the interface.  ║
║  The network does the thinking.                      ║
║                                                      ║
║  API without device = last in queue.                 ║
║  We do NOT train on API data.                        ║
║                                                      ║
║  Code is open. Rules change by vote.                 ║
║                                                      ║
╚══════════════════════════════════════════════════════╝
```

---

## 1. Overview

The GPU Network is a distributed AI system where participating devices collectively train and run language models. Each device hosts one or more model tracks (parallel branches of the model), trains the shared model with idle compute, and serves inference requests based on a contribution-based priority system.

All compute, all AI, all services are free. The model belongs to everyone. The network is alive — you can't download it, because its power comes from thousands of specialized tracks running on thousands of devices simultaneously.

## 2. Architecture

### 2.1 Roles

- **Coordinator**: Manages request routing, track assignment, training task distribution, and result merging. Open source. Anyone can run one. Devices choose which coordinator to trust.
- **Track Node**: Any device running one or more model tracks. Computes inference and training tasks. Earns rating.
- **Client**: Any application making inference requests. Developers build the app, but the USER's rating determines priority — not the developer's wallet. Building on GPU Network costs $0. Your users bring their own compute and rating. You bring the idea and the interface.

### 2.2 Request Flow

```
Client → Coordinator: InferenceRequest(tokens, user_rating)
Coordinator: assigns compute proportional to √(user_rating) / Σ√(all_in_queue)
Coordinator → Track Nodes: TrackCompute(input_tensor, track_id)
Track Nodes → Coordinator: TrackResult(output_tensor, track_id, node_id)
Coordinator: CrossTrackAttention + TokenDependentMerge
Coordinator → Client: InferenceResponse(tokens)
```

### 2.3 Priority System

Three levels, in order:

1. **Self**: your device computes for you. Instant. No queue. No rating needed.
2. **Network users (with devices)**: served by contribution rating. Queue = √rating / Σ√all.
3. **API (without device)**: last in line. Served when network has spare capacity.

### 2.4 Redundancy

For each track, the coordinator dispatches to 2N nodes (N required). First N valid responses are used. Fault tolerance without additional latency.

## 3. Track Specification

### 3.1 Track Format

Tracks are distributed as ONNX or SafeTensors bundles:

```json
{
  "track_version": "0.2.0",
  "model_family": "gpu-v2",
  "d_model": 1024,
  "layers": 6,
  "experts": 8,
  "top_k": 2,
  "quantization": "int4",
  "size_bytes": 157286400,
  "checksum_sha256": "..."
}
```

### 3.2 Track Tiers

| Tier | Tracks | Storage | RAM | Example Devices |
|------|--------|---------|-----|----------------|
| 1 | 1 | ~160 MB | 512 MB | Budget phones |
| 2 | 2 | ~320 MB | 1 GB | Mid-range phones |
| 3 | 4 | ~640 MB | 2 GB | Flagship phones |
| 4 | 8+ | 1.3+ GB | 4+ GB | Desktop, workstation |

More tracks = more compute contributed = rating grows faster = higher priority. Built-in incentive, no artificial bonus needed.

## 4. Communication Protocol

### 4.1 Transport

WebSocket over TLS 1.3. Binary frames:

```
Bytes 0-1:   Message type (uint16)
Bytes 2-5:   Payload length (uint32)
Bytes 6-9:   Sequence number (uint32)
Bytes 10-13: Track ID (uint32)
Bytes 14+:   Payload
```

### 4.2 Tensor Encoding

Tensors transmitted in INT8 for network efficiency (quantized from bfloat16 on device). Coordinator dequantizes for merge operations. Quality loss < 1%.

Delta encoding for sequential tokens:
```
Frame N:   full tensor (int8)
Frame N+1: delta from frame N (int8, scaled)
```

### 4.3 Streaming

Track nodes stream intermediate results during computation. Partial results transmitted after every ceil(L/2) layers. Overlaps computation with network transfer, reducing latency by 20-40% for multi-group configurations.

## 5. Rating System

### 5.1 Definition

**Rating** = total tokens processed for model training over the device's lifetime.

Rating is NOT a currency. It is a permanent record of contribution. It only goes up (when training) and can be transferred (to family, friends, new accounts).

### 5.2 Earning Rating

All idle compute → model training (eval tasks, gradient computation). Rating earned:

```
rating_earned = tokens_processed × data_contribution_multiplier
```

| Mode | Data Shared | Multiplier |
|------|------------|------------|
| Private (default) | Signals only (predicted vs actual token ID) | 1x |
| Training (opt-in) | Signals + full context tokens (anonymized) | 2x |

Self-compute (your own inference requests) does NOT earn rating. Only training the shared model earns rating.

### 5.3 Priority Queue

When multiple users request compute simultaneously:

```
your_share = √(your_rating) / Σ√(all_ratings_in_queue)
```

Square root ensures:
- Higher rating = higher priority (fair)
- But not linearly (prevents whales from starving everyone)
- Rating 10,000x higher = only ~100x more share

### 5.4 Rating Properties

- **Permanent**: rating never decays, never expires. Your contribution stands forever.
- **Transferable**: you can transfer any amount of your rating to another account.
  - Parent → child (inheritance)
  - Old account → new account (device change)
  - Gift to anyone
  - Transfer reduces sender's rating, increases receiver's. Sum unchanged.
- **Cross-app**: rating is tied to your account, not to any specific app. Open any app built on GPU Network — your rating follows you.
- **Not a currency**: rating cannot be "spent." Using AI does not reduce your rating. Rating determines priority, not balance.

### 5.5 API Access (No Device)

API clients without devices in the network:
- Served at lowest priority (after all device-owning users)
- Must provide full interaction data (anonymized) — this is the cost of entry
- We do NOT train on API data (stated openly, verified by open-source code)
- API data is processed and discarded, never stored
- Rating: 0 (no device = no contribution = no rating)

### 5.6 Why The Network Cannot Be Replaced

The model is open source. Anyone can download it. But:

```
1 device running 1 track:          basic AI (like GPT-2)
8 devices running 8 tracks:        good AI (like LLaMA-7B)
1,000 devices with specialized tracks: powerful AI (like LLaMA-70B)
```

The model's power comes from the NETWORK of diverse, specialized tracks running simultaneously. A single download gives you one track. The network gives you thousands. You can fork the code, but you can't fork the living network.

## 6. Idle Compute → Model Training

### 6.1 Training Priority

When a device has no pending inference requests, the coordinator assigns training tasks:

```
Priority 1: Self-inference (user's own requests) — instant, free
Priority 2: Network inference (other users' requests) — by rating
Priority 3: API inference (no-device clients) — last
Priority 4: Model training (eval tasks) — fills remaining capacity
Priority 5: Model training (gradient computation, V3+) — background
```

Target: 100% device utilization at all times. No idle compute wasted.

### 6.2 Training Task Types

| Type | Description | Rating Earned | Version |
|------|------------|--------------|---------|
| Eval task | Forward pass on test data, report accuracy | 1x per token | V2 |
| LoRA fine-tune | Train small adapter on-device | 1x per token | V3 |
| Gradient computation | Full backward pass, send gradients | 1x per token | V4 |

All types earn the same rating per token — compute is compute, regardless of task type.

## 7. Anti-Abuse

### 7.1 Proof of Useful Work

Rating is earned ONLY by processing real training tasks assigned by the coordinator. No self-mining. The coordinator validates results against known answers (for eval tasks) or cross-checks with other devices.

### 7.2 Behavioral Fingerprinting

Detects emulators and bot farms:
- Battery patterns (real devices fluctuate)
- Network changes (real devices switch WiFi/cellular)
- Sensor noise (real devices have micro-movements)
- Response time variance (real devices have thermal throttling)

Suspicious devices receive fewer training tasks → earn less rating → natural consequence.

### 7.3 API Data Poisoning Prevention

We do NOT train on API data. This eliminates the entire attack surface of data poisoning through API. API data is processed for inference and immediately discarded.

Training data comes ONLY from opted-in device users whose devices are in the network and have rating history. Trusted sources only.

### 7.4 Append-Only Rules

Rules are versioned and extend, never retroactively change:

```
Rule v1.0:  Rating earned for valid training compute
Rule v1.1:  + Behavioral fingerprinting
Rule v1.2:  + Cross-device result validation
```

Old ratings are never invalidated. Old devices are never retroactively punished.

## 8. Governance

### 8.1 Rule Updates

- **Bootstrap (V2)**: Maintainers propose changes. Users accept by updating app.
- **Mature (V4+)**: Community vote. 1 device = 1 vote. >67% approval required.

### 8.2 Automatic vs. Voted

| Automatic | Requires Vote |
|-----------|--------------|
| Surge/priority formula | New anti-abuse methods |
| Training task assignment | Rating transfer rules |
| Tensor validation | New data collection types |
| | Protocol breaking changes |

### 8.3 Coordinator Independence

Open-source coordinator. Anyone can run one. Devices choose their coordinator. If the primary coordinator is unfair → community forks → devices switch. Ultimate check on power.

## 9. Rating Transfer Protocol

Rating can be transferred between accounts at any time.

```
POST /v1/rating/transfer
{
  "from_account": "ed25519_public_key_hex",
  "to_account": "ed25519_public_key_hex",
  "amount": 50000,
  "nonce": 1234567890,
  "timestamp": "2026-03-26T12:00:00Z",
  "signature": "ed25519_signature_hex"
}
```

- Transfers are irreversible
- No fee
- Sender's rating decreases, receiver's increases
- Sum unchanged — no rating created or destroyed

## 10. Security

### 10.1 Device Identity

Ed25519 keypair generated on first launch. Public key = device identifier. Private key never leaves device. Account tied to email/phone for recovery across devices.

### 10.2 Track Result Validation

- Tensor shape verification
- Finite value check (no NaN/Inf)
- Cross-device comparison (same input to multiple nodes)
- Response time outlier detection

### 10.3 Privacy

- Device identifiers are pseudonymous (public keys)
- Routing uses latency, not geography
- No individual device location or activity exposed
- API data processed and discarded, never stored
- Open source — anyone can verify

---

*GPU Network. Free AI for everyone. The network is alive — you can't download it.*

*Specification maintained by GPU Network contributors. Contact: kustyuka@gmail.com | Telegram: @yuka_k*
