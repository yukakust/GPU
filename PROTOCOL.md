# Miracle Network Protocol Specification

Version 0.1.0 — Draft

## 1. Overview

The Miracle Network is a distributed inference system where participating devices contribute compute resources to collectively run language models. Each device hosts one or more model tracks (parallel branches of the model) and processes inference requests from the network coordinator.

This document specifies the communication protocol, resource accounting, and coordination mechanisms.

## 2. Architecture

### 2.1 Roles

- **Coordinator**: Manages request routing, track assignment, and result merging. Runs on infrastructure operated by network maintainers during bootstrap phase; transitions to elected coordinators as the network matures.
- **Track Node**: Any device running one or more model tracks. Receives input tensors, computes track output, returns results. Track nodes are untrusted by default.
- **Client**: Any application making inference requests to the network. Clients spend Compute Miles for requests.

### 2.2 Request Flow

```
Client → Coordinator: InferenceRequest(tokens, groups, priority)
Coordinator → Track Nodes: TrackCompute(input_tensor, track_id, group_id)
Track Nodes → Coordinator: TrackResult(output_tensor, track_id, node_id)
Coordinator: CrossTrackAttention + TokenDependentMerge
Coordinator → Client: InferenceResponse(tokens, miles_spent)
```

### 2.3 Redundancy

For each track in a request, the coordinator dispatches to `2N` nodes (where N is the required number of tracks). The first N valid responses are used; remaining are discarded. This provides fault tolerance without additional latency.

## 3. Track Specification

### 3.1 Track Format

Tracks are distributed as ONNX or SafeTensors bundles with the following metadata:

```json
{
  "track_version": "0.1.0",
  "model_family": "miracle-v2",
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

Devices host tracks based on available resources:

| Tier | Tracks | Storage | RAM Required | Example Devices |
|------|--------|---------|-------------|----------------|
| 1 | 1 | ~160 MB | 512 MB | Budget phones, older devices |
| 2 | 2 | ~320 MB | 1 GB | Mid-range phones |
| 3 | 4 | ~640 MB | 2 GB | Flagship phones, tablets |
| 4 | 8+ | 1.3+ GB | 4+ GB | Desktop, workstation |

## 4. Communication Protocol

### 4.1 Transport

WebSocket over TLS 1.3. Binary frames with the following header:

```
Bytes 0-1:   Message type (uint16)
Bytes 2-5:   Payload length (uint32)
Bytes 6-9:   Sequence number (uint32)
Bytes 10-13: Track ID (uint32)
Bytes 14+:   Payload (tensor data or control message)
```

### 4.2 Tensor Encoding

Tensors are transmitted in bfloat16 format. Shape is inferred from track configuration and sequence length (transmitted in the request header).

For bandwidth optimization, delta encoding may be used for sequential tokens:
```
Frame N:   full tensor (bfloat16)
Frame N+1: delta from frame N (int8, scaled)
```

### 4.3 Streaming

Track nodes stream intermediate results during computation. When a track has L layers, partial results are transmitted after every ceil(L/2) layers. This overlaps computation with network transfer, reducing effective latency by 20-40% for multi-group configurations.

## 5. Compute Miles

### 5.1 Definition

A **Compute Mile** is the base unit of resource accounting in the Miracle Network.

```
1 Compute Mile = 1 token processed × 1 layer computed on a track node
```

### 5.2 Earning

Track nodes earn miles for processing inference requests from other devices. Self-originated requests (where the requesting client and track node are the same device) do not generate miles.

Miles earned per request:

```
miles = tokens × layers_per_track × tracks_hosted × network_multiplier
```

### 5.3 Spending

Clients spend miles to submit inference requests:

```
miles_cost = tokens × layers_per_track × tracks_needed × groups
```

| Request Type | Groups | Tracks | Cost (100 tokens, 6 layers) |
|-------------|--------|--------|---------------------------|
| Keyboard    | 0-1    | 1-2    | 0 (local) — 1,200        |
| Chat        | 1-2    | 4      | 2,400 — 4,800            |
| API Light   | 2      | 4      | 4,800                     |
| API Full    | 3      | 4      | 7,200                     |
| API Deep    | 4      | 4      | 9,600                     |

### 5.4 Network Multiplier

The network multiplier adjusts based on total participating devices to incentivize early adoption and account for network growth phases.

| Active Devices | Multiplier | Effective Rate |
|---------------|-----------|---------------|
| 0 — 100       | 50.0      | 50 miles/token/layer |
| 101 — 1,000   | 24.0      | 24 miles/token/layer |
| 1,001 — 10,000 | 12.0     | 12 miles/token/layer |
| 10,001 — 100,000 | 6.0    | 6 miles/token/layer |
| 100,001 — 1,000,000 | 3.0 | 3 miles/token/layer |
| 1,000,001+    | 1.0       | 1 mile/token/layer |

Multiplier transitions are determined by the 30-day rolling average of unique active devices. Transition is irreversible — the multiplier never increases.

### 5.5 Device Capability Bonus

Devices hosting more tracks receive a logarithmic bonus:

```
capability_bonus = 1 + log2(tracks_hosted)

1 track:  1.0x
2 tracks: 2.0x
4 tracks: 3.0x
8 tracks: 4.0x
```

Total miles earned:

```
miles = tokens × layers × tracks × network_multiplier × capability_bonus
```

### 5.6 Data Contribution Bonus

Devices that opt in to training data sharing receive additional miles:

| Mode | Data Shared | Bonus |
|------|------------|-------|
| Private (default) | Signals only (predicted vs actual token ID) | 1.0x |
| Training | Signals + full context (token IDs) | 1.5x |
| Full (future) | Signals + context + local gradients | 2.0x |

### 5.7 Mile Ledger

Each device maintains a local mile balance. Balances are synchronized with the coordinator periodically (every 5 minutes or on significant change). The coordinator maintains the authoritative ledger.

Mile balances are unsigned 64-bit integers. Maximum balance: 2^64 - 1 (sufficient for all practical purposes).

## 6. Mile Transfer Protocol

Miles can be transferred between devices using signed messages.

### 6.1 Transfer Request

```
POST /v1/miles/transfer
Content-Type: application/json

{
  "from_device": "ed25519_public_key_hex",
  "to_device": "ed25519_public_key_hex",
  "amount": 50000,
  "nonce": 1234567890,
  "timestamp": "2026-03-24T12:00:00Z",
  "signature": "ed25519_signature_hex"
}
```

### 6.2 Validation

- Signature must be valid for the `from_device` public key
- `amount` must not exceed sender's balance
- `nonce` must be strictly greater than sender's last used nonce
- `timestamp` must be within 5 minutes of coordinator time

### 6.3 Properties

- Transfers are irreversible
- No fee is applied by the protocol
- Minimum transfer: 1 mile
- Transfer history is queryable by device (own transfers only)

## 7. Security

### 7.1 Device Identity

Each device generates an Ed25519 keypair on first launch. The public key serves as the device identifier. The private key never leaves the device.

### 7.2 Track Result Validation

The coordinator validates track results using:
- Tensor shape verification
- Finite value check (no NaN/Inf)
- Response time monitoring (outlier detection)
- Periodic blind comparison (same input to multiple nodes, cross-check outputs)

Nodes that consistently produce invalid or outlier results have their reputation score reduced and eventually stop receiving requests.

### 7.3 Privacy

- Device identifiers are pseudonymous (public keys, not linked to personal identity)
- Routing uses latency measurements, not geographic coordinates
- Aggregated network statistics are published at country level only
- No individual device location, specialization, or activity is ever exposed

## 8. Versioning

This protocol follows semantic versioning. Breaking changes increment the major version. Track nodes and coordinators negotiate protocol version during WebSocket handshake.

---

*Specification maintained by the Miracle Network contributors. Submit issues and proposals via GitHub.*
