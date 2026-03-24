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

| Request Type | Cost per request | Notes |
|-------------|-----------------|-------|
| Keyboard    | 0               | Always free (local compute) |
| Translation | 0               | Always free (improves multilingual model) |
| Any free-tier app | 0        | Apps that generate training data = free |
| API Light   | 10 miles        | 2 groups |
| API Full    | 30 miles        | 3 groups |
| API Deep    | 50 miles        | 4 groups |

Free-tier principle: anything that improves the model (keyboard, translation, other first-party apps) costs 0 miles. Only API requests that consume network compute from other devices cost miles.

### 5.4 Network Multiplier

The network multiplier adjusts based on total participating devices to incentivize early adoption and account for network growth phases.

| Active Devices | Multiplier | Effective Rate |
|---------------|-----------|---------------|
| 0 — 100       | 100.0     | 100 miles/token/layer |
| 101 — 1,000   | 50.0      | 50 miles/token/layer |
| 1,001 — 10,000 | 30.0     | 30 miles/token/layer |
| 10,001 — 100,000 | 18.0   | 18 miles/token/layer |
| 100,001 — 1,000,000 | 10.0 | 10 miles/token/layer |
| 1,000,001 — 10,000,000 | 6.0 | 6 miles/token/layer |
| 10,000,001 — 100,000,000 | 4.0 | 4 miles/token/layer |
| 100,000,001 — 1,000,000,000 | 2.5 | 2.5 miles/token/layer |
| 1,000,000,001+ | 1.5      | 1.5 miles/token/layer |

Multiplier transitions are determined by the 30-day rolling average of unique active devices. Transition is irreversible — the multiplier never increases. Minimum multiplier is 1.5 — there is always a bonus for participation.

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

### 5.8 Welcome Gifts

New devices receive miles as a gift upon joining. No debt, no credit — pure gift.

| Action | Miles Gifted |
|--------|-------------|
| Install app + create device identity | 500 |
| Enable Network mode | 500 |
| First 24 hours online | 200 |
| Invite a friend (both receive) | 300 |
| Install additional first-party app | 500 |

Total potential from day 1: 1,000+ miles = 100+ API Light requests.

Welcome gifts are one-time per device. Device identity is tied to hardware ID (IDFV on iOS, Android ID on Android). Factory reset = new device ID, but behavioral fingerprinting detects anomalies.

### 5.9 Surge Pricing

Request costs and earning rates adjust dynamically based on network load. This balances supply and demand in real-time.

```
surge_ratio = active_requests / available_compute_slots
```

| Surge Ratio | Demand Multiplier | Earn Multiplier | Indicator |
|------------|------------------|----------------|-----------|
| < 0.3      | 0.5x             | 0.7x           | Green: cheap API |
| 0.3 — 0.7  | 1.0x             | 1.0x           | Normal |
| 0.7 — 0.9  | 1.5x             | 1.5x           | Yellow: busy |
| 0.9 — 0.95 | 2.0x             | 3.0x           | Orange: high load |
| > 0.95     | 3.0x             | 5.0x           | Red: max load |

Surge is calculated per-region (latency-based clusters, not geographic). Updated every 60 seconds. Clients see current surge level before submitting requests.

Surge creates a self-balancing system: high demand → higher earn rate → device owners enable compute → supply increases → surge drops.

### 5.10 Positive-Sum Economics

Compute Miles are fundamentally different from money:

- **Created by work**: miles appear when a device computes a real request. No mining into void.
- **Destroyed by usage**: miles are spent (burned) when using API. They don't transfer to someone else's pocket.
- **No scarcity by design**: more devices = more compute = more miles for everyone. One person earning more does NOT mean another earns less.
- **Cannot be created without demand**: if no one sends requests, no miles are generated. Prevents inflation.
- **No fiat exchange** (V2): miles cannot be officially converted to money. They are utility tokens for compute access only.

## 6. Anti-Abuse Rules

### 6.1 Proof of Useful Work

Miles are generated ONLY when a device computes a real inference request from a real client. There is no way to mine miles without actual demand from the network. Fake demand (sending requests to yourself) costs miles to send, making it a net loss.

### 6.2 Behavioral Fingerprinting

The coordinator monitors device behavior patterns to detect emulators and bot farms:

- Battery level patterns (real devices fluctuate; emulators show constant 100%)
- Network characteristics (real devices change WiFi/cellular; emulators stay constant)
- Gyroscope/accelerometer noise (real devices have micro-movements; emulators are perfectly still)
- Response time variance (real devices vary with thermal throttling; emulators are consistent)

Devices with suspicious patterns receive a reputation penalty. This is automatic — no human decision involved.

### 6.3 Reputation System

Each device has a reputation score (0.0 — 1.0) that affects request routing:

```
New device:     0.5 (neutral)
Good behavior:  +0.01 per day of valid compute (max 1.0)
Bad behavior:   -0.1 per invalid result or suspicious pattern
Inactive:       -0.01 per day offline (min 0.1)
```

Low-reputation devices receive fewer requests → earn fewer miles → natural consequence without explicit "banning."

### 6.4 Append-Only Rules

Anti-abuse rules are versioned and append-only. New rules EXTEND existing rules, never retroactively change them:

```
Rule v1.0:  Miles earned for valid compute
Rule v1.1:  + Behavioral fingerprinting for emulator detection
Rule v1.2:  + Reputation decay for inactive devices
```

Old miles are never invalidated. Old devices are never retroactively punished. Rules are published in the open-source repository before activation.

## 7. Governance

### 7.1 Rule Updates

- **Bootstrap phase (V2)**: Core maintainers propose rule changes. Users accept by updating their app. No update = old rules still work.
- **Mature phase (V4+)**: Rule proposals submitted to public repository. 2-week discussion period. Community vote: 1 device = 1 vote (not 1 mile = 1 vote, to prevent plutocracy). >67% approval required.

### 7.2 Automatic vs. Voted Rules

| Automatic (no vote needed) | Voted (community approval) |
|---------------------------|--------------------------|
| Surge pricing formula (pure math) | New anti-abuse detection methods |
| Reputation decay rate | Multiplier table changes |
| Welcome gift amounts | New data collection types |
| Tensor validation checks | Transfer endpoint activation (V3) |
| | Fiat exchange policy (V4+) |

### 7.3 Coordinator Independence

The coordinator is open-source software. Anyone can run their own coordinator. Devices can connect to any coordinator they trust. If the primary coordinator behaves unfairly:

1. Community forks the coordinator code
2. Launches alternative coordinator
3. Devices switch (change coordinator URL in settings)
4. Primary coordinator loses devices → loses relevance

This is the ultimate check on power: the coordinator has no lock-in. It must earn trust continuously.

## 8. Mile Transfer Protocol

**Availability: V3+** — Transfer functionality is disabled in V2. The protocol is specified here for completeness. Activation requires community vote (see Section 7.2).

Miles can be transferred between devices using signed messages.

### 8.1 Transfer Request

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

### 8.2 Validation

- Signature must be valid for the `from_device` public key
- `amount` must not exceed sender's balance
- `nonce` must be strictly greater than sender's last used nonce
- `timestamp` must be within 5 minutes of coordinator time

### 8.3 Properties

- Transfers are irreversible
- No fee is applied by the protocol
- Minimum transfer: 1 mile
- Transfer history is queryable by device (own transfers only)

## 9. Security

### 9.1 Device Identity

Each device generates an Ed25519 keypair on first launch. The public key serves as the device identifier. The private key never leaves the device.

### 9.2 Track Result Validation

The coordinator validates track results using:
- Tensor shape verification
- Finite value check (no NaN/Inf)
- Response time monitoring (outlier detection)
- Periodic blind comparison (same input to multiple nodes, cross-check outputs)

Nodes that consistently produce invalid or outlier results have their reputation score reduced and eventually stop receiving requests.

### 9.3 Privacy

- Device identifiers are pseudonymous (public keys, not linked to personal identity)
- Routing uses latency measurements, not geographic coordinates
- Aggregated network statistics are published at country level only
- No individual device location, specialization, or activity is ever exposed

## 10. Versioning

This protocol follows semantic versioning. Breaking changes increment the major version. Track nodes and coordinators negotiate protocol version during WebSocket handshake.

---

*Specification maintained by the Miracle Network contributors. Submit issues and proposals via GitHub.*
