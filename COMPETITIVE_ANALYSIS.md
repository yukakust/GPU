# Competitive Analysis: Distributed AI Inference Landscape (2024-2026)

*Last updated: March 2026*

## Executive Summary

Our approach — splitting a language model into **parallel tracks** distributed across **consumer phones** with a free compute-for-access economy — occupies an **unclaimed intersection** in the market. Apple validated the parallel-track architecture (Feb 2026) for datacenters. No one has applied it to phone swarms over mobile networks.

---

## Market Map

```
                    PARALLEL TRACKS/BRANCHES
                         ↑
                         │
                    [US] ★  ← no one here
                         │
    PHONES ←─────────────┼─────────────→ GPU SERVERS
                         │
                   DigiMorphLab         Prime Intellect ($70M)
                   (patent only)        Petals, Exo
                                        PRIMA.CPP
                         │
                    SEQUENTIAL LAYER SHARDING
```

---

## 1. Distributed Inference Frameworks

### Prime Intellect — $70.4M funded
- **What:** "Planetary-scale inference engine" for consumer GPUs over public internet. Pipeline-parallel vLLM optimized for 100ms+ latencies.
- **Distribution:** Sequential pipeline parallelism (layer-by-layer chain across nodes)
- **Models:** 32B (training), DeepSeek-R1 scale (inference)
- **Users:** Consumer GPU contributors across the internet
- **Funding:** $70.4M total. $15M seed extension (Feb 2025, Founders Fund; Karpathy, Clem Delangue as angels). $49.9M Series B (Dec 2025)
- **vs Us:** Still sequential layers — request must traverse ALL nodes. Targets GPU owners, not phone users. Our parallel tracks have no serial dependency chain.

### Petals (BigScience/Yandex Research)
- **What:** BitTorrent-style decentralized LLM serving. Each volunteer hosts a few layers, requests route through the chain.
- **Distribution:** Sequential layer sharding across internet-connected servers
- **Models:** Llama 3.1 405B, Mixtral 8x22B, BLOOM 176B
- **Users:** ~100 active nodes (via Kwaai initiative). Requires GPU servers.
- **Funding:** Academic, no VC. KwaaiNet (Rust rewrite targeting WASM/WebGPU for browser/mobile) is early-stage.
- **vs Us:** Sequential = latency proportional to node count. GPU servers only. We're parallel = latency determined by slowest single phone.

### Exo (exo-explore)
- **What:** Cluster everyday Apple devices (iPhones, iPads, Macs) into one AI inference cluster. P2P, no master node.
- **Distribution:** Dynamic model partitioning via MLX distributed. Thunderbolt 5 RDMA for local connections. Auto-discovery via mDNS.
- **Models:** Frontier models limited by aggregate memory
- **Users:** 19,000+ GitHub stars. Hobbyists with Apple Silicon clusters.
- **Funding:** Open source. Not production-ready (Oct 2025 analysis).
- **vs Us:** Still pipeline parallelism by layers. Requires fast local connections (Thunderbolt/USB4). Our parallel tracks work over slow mobile networks since tracks are independent.

### PRIMA.CPP (ICLR 2026)
- **What:** Distributed inference for 30-70B models on heterogeneous home clusters with mixed CPUs/GPUs, slow disks, Wi-Fi.
- **Distribution:** Pipelined-ring parallelism. Halda scheduler co-optimizes CPU/GPU per device.
- **Models:** 30-70B. Achieves 674ms/token for 70B on 4 consumer devices.
- **Funding:** Academic paper (ICLR 2026 poster). Claims 5-17x lower latency than llama.cpp, exo, dllama.
- **vs Us:** Still layer-based pipeline parallelism. Local Wi-Fi only. Not parallel-track splitting.

### Distributed Llama (b4rtaz)
- **What:** Connects home devices via Ethernet for accelerated LLM inference using tensor parallelism.
- **Distribution:** Tensor parallelism over Ethernet
- **Models:** Up to 70B+
- **Funding:** Solo developer project
- **vs Us:** Requires high-speed local Ethernet. Not designed for phones or WAN.

---

## 2. Crypto / Web3 GPU Marketplaces

All of these are **GPU rental marketplaces** — the model is NOT split. Each miner/node runs a complete model. Requires expensive GPUs.

| Project | What | Funding | Users/Scale |
|---------|------|---------|-------------|
| **Bittensor** | Decentralized AI service network, miners earn TAO tokens | $350M+ institutional | 100+ subnets, GPU servers |
| **io.net** | GPU marketplace on Solana | $30M Series A (Hack VC, Solana Labs) | 327K verified GPUs, $20M+ ARR |
| **Nosana** | GPU grid on Solana | Token-based (NOS) | 50K GPU hosts, 4.2K active |
| **Hyperbolic** | GPU marketplace + serverless inference | $20M (Variant, Polychain) | 40K+ developers, 1B+ tokens/day |
| **Gensyn** | Verifiable distributed ML training | $50-80M (a16z Crypto) | Testnet March 2025 |
| **Render Network** | Distributed GPU rendering → expanding to AI | Established token (RNDR) | Large network |

**vs Us:** These are compute marketplaces where providers need expensive GPUs. Our model: free compute from ordinary phones, model is split into pieces small enough for any device.

---

## 3. On-Device / Mobile Inference (Single Device)

| Project | What | Stage | vs Us |
|---------|------|-------|-------|
| **Cactus** (YC S25) | Cross-platform on-phone inference, <50ms TTFT | Beta, team of 8 | Single device only. No network distribution. |
| **WebLLM / MLC-LLM** | In-browser LLM via WebGPU/WASM, ~80% native perf | Open source (CMU) | Single browser tab. No distribution. |
| **Tether QVAC** | Edge-first inference runtime for heterogeneous devices | Backed by Tether, Dec 2025 | Single device runtime. Not distributed. |

**vs Us:** All run on ONE device. Limited to small models (≤7B quantized). We combine many phones to run models larger than any single phone could handle.

---

## 4. Closest Competitor: DigiMorphLab (Patent)

- **What:** Patent filed (AU, US, CN in 2024) for distributed LLM inference across mobile edge devices using MILP-based optimal task allocation.
- **Distribution:** Dynamic task distribution across multiple mobile devices
- **Status:** Patent + open-source docs. No known product, team, or funding.
- **vs Us:** Explicitly targets mobile devices for distributed inference — closest concept. But uses optimization-based partitioning of **sequential** computation, not parallel architectural tracks.

---

## 5. Key Architectural Validation

### Apple Parallel Track Transformer (February 2026)
- **Paper:** arXiv 2602.07306
- **What:** Transformer split into independent parallel tracks, synchronizing only at input/output boundaries. Up to 16x reduction in synchronization operations. Extended to PT-MoE.
- **Results:** 15-30% reduced TTFT, up to 31.9% increased throughput
- **Significance:** **Validates our core architectural hypothesis** — parallel tracks that merge at boundaries work as well as sequential layers. Apple designed it for datacenter multi-GPU; we apply it to phone swarms over mobile networks.

### ParaFormer (October 2025)
- Progressive approximation for parallel shallow transformers. Proves that reducing depth while adding parallel width maintains quality.

### OD-MoE: On-Demand Expert Loading for Edge (2025)
- Distributes MoE experts across edge nodes with predictive loading. Shows MoE is a natural fit for distribution — different experts on different devices.

### Kraken (NeurIPS 2024)
- Parallel branches with overlapped compute/communication for multi-device inference. +35% speed. Proves inter-branch communication during computation is key.

### Split Federated Learning for Mobile LLMs (June 2025)
- Edge-assisted split learning where phones handle lower layers, server handles upper. 79% memory reduction. Proves phone-based split model execution is feasible.

---

## 6. Competitive Advantages

### Architecture
- **Parallel tracks = no sequential bottleneck.** All competitors use pipeline parallelism where the slowest node determines total latency. Our tracks run independently — latency = single slowest track, not cumulative chain.
- **Apple-validated.** PT Transformer proven at scale, we adapt it to distributed phones.

### Economics
- **Free compute from users.** No GPU rental costs. Users donate idle phone compute in exchange for free AI (Compute Miles).
- **vs Crypto marketplaces:** They need $1000+ GPUs. We need any smartphone.

### Network Effect
- Every new user is BOTH a consumer AND a compute provider
- More users → more compute → smarter model → more users
- More users → more training signals → better model → more users
- Self-reinforcing flywheel that crypto marketplaces don't have (their providers ≠ consumers)

### Scalability
- **2B+ smartphones worldwide** vs ~10M discrete GPUs
- Model scales with network size: more phones → more tracks → smarter model
- Tiered participation: weak phone = 1 track, strong phone = 4 tracks, PC = 8+ tracks

---

## 7. Threats

| Threat | Severity | Mitigation |
|--------|----------|------------|
| **Prime Intellect** pivots to parallel tracks | High | They're deep into pipeline parallelism infra. Switching architecture is expensive. Our head start on PT-MoE for phones. |
| **Exo** adds internet tolerance | Medium | Their codebase assumes fast local links. Fundamental redesign needed. |
| **Apple** launches crowd-compute | High | Privacy policy makes this unlikely — Apple positions as "on-device AI" company. They won't aggregate user data/compute. |
| **Google/Meta** copies architecture | Medium | They have the research teams. But crowd-compute conflicts with their ad-based business model (they WANT centralized data). |
| **Crypto projects** add model splitting | Low | Token economics ≠ our Compute Miles. Different user base (miners vs consumers). |

---

## 8. The Unclaimed Intersection

No existing project combines ALL of:
1. ✅ Parallel-track architecture (not sequential layer sharding)
2. ✅ Consumer phones as compute nodes (not GPU servers)
3. ✅ Free compute economy (not paid GPU rental)
4. ✅ Model improves from user signals (not static serving)
5. ✅ Any subset of tracks works (graceful degradation)
6. ✅ Variable depth (0-4 groups) for different task complexity

## 9. Why We're Different From ALL of Them

Every project above falls into one of two categories:
1. **Selling compute** — corporations/crypto networks that monetize GPU access
2. **Single-device** — running small models on one phone/browser

We are neither. We're building **public infrastructure** — like the internet itself.

| | Them | Us |
|---|---|---|
| **Who owns the model?** | A company or token holders | No one. It's distributed across millions of phones. |
| **Who controls it?** | The company (censorship, policies, kill switch) | No one. No central server to shut down. |
| **Who profits?** | The company/miners | Users themselves (Compute Miles are theirs) |
| **Business model?** | Sell compute/API/tokens | None. Open infrastructure. |
| **Why build it?** | Money | Eliminate civilizational risk of AI controlled by governments/corporations |

The moat isn't a business advantage — it's architectural. The model physically cannot be controlled by one entity because it exists across millions of devices. No company can replicate this because companies need control to monetize. Our lack of monetization IS the feature.
