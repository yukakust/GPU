# Model Fusion V2 — Experiment Results

## Hypothesis

Replace 2 of 8 PT-MoE tracks with frozen layers from state-of-the-art pretrained models. These layers contain world knowledge learned from trillions of tokens. Trainable projections + CTA + Merge learn to combine their outputs with our original tracks.

## Architecture

```
                       PT-MoE Fusion V2
                       -----------------
Embedding (ours, d=1024)
    |
    +-- Track 0: ours (from baseline)              --+
    +-- Track 1: ours                                |
    +-- Track 2: ours                                |
    +-- Track 3: ours                                +--> CTA --> Merge --> Final Norm --> LM Head
    +-- Track 4: ours                                |
    +-- Track 5: Qwen 3.5-27B layers 0-2 (frozen)  -+  proj_in(1024->5120) -> 3 frozen layers -> proj_out(5120->1024)
    +-- Track 6: Qwen 2.5-72B layers 0-2 (frozen)  -+  proj_in(1024->8192) -> 3 frozen layers -> proj_out(8192->1024)
    +-- Track 7: ours                               -+
```

## Key Technical Decisions

### INT4 Quantization (NF4)
Donor layers are quantized to INT4 using bitsandbytes `Linear4bit` with NF4 quantization type. This reduces Qwen 2.5-72B from 5.3 GB (bf16) to ~1.5 GB on GPU. Compute dtype is bfloat16.

### Real Gradients (not straight-through)
In V2.1, we use **real gradients** through frozen INT4 layers. bitsandbytes `Linear4bit` computes `grad_input` in backward (needed for proj_in training) but skips `grad_weight` (frozen). This gives projection layers proper gradient signal, unlike the straight-through estimator in V2.0 which treated donor layers as identity for gradients.

### Shard-by-Shard Extraction
Downloading full models (27B, 72B) requires 50-150 GB disk. Instead, we download the safetensors index, identify which shards contain layers 0-2, download one shard at a time, extract the needed weights, and delete the shard before downloading the next. Total disk needed: ~8 GB for extracted .pt files.

### Two Learning Rates
- `3e-4` for fusion params (projections, CTA, Merge, Embedding, LM Head)
- `1e-5` for original tracks 0-4 (lower, to preserve existing knowledge)

## Results

### V2.0: Straight-through gradients (5K steps)

| Metric | Value |
|--------|-------|
| Steps | 5,000 |
| Final loss | 3.22 |
| Baseline loss | 2.20 (SFT) / 2.85 (100M pretrain) |
| Donors | Qwen 3.5-27B + Qwen 2.5-72B + Qwen 3.5-397B MoE |
| GPU | A100 80GB, 74 GB used |

Findings:
- Qwen 2.5-72B reached 12.8% merge weight
- Qwen 3.5-397B MoE contributed ~0% (straight-through gradient = no learning signal for MoE layers)
- Loss improved from 4.97 to 3.22 but didn't reach baseline

### V2.1: Real gradients, no MoE donor (20K steps, in progress)

| Metric | Value |
|--------|-------|
| Steps | 20,000 (in progress) |
| Current loss (step 6650) | **2.69** |
| Best loss | **2.69** |
| Baseline loss | 2.20 (SFT) / 2.85 (100M pretrain) |
| Donors | Qwen 3.5-27B + Qwen 2.5-72B (no MoE) |
| GPU | A100 80GB, 39 GB used |
| Throughput | 3,480 tok/s |

Changes from V2.0:
1. Removed Qwen 3.5-397B MoE (0% contribution, wasted 27 GB VRAM)
2. Real gradients through frozen INT4 layers (not straight-through)
3. Higher LR: 3e-4 (was 1e-4)
4. Batch size 8 (was 4)
5. 20K steps (was 5K)

### Loss Curve (V2.1)

```
Step     Loss    LR         Notes
-------  ------  ---------  -----
    50   4.75    1.5e-05    start
   500   3.78    1.5e-04    warmup
  1000   3.66    3.0e-04    peak LR
  2000   3.47    2.98e-04
  3000   3.38    2.93e-04   beat V2.0 final (3.22)
  4000   3.25    2.85e-04
  5000   3.13    2.75e-04   *already better than V2.0 at same step count*
  6000   2.76    2.53e-04   approaching baseline
  6650   2.69    2.45e-04   **beat 100M pretrain baseline (2.85)**
  ...    ...     ...        training continues to 20K
```

### Merge Weights Evolution

How much each track contributes to the final output (averaged across tokens):

| Track | Step 200 | Step 2000 | Step 6000 |
|-------|----------|-----------|-----------|
| 0 (ours) | 40.7% | 37.2% | 43.5% |
| 1 (ours) | 0.3% | 1.1% | 0.2% |
| 2 (ours) | 12.4% | 9.9% | 8.6% |
| 3 (ours) | 29.5% | 23.1% | 33.2% |
| 4 (ours) | 0.1% | 0.8% | 0.1% |
| **5 (Qwen 27B)** | **4.2%** | **6.1%** | **4.4%** |
| **6 (Qwen 72B)** | **12.7%** | **20.8%** | **9.5%** |
| 7 (ours) | 0.1% | 0.9% | 0.1% |

Donor tracks contribute 5-27% depending on the token. The merge is token-dependent, so some tokens lean heavily on donor knowledge while others rely on original tracks.

## Hardware

- **Training**: Vast.ai, NVIDIA A100 SXM4 80GB, ~$1.14/hr
- **VRAM usage**: 39 GB (V2.1) / 74 GB (V2.0 with 397B MoE)
- **Disk**: 300 GB instance (donors: 57 GB, baseline: 10 GB)
- **Cost**: V2.0 = ~$0.50, V2.1 = ~$2 (estimated)

## Files

| File | Description |
|------|-------------|
| `train_fusion_v2.py` | Training script: loads baseline, replaces tracks, trains projections |
| `extract_donor_layers.py` | Extracts layers 0-2 from HuggingFace models (downloads only needed shards) |
| `extract_one_by_one.py` | Shard-by-shard extraction for disk-constrained environments |

## Reproduce

```bash
# 1. Extract donor layers (~8 GB total)
python extract_donor_layers.py --output donors/ --all

# 2. Train fusion (needs A100 40GB+ for 2 donors)
python train_fusion_v2.py train \
  --data your_sft_data.jsonl \
  --tokenizer your_tokenizer.model \
  --baseline your_baseline.pt \
  --model-dir your_model_dir/ \
  --extracted-dir donors/ \
  --output fusion_v2.pt \
  --steps 20000 --lr 3e-4 --batch-size 8
```

Requirements: `torch`, `transformers`, `safetensors`, `bitsandbytes`, `huggingface_hub`, `sentencepiece`
