# GPU Network — Model Benchmarks

All experiments use the PT-MoE (Parallel Track Mixture-of-Experts) architecture on 2× NVIDIA L20 48GB GPUs.

## Architecture Key

```
Track  = independent parallel expert (runs on one phone/device)
Group  = sequential round of computation (tracks communicate between groups)
TDM    = Token-Dependent Merge (each token chooses which tracks matter most)
CTA    = Cross-Track Attention (tracks see each other's outputs before merging)
Div    = Diversity Loss (penalizes tracks for producing similar outputs)
Drop   = DropTrack (randomly disables tracks during training for resilience)
```

## Quick Ablation (10M tokens)

Purpose: fast architectural comparison. Which components help?

| Model | Architecture | Params | Loss | Time | Notes |
|-------|-------------|--------|------|------|-------|
| A_baseline | 1 group × 4 tracks, scalar merge | 1,324M | 8.14 | 20 min | Baseline |
| B_tdm | A + TDM | 1,328M | 8.12 | 20 min | TDM helps slightly |
| C_cta | B + CTA | 1,328M | 8.12 | 20 min | CTA helps slightly |
| D_full | C + Div + Drop | 1,328M | 8.30 | 20 min | **Worse!** Regularization hurts |
| E_2groups | D but 2 groups × 4 tracks | 2,624M | 8.56 | 40 min* | **Worst.** 2 groups = bad |
| F_8tracks | D but 1 group × 8 tracks | 2,587M | 8.09 | 40 min* | **Best.** More tracks = better |
| G_2tracks | D but 1 group × 2 tracks | 699M | 8.29 | 10 min | Fewer tracks = worse |

*E, F used 8-bit Adam due to memory constraints (unfair comparison — see Solid Ablation for fair results)

### Quick Ablation Findings

1. **More tracks = better**: F(8) > A-D(4) > G(2)
2. **TDM + CTA work**: B, C slightly better than A
3. **Diversity + DropTrack hurt**: D worse than C at 10M tokens
4. **2 groups = worst**: E lost to everything including baseline
5. **1 group is optimal**: minimum latency + best quality

## Solid Ablation (100M tokens)

Purpose: validate findings at scale with fair comparison.

| Model | Architecture | Params | Loss | Time | GPU | Optimizer |
|-------|-------------|--------|------|------|-----|-----------|
| C_cta | 1g × 4t + TDM + CTA | 1,328M | **3.81** | 7.5h | 1× L20 | AdamW fp32 |
| D_full | 1g × 4t + TDM + CTA + Div + Drop | 1,328M | 4.36 | 7.5h | 1× L20 | AdamW fp32 |
| F_8tracks | 1g × 8t + TDM + CTA + Div + Drop | 2,587M | 4.40 | 17h | 1× L20 | AdamW **8-bit** |
| **F2_clean** | **1g × 8t + TDM + CTA** | **2,587M** | **2.85** | **17h** | **2× L20 FSDP** | **AdamW fp32** |

### Why F_8tracks (4.40) lost to C (3.81)

F_8tracks used 8-bit Adam optimizer and batch_size=1 due to memory constraints. This was an unfair comparison. F2_clean used the same AdamW fp32 optimizer as C (via FSDP across 2 GPUs), proving the architecture is superior when conditions are equal.

### Solid Ablation Findings

1. **8 tracks >> 4 tracks**: F2 (2.85) vs C (3.81) = 25% better with fair comparison
2. **Diversity + DropTrack still hurt at 100M**: D (4.36) vs C (3.81)
3. **Fair conditions matter**: same model, different optimizer = completely different results (F: 4.40 vs F2: 2.85)

## Winner: F2 Architecture

```
Configuration:  1 group × 8 tracks
Components:     TDM (Token-Dependent Merge) + CTA (Cross-Track Attention)
NOT included:   Diversity Loss, DropTrack (proven harmful)
Total params:   2,587M
Active params:  742M (per token, top-2 of 8 experts)
Per track:      314M total, 88M active
Track size:     ~157 MB (INT4 quantized)
```

### What Each Component Does (Plain English)

**TDM**: For each word, the model decides which track matters most. "Hello" → Language track 70%. "def main" → Code track 70%. Smart mixing instead of dumb averaging.

**CTA**: Before combining results, tracks see what others computed. Code track learns that Language track identified "this is Russian" → outputs code with Russian comments.

**Diversity Loss** (rejected): Penalizes tracks for similar outputs. Theory: forces specialization. Practice: made quality worse. Tracks specialize naturally through TDM + CTA.

**DropTrack** (rejected): Randomly disables tracks during training. Theory: resilience. Practice: model never learned to use all tracks together properly.

**2 Groups** (rejected): Two rounds of "discussion" between tracks. Theory: deeper reasoning. Practice: added latency AND hurt quality. 1 group (single round) is optimal.

## Training Trajectory: F2 on 100M

```
Step     Epoch   Loss    Tokens
──────────────────────────────
   50    0       10.82    3M
  550    0        6.02   36M
  850    1        5.27   56M
 1250    1        4.83   82M
 1750    1        4.61  115M
 2350    1        3.76  154M  ← overtook C (3.81) here
 2800    1        3.48  184M
 4150    2        2.95  272M
 4550    2        2.85  298M  ← final
```

## SFT Training

F2 architecture fine-tuned on supervised data (Russian SFT, high-quality examples scored by opus).

| Metric | Value |
|--------|-------|
| Architecture | F2 (1 group x 8 tracks + TDM + CTA) |
| Pretrain loss (100M) | 2.85 |
| **SFT loss** | **2.20** |
| SFT checkpoint | step 45,798 |
| SFT data | ~40K Russian examples |
| Hardware | 2x L20 via FSDP |

This SFT checkpoint (loss 2.20) serves as the baseline for all fusion experiments below.

---

## Model Fusion V2: Borrowing Intelligence from SOTA Models

**Idea:** Replace 2 of 8 PT-MoE tracks with frozen layers from Qwen 3.5-27B and Qwen 2.5-72B. These layers contain world knowledge from trillions of tokens. Trainable projections (1024 <-> donor_d_model) + CTA + Merge learn to integrate their outputs.

```
Track 0-4: our tracks        (trainable, lr=1e-5)
Track 5:   Qwen 3.5-27B      (frozen INT4, proj trained at lr=3e-4)
Track 6:   Qwen 2.5-72B      (frozen INT4, proj trained at lr=3e-4)
Track 7:   our track          (trainable, lr=1e-5)
```

### Results

| Run | Steps | Final Loss | Baseline | Donors | GPU Memory |
|-----|-------|-----------|----------|--------|------------|
| V2.0 (straight-through grad) | 5K | 3.22 | 2.20 | 27B + 72B + 397B MoE | 74 GB |
| **V2.1 (real gradients)** | **20K** | **2.69*** | **2.20** | **27B + 72B** | **39 GB** |

*V2.1 still in progress at step 6650/20000. Loss still decreasing.

### Key Findings

1. **Real gradients >> straight-through**: V2.1 reached 2.69 at step 6650, V2.0 plateaued at 3.22 after 5K
2. **MoE donors don't work**: Qwen 3.5-397B MoE contributed 0% merge weight with straight-through gradient. Removed in V2.1, freeing 27 GB VRAM
3. **Dense donors work**: Qwen 2.5-72B reaches up to 20% merge weight (token-dependent)
4. **INT4 quantization is practical**: 72B model = 1.5 GB on GPU, real gradients flow through bnb Linear4bit

### Merge Weight Distribution (V2.1, step 6000)

| Tracks 0-4 (ours) | Track 5 (Qwen 27B) | Track 6 (Qwen 72B) |
|--------------------|--------------------|--------------------|
| 85.6% | 4.4% | 9.5% |

Full results and code: [experiments/fusion_v2/](./experiments/fusion_v2/)

---

*All experiments reproducible from code in this repository. Contact: kustyuka@gmail.com | Telegram: @yuka_k*
