# Enhancement Options for PT-MoE Architecture

Ideas to explore — not yet decided, need testing.

---

## 1. Multi-Token Prediction (Meta, 2024)

**Problem:** Model predicts 1 token per forward pass = 1 network round-trip per token.

**Solution:** Train model to predict 4-8 tokens at once.

```
Standard:      "Прив" → "ет" → "," → " как" → " дела"   = 4 round-trips
Multi-token:   "Прив" → "ет, как дела?"                   = 1 round-trip
```

**Impact:**
- Keyboard: 4× faster suggestions (whole phrases, not single words)
- API: 4× faster generation
- Network: 4× fewer round-trips to phone swarm

**Implementation:**
- Add N prediction heads (one per future token) during training
- Only head 1 is needed for single-token inference (backward compatible)
- Meta showed quality IMPROVES (multi-token acts as regularization)

**Cost:** ~20% more compute during training, zero cost at inference.

**Priority:** HIGH for keyboard (phrase prediction is the killer feature)

**Reference:** "Better & Faster Large Language Models via Multi-token Prediction" (Meta, 2024)

---

## 2. Case-Insensitive Tokenization (Keyboard Mode)

**Problem:** "Привет", "привет", "ПРИВЕТ" = 3 separate tokens. Each gets 1/3 of training data.

**Solution:** Lowercase everything during tokenization.

```
With case:     ~32,000 tokens, each sees fewer examples
Without case:  ~20,000 effective tokens, each sees 3× more data
```

**Capitalization recovery (post-processing, no neural network):**
- Rule 1: After sentence boundary (.!?) → capitalize first letter
- Rule 2: Named entity dictionary → "москва" → "Москва"
- Rule 3: Abbreviation dictionary → "api" → "API"
- Rule 4: User preference learning → adapt to user's style

**Two modes:**
- Keyboard mode: lowercase model + capitalization rules (faster, smaller vocab)
- API mode: full case-sensitive model (for code, formal text)

**Impact:**
- ~30-40% smaller effective vocabulary
- Each token trained on 3× more data → better predictions
- Particularly helps rare words that have multiple case forms

**Risk:** Code generation needs case (myVariable, MyClass). Solution: separate code track keeps case.

**Priority:** MEDIUM — easy to implement, measurable improvement

---

## 3. Subword Compression for Network Transfer

**Problem:** Sending hidden states (1024 × float16 = 2KB per token) over mobile network.

**Options explored:**
- float16 → int8: 2× compression, minimal quality loss ✅ DECIDED
- Delta encoding: send difference from previous step
- Sparse transfer: only send top-K changed dimensions

**Priority:** int8 is decided (see SOLUTIONS.md). Delta/sparse are V3+.

---

## 4. Adaptive Sequence Length

**Problem:** Fixed seq_len=2048 wastes compute on short inputs.

**Solution:** Dynamic batching — short inputs (keyboard, 10-50 tokens) get shorter context.

```
Keyboard autocomplete: seq_len=256  (last ~200 characters)  → 8× faster
Chat message:          seq_len=1024 (last few messages)     → 2× faster
API document:          seq_len=4096 (full context)           → standard
```

**Impact:** Keyboard inference 8× faster (most computation is in attention, which is O(n²) in sequence length).

**Priority:** HIGH for keyboard latency

---

## 5. Vocabulary Specialization per Track

**Problem:** Generic 32K vocabulary covers everything poorly.

**Idea:** Each track has a small specialized vocabulary extension.

```
Shared vocabulary:  32K tokens (common words, all languages)
Code track:        +2K tokens (programming keywords, syntax patterns)
Medical track:     +2K tokens (medical terms, drug names)
User personal:     +500 tokens (user's frequent words/phrases)
```

**Impact:** Specialized tracks tokenize their domain 2-3× more efficiently.

**Risk:** Complexity. Different tracks produce different token IDs → merge layer needs to handle this.

**Priority:** LOW — explore in V4+

---

## 6. Speculative Decoding for Distributed Inference

**Problem:** Each token requires full network round-trip to phone swarm (~115ms).

**Solution:** Small local model on user's device generates 4-8 draft tokens. Phone swarm verifies/corrects in ONE round-trip.

```
Without speculative:  8 tokens × 115ms = 920ms
With speculative:     local draft (10ms) → verify all 8 at once (115ms) = 125ms
```

**Impact:** 4-8× faster generation for end user.

**Requirement:** Small draft model (~50MB) on user's device.

**Priority:** HIGH for V3 (mentioned in SOLUTIONS.md roadmap)

---

## 7. Hierarchical Merge for 64+ Tracks

**Problem:** CTA trained on 8 tracks has glass ceiling — doesn't generalize to 64.

**Solution:** Group tracks into clusters of 8, merge within cluster, then merge clusters.

```
64 tracks → 8 groups of 8 → CTA(8) per group → 8 results → CTA(8) → final
```

**Impact:** Scales to 512+ tracks without quality degradation.

**Cost:** +85ms per merge level (coordinator-to-coordinator, fast network).

**Priority:** V3-V4 (when network exceeds 8 active tracks)

**See:** SOLUTIONS.md "Hierarchical Merge" section

---

## 8. Case-Insensitive Tokenization — Updated Analysis

Real benefit is NOT "more data" but "less wasted model capacity."

Capital letters are 95%+ predictable by trivial rules (first letter after period). Model wastes parameters learning this pattern instead of useful knowledge. For small models (700M active) every % of capacity matters.

Implementation: lowercase_block(input) → model → capitalize_block(output). Two functions, no architecture changes.

**Priority:** LOW-MEDIUM — free optimization, do for keyboard mode, skip for code/API

---

---

## 9. Predictive Pre-computation — Zero Latency Illusion

**STATUS: V2 — DO IT**

While user types a word, local model predicts the full word and sends it to phone swarm BEFORE user finishes typing. By the time user presses space, the prediction is already computed.

```
User types: "П" "р" "и" ...
Local model: 90% sure this is "привет" → sends to network NOW
User finishes "привет " → result already waiting → 0ms perceived delay
Wrong 10-20% of the time → discard, recompute (user doesn't notice)
```

**Impact:** 0ms perceived latency instead of 115ms for 80-90% of requests.
**Risk:** Wasted compute on wrong predictions (~15%). Acceptable.
**Implementation:** Local model runs alongside keyboard, fires network request at 3+ characters.

---

## 10. Backspace = Golden Signal

**STATUS: V2 — DO IT**

Track what happens AFTER user accepts a prediction. "Accepted then deleted" = model deceived the user. 10× worse than "not accepted."

```
Signal weights for training:
  Prediction not accepted:           -1  (normal)
  Accepted and kept:                 +1  (good)
  Accepted, then deleted:           -10  (model broke trust)
  Accepted, then extended:           +2  (model gave good foundation)
```

**Impact:** Higher quality training signal → faster model improvement.
**Risk:** Zero. Pure software, 10 lines of code.

---

## 11. Anti-Prediction — Know When to Shut Up

**STATUS: V2 — DO IT**

Best prediction is sometimes NO prediction.

```
Password field detected        → suppress all predictions
3 rejections in a row          → go silent for 30 seconds
Pattern looks like phone/card  → suppress
User is deleting rapidly       → suppress (they're fixing, not writing)
```

**Impact:** User trust. Not a metric improvement — a retention improvement.
**Risk:** Zero.

---

## 12. Sleep Training (2× rating for full data mode) — Phone Learns While You Sleep

**STATUS: V3 — TRY IT**

Phone on charger at night = 8 hours of free compute. Accumulate a week of typing data, fine-tune local LoRA adapter on Sunday night. Full data mode earns 2× rating per token trained.

**Problem at V2:** One day of typing (~100-500 sentences) is too little for fine-tuning. Need weekly accumulation.

```
Week 1-4: Collect signals locally on device
Sunday night: LoRA fine-tune on accumulated data (~2000-5000 sentences)
Monday morning: Keyboard understands user better
```

**Impact:** 10M phones × 8 hours × 52 Sundays = 4.16B free GPU-hours/year.
**Risk:** Phone heating while "charging" — need thermal throttling.

---

## 13. Distillation Waterfall — Big Model Teaches Small

**STATUS: V3 — TRY IT**

When user makes API request (big model, 64 tracks), save the output. Reformat as (context → next token) pairs. Feed to keyboard model for fine-tuning.

```
Month 1:  Keyboard dumb, API smart
Month 6:  Keyboard medium (learned from API outputs)
Month 12: Keyboard smart (absorbed thousands of API answers)
```

**Problem:** API generates paragraphs, keyboard predicts tokens — different tasks. Need adapter layer to bridge.

**Impact:** Keyboard gets smarter for free every time someone uses API.

---

## 14. Swarm Awareness — Collective Intelligence from Keyboard Data

**STATUS: V3-V4 — MAJOR FEATURE**

Millions of people typing = real-time pulse of what's happening everywhere. Not what people post publicly (Twitter), but what they're actually writing to friends right now. Anomaly detection on aggregated, anonymized word frequencies reveals events before news does.

**Core mechanic:** anomaly_score = current_frequency / avg_frequency_last_30_days. "привет" at 10K/hr = normal (score 1.0). "наводнение" jumping from 5/hr to 5000/hr = 🔴 signal (score 1000). Cluster detection (multiple related emergency words spiking together) eliminates false positives.

**Why it matters:**
- Earthquake detection in 30 seconds (faster than seismographs)
- Uncensorable awareness in countries with media blackout (it's a keyboard, can't block it)
- Viral growth: one real emergency event = millions of installs
- Privacy-safe: only anonymized word counters, no text, no identity, open source

**Viral growth thesis:** Telegram grew on protests. We grow on "this keyboard tells you what's happening before the news does." The keyboard is the trojan horse, Swarm is the real product.

**Full design:** See [SWARM.md](./SWARM.md)

**Priority:** V3 (global trends), V3.5 (regional signals), V4 (emergency alerts)

---

## 15. Phantom Tokens — Hidden Chain-of-Thought

**STATUS: HYPOTHESIS — V4+**

Model generates invisible "thinking" tokens before the visible answer. Like chain-of-thought but inside the model, not shown to user.

```
Input: "Столица Бразилии —"
Hidden: [phantom: country=Brazil, type=capital, not Rio]
Output: "Бразилиа"
```

Cost: 2-4 extra tokens of computation. Benefit: more accurate for knowledge-heavy queries. For small models, this could be the difference between right and wrong answers.

**Research needed:** Does this help for 700M-6B param models? Meta's "inner monologue" work suggests yes for larger models.

**Priority:** HYPOTHESIS — test in V4 when model is larger

---

## 16. Compression Race — Tracks Compete on Information Theory

**STATUS: HYPOTHESIS — V4+**

Better model = better compressor (Shannon). Instead of perplexity, rank tracks by compression ratio on user's text.

```
Track A: compresses to 2.3 bits/char
Track B: compresses to 2.1 bits/char ← objectively better
Track B contributes more to model quality → gets more requests → higher rating
```

Objective quality metric. No human votes needed. Better compression = more useful training = faster rating growth.

**Priority:** HYPOTHESIS — interesting for reputation system in expertise marketplace

---

## DROPPED IDEAS

**Typing Speed as Context** — DROPPED. Correlation between speed and intent is weak and unreliable. Fast typing = knows what to write OR angry/rushing. Can't tell. Research shows +0.5-1% improvement — not worth complexity.

**Local Trending** — DROPPED. Requires approximate geolocation even when anonymized. One privacy scandal = dead product. Marginal benefit doesn't justify risk.

---

## Evaluation Priority

| # | Enhancement | Impact | Effort | When |
|---|------------|--------|--------|------|
| 9 | Predictive pre-computation | 0ms perceived latency | Medium | V2 |
| 1 | Multi-token prediction | 4× generation speed | Medium | V2 |
| 10 | Backspace golden signal | Better training data | Low | V2 |
| 11 | Anti-prediction | User trust/retention | Low | V2 |
| 2 | Case-insensitive (keyboard) | Less wasted capacity | Low | V2 |
| 4 | Adaptive seq_len | 8× keyboard speed | Low | V2 |
| 3 | int8 network transfer | 2× less bandwidth | Low | V2 (decided) |
| 6 | Speculative decoding | 4-8× generation speed | Medium | V3 |
| 12 | Sleep training | Free compute + personalization | Medium | V3 |
| 13 | Distillation waterfall | Keyboard learns from API | Medium | V3 |
| 7 | Hierarchical merge | 64+ track scaling | Medium | V3-V4 |
| 5 | Vocab specialization | Domain efficiency | High | V4+ |
