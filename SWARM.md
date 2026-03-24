# Swarm Awareness — Collective Intelligence from Keyboard Data

## Vision

Millions of people typing on keyboards in real-time = the world's fastest information network. Not what people CHOOSE to post (Twitter), but what they're ACTUALLY writing right now (messages to friends, colleagues, family). Raw, unfiltered, real-time pulse of humanity.

**The keyboard is the trojan horse. Swarm Awareness is the real product.**

---

## How It Works

### Data Collection (on-device, privacy-first)

Each device locally counts word frequencies and sends ONLY anonymized counters:

```
What is sent:     { word_id: 4521, count: 7, region: "TR", hour: 14 }
What is NOT sent:  context, sentences, user identity, precise location, word order
```

- No raw text ever leaves the device
- Word IDs, not actual words (server maps IDs to words)
- Region = country or large area (1M+ population), NEVER city-level unless 1000+ users
- Aggregated hourly, not real-time per keystroke

### Anomaly Detection (on coordinator)

Not raw frequency — **anomaly score** relative to baseline:

```
anomaly_score = current_frequency / avg_frequency_last_30_days

Examples:
  "привет":     10,000 / 10,000 = 1.0    → normal, ignore
  "наводнение":  5,000 / 5      = 1,000  → 🔴 something is happening
  "эвакуация":     800 / 2      = 400    → 🔴 confirms emergency
```

Threshold: anomaly_score > 10 → candidate signal.

### Cluster Detection (reduces false positives)

Single word anomaly = might be noise. Multiple RELATED words = real signal.

```
Cluster: "наводнение" ×1000 + "эвакуация" ×400 + "вода" ×200 + "мост" ×150
  → Emergency cluster detected → 🔴 alert

vs.

Single: "футбол" ×50 (but no related emergency words)
  → Probably a match → 🟡 trend, not emergency
```

Cluster detection uses pre-built word association graphs:
- Emergency cluster: disaster + evacuation + help + rescue + ...
- Health cluster: fever + cough + hospital + doctor + ...
- Transport cluster: traffic + closed + accident + delay + ...
- Political cluster: protest + rally + police + ...

### Signal Levels

```
🟢 Trend (anomaly 10-50×):     "AI" trending globally this week
🟡 Notable (anomaly 50-200×):  "метро закрыто" trending in Moscow
🔴 Alert (anomaly 200-1000×):  "землетрясение" spike in region
🚨 Emergency (anomaly 1000×+): Multi-word emergency cluster detected
```

---

## User Experience

### Normal Day

```
Keyboard works as usual. Small 🌊 icon in corner.
Tap 🌊 → "Trending: World Cup, AI, iPhone 18"
Not critical, just interesting.
```

### When Something Happens

```
🌊 icon turns 🔴
"🔴 наводнение +5000% in your region"
"🔴 эвакуация trending"

User taps → sees: thousands of people nearby are typing about flooding
Knows what's happening BEFORE official news
Tells friends: "install this keyboard, it shows what's really happening"
```

### Emergency Mode

```
When anomaly > 1000× AND emergency cluster detected:
  → Push notification to ALL users in affected region
  → "⚠️ Possible earthquake in your area — stay safe"
  → No opt-in needed for life-safety alerts
```

---

## Why This is Uncensorable

```
Telegram:
  ❌ Government sees: "user X joined protest channel"
  ❌ Can block the app
  ❌ Can demand chat logs

Swarm:
  ✅ Government sees: "user X uses a keyboard" (so does everyone)
  ✅ Can't block a keyboard app without blocking all keyboards
  ✅ No chat logs — only anonymized word counters
  ✅ No central server with "full picture" (federated aggregation)
  ✅ Open source — anyone can verify what's sent
```

A government would need to ban keyboards to stop Swarm. That's not happening.

---

## Viral Growth Mechanics

```
1. Person installs keyboard (free, good predictions)
2. One day: 🔴 alert — "earthquake trending in your region"
3. Person: "How does it know?!" → tells 5 friends
4. Friends install → more data → alerts more accurate
5. Next alert → each tells 5 more friends
6. One real emergency event = millions of installs in days
```

**Not gradual growth. Explosive growth from real events.**

One earthquake in Turkey → all of Turkey installs in a week.
One protest in Iran → million downloads in a day.
One flood in Brazil → South America discovers the app.

---

## Privacy Architecture

### What We Can NEVER Do

- ❌ Read what any specific person typed
- ❌ Identify who contributed to a trend
- ❌ Provide data to governments (we don't have it)
- ❌ Track individual users across time
- ❌ Correlate word patterns to identify users

### How We Enforce This

```
On-device aggregation:
  Raw text → word tokenizer → counter increment → discard text
  Text NEVER stored, even locally (streaming counter)

Differential privacy:
  Add random noise to counters before sending
  Individual contribution is mathematically unrecoverable

Minimum threshold:
  Region data only if 1000+ active users in region
  Prevents small-group identification

Open source:
  Every line of code is auditable
  Community can verify no backdoors
  Reproducible builds (binary matches source)

No logs:
  Coordinator processes counters in memory
  No persistent storage of per-device data
  Even if server is seized — nothing to find
```

### Trust Model

```
Users trust us because:
  1. Code is open source (verify yourself)
  2. Only word counters are sent (no text, no context)
  3. Differential privacy (mathematically provable)
  4. Federated aggregation (no central database)
  5. Minimum thresholds (can't track individuals)

Even if we WANTED to spy — the architecture makes it impossible.
Privacy by design, not by policy.
```

---

## Real-World Use Cases

### Natural Disasters (seconds, not hours)

```
Earthquake:  "трясёт" ×10000 spike → alert in 30 seconds
             Official seismograph data: 2-5 minutes
             News coverage: 15-30 minutes

Flood:       "вода поднимается" cluster → alert in 5 minutes
Wildfire:    "дым" + "горит" + "пожар" cluster → early warning
Tsunami:     "волна" + "берег" + "уходим" → fastest possible warning
```

### Public Health (weeks before official data)

```
Flu epidemic:  "температура" + "кашель" + "больничный" rising
               Detected 1-2 weeks before official statistics
               (Like Google Flu Trends, but from actual human communication)

COVID-like:    "новый вирус" + "карантин" + "маска" emerging cluster
               Early warning before WHO declares anything
```

### Censored Information (impossible to block)

```
Country with media blackout:
  Official news: "Everything is fine"
  Swarm data: "протест" ×5000, "полиция" ×3000, "слезоточивый газ" ×800

  People see: something is REALLY happening, regardless of what TV says
  No one can be arrested for "participating" — they were just typing
```

### Urban Utility

```
"пробка" + "МКАД" = traffic accident
"отключили" + "воду" = utility outage
"задержка" + "рейс" + "Шереметьево" = airport delays
"очередь" + "МФЦ" = long wait at government office
```

---

## Comparison with Existing Solutions

| Feature | Twitter/X | Google Trends | Waze | Our Swarm |
|---------|-----------|--------------|------|-----------|
| Real-time | Minutes | Hours | Real-time | Seconds |
| Requires posting | Yes | Search needed | Report needed | No action needed |
| Censorable | Yes (ban app) | Yes (block Google) | Yes | No (it's a keyboard) |
| Privacy | Public posts | Search history tracked | Location tracked | Anonymous counters |
| Coverage | Tech-savvy users | Desktop users | Drivers only | Everyone who types |
| Works offline | No | No | Partial | Aggregation queued |

---

## Roadmap

```
V2:    Keyboard only. Swarm NOT enabled. Build user base.
       Secretly collect baseline word frequencies (opt-in, for "improving predictions").

V3:    Global trends (opt-in). "What's hot worldwide?"
       Test anomaly detection on real events.
       No geographic component yet.

V3.5:  Regional signals (threshold 1000+). "What's happening near you?"
       🔴 alerts for emergency clusters.
       First viral growth events expected here.

V4:    Emergency mode. Push notifications for 1000×+ spikes.
       "⚠️ Possible earthquake in your area"
       Partnerships with emergency services (optional, not required).

V5+:   Community-voted expansion of alert types.
       Open API for researchers (anonymized, aggregated).
       Local community awareness features.
```

---

## Why This Matters for the Mission

```
Mission: Decentralize AI, give power to people, not corporations.

Swarm extends this to INFORMATION:
  Not just AI belongs to people.
  AWARENESS belongs to people.

  No government controls what you know.
  No corporation filters what you see.
  No algorithm decides what's "trending."

  Raw, unfiltered pulse of millions of humans.
  Available to everyone. Controlled by no one.
```
