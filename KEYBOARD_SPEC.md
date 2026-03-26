# GPU Keyboard — Platform Specification

Build a keyboard for any platform (Android, iOS, Web) that connects to the GPU distributed AI network.

**Contact:** kustyuka@gmail.com | Telegram: @yuka_k

---

## Architecture Overview

```
┌──────────────────┐         ┌──────────────────┐
│   Keyboard App   │◄──WSS──►│  Predict Server   │
│  (iOS/Android)   │         │  (FastAPI + WS)   │
│                  │──HTTP──►│                    │
│  - T9 offline    │         │  - Model inference │
│  - Autocorrect   │         │  - Signal store    │
│  - Prediction UI │         │  - Rating   │
│  - Signal collect│         │                    │
└──────────────────┘         └──────────────────┘
```

Two communication channels:
1. **WebSocket** — real-time predictions (low latency)
2. **HTTP POST** — batch signal upload (background, non-blocking)

---

## 1. WebSocket Prediction Protocol

### Connection

```
URL:      wss://{HOST}/ws/predict
Timeout:  10 seconds
Ping:     every 30 seconds (keep-alive)
```

### Reconnection Strategy

Exponential backoff with jitter:
```
attempt 1:  1s  + random(0-1s)
attempt 2:  2s  + random(0-1s)
attempt 3:  4s  + random(0-1s)
attempt 4:  8s  + random(0-1s)
...
max attempts: 10
max delay:    ~1024s + jitter
```

After 10 failed attempts, stop reconnecting. Retry on next user keystroke.

### Client → Server Message

```json
{
  "context": "Привет, как ",
  "language": "ru",
  "client_id": "device_uuid_here",
  "n": 3,
  "max_tokens": 20,
  "timestamp": 1711324800000
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `context` | string | yes | Text before cursor (last 200-500 chars is enough) |
| `language` | string | no | "ru" or "en" (auto-detect if omitted) |
| `client_id` | string | no | Persistent device UUID |
| `n` | int | no | Number of suggestions (default: 3) |
| `max_tokens` | int | no | Max tokens per suggestion (default: 20) |
| `timestamp` | int | no | Client timestamp in ms |

### Server → Client Message

```json
{
  "predictions": [
    {"text": "дела?", "confidence": 0.87},
    {"text": "делишки?", "confidence": 0.45},
    {"text": "погода?", "confidence": 0.23}
  ],
  "entropy": 0.342,
  "depth": 3
}
```

| Field | Type | Description |
|-------|------|-------------|
| `predictions` | array | Ordered by confidence (highest first) |
| `predictions[].text` | string | Continuation text |
| `predictions[].confidence` | float | 0.0 - 1.0 |
| `entropy` | float | Model uncertainty (lower = more confident) |
| `depth` | int | Suggested display depth: 1=word, 2=phrase, 3=sentence |

### Debounce

Client SHOULD debounce requests: wait 150ms after last keystroke before sending. Cancel pending request if user types again within 150ms.

```
User types: П → Пр → При → Прив → Приве → Привет
Requests:   ✗    ✗    ✗     ✗      ✗       ✓ (150ms silence)
```

---

## 2. Signal Collection Protocol

Signals = predicted vs. actually typed. Used for model improvement via federated learning.

### Signal Format

```json
{
  "predicted": "дела",
  "typed": "делишки",
  "language": "ru",
  "accepted": false,
  "timestamp": 1711324800.123
}
```

| Field | Type | Description |
|-------|------|-------------|
| `predicted` | string | What the model suggested |
| `typed` | string | What the user actually typed |
| `language` | string | "ru" or "en" |
| `accepted` | bool | true if user tapped the suggestion |
| `timestamp` | float | Unix timestamp with ms precision |

### Enhanced Signals (V2+)

```json
{
  "predicted": "дела",
  "typed": "дела",
  "language": "ru",
  "accepted": true,
  "deleted_after": true,
  "delete_delay_ms": 1200,
  "timestamp": 1711324800.123
}
```

| Field | Type | Description |
|-------|------|-------------|
| `deleted_after` | bool | User accepted then deleted within 5s (golden signal) |
| `delete_delay_ms` | int | Time between accept and delete |

### Batch Upload

```
POST /signals
Content-Type: application/json

{
  "signals": [ ... array of signals ... ],
  "device_id": "uuid",
  "batch_size": 50
}
```

Upload triggers:
- Queue reaches 50 signals, OR
- 5 minutes since last upload, OR
- App going to background

### Offline Queue

If upload fails, persist queue to local storage. Retry on next successful connection. Use atomic writes to prevent corruption if app is killed.

```
Storage: App Group container (iOS) / SharedPreferences (Android)
File:    signal_queue.json
Write:   atomic (write to temp → rename)
Max:     1000 signals (drop oldest if exceeded)
```

---

## 3. Offline T9 Engine

Keyboard MUST work without network. T9 provides basic predictions offline.

### Dictionary Format

```
frequency_indexed dictionary files:
  dict-ru.txt    — Russian words, one per line, sorted by frequency
  dict-en.txt    — English words
  t9-dict-ru.txt — T9 number mappings for Russian
  t9-dict-en.txt — T9 number mappings for English

Format per line:
  word<TAB>frequency

Example:
  привет	98432
  как	87654
  дела	76543
```

### Prefix Index

Build in-memory prefix trie on app start:
```
"пр" → [привет, проект, просто, правильно, ...]
"при" → [привет, принять, пример, ...]
```

Return top 3 by frequency for prefix length >= 2 characters.

### Autocorrect

Edit distance <= 2 with keyboard proximity weighting:

```
Keyboard proximity map (ЙЦУКЕН):
  й: [ц, ф]
  ц: [й, у, ы, ф]
  у: [ц, к, в, ы]
  ...

Proximity substitution cost: 0.5 (vs 1.0 for non-adjacent)
```

---

## 4. Keyboard UI Requirements

### Prediction Strip

```
┌────────────────────────────────────────────┐
│  "дела?"     │  "делишки?"  │  "погода?"   │
└────────────────────────────────────────────┘
                    ↑
            3 prediction slots
            Tap to accept
            Swipe right to see more
```

- Show top 3 predictions
- Tap prediction → insert text + space
- If entropy < 0.5 → show full phrase (depth 3)
- If entropy > 1.0 → show single word (depth 1)
- Empty strip when no predictions / password field

### Anti-Prediction Rules

Do NOT show predictions when:
- Input field is password type
- 3 consecutive predictions rejected (pause for 10 keystrokes)
- User is deleting text (backspace streak)
- Context is less than 2 characters

### Keyboard Features

| Feature | Description |
|---------|-------------|
| **Layouts** | ЙЦУКЕН (RU), QWERTY (EN), Number/Symbol |
| **Shift states** | Off → On (one char) → Locked (all caps) |
| **Backspace** | Tap = delete char, hold = accelerate (150ms → 50ms) |
| **Space** | Single tap = space, double tap = ". " + shift |
| **Swipe on space** | Left/right = move cursor |
| **Long press** | Special chars (Ь → Ъ, е → ё) |
| **Haptic** | Light tap on keypress (configurable) |
| **Themes** | Light / Dark (follow system setting) |
| **Emoji** | Emoji panel toggle button |

### Language Switching

- Globe button to switch RU ↔ EN
- Auto-detect based on first chars typed (if enabled)
- Persist last language in UserDefaults / SharedPreferences

---

## 5. Rating Integration

### Display

Show rating counter in keyboard settings:
```
Settings → GPU Keyboard → ⚡ 847
```

No label, no explanation. Just the number. It grows when device computes for the network.

### Background Compute

When Network Mode is enabled:
1. Device registers with coordinator as a compute node
2. Receives inference requests from other users
3. Processes requests using local track (ONNX model)
4. Returns results to coordinator
5. Earns Rating

See [PROTOCOL.md](./PROTOCOL.md) for full Rating specification.

---

## 6. HTTP API Endpoints

### Health Check
```
GET /health
→ {"status": "ok", "mock": false}
```

### Server Stats
```
GET /stats
→ {
    "learning": {"total_signals": 12345, "batches": 67},
    "connections": 42,
    "model": "pt-moe-v2"
  }
```

### Model Version
```
GET /model/version
→ {"version": "v2.1.0-abc123", "timestamp": 1711324800}
```

Client should check model version periodically and prompt user to update local track if newer version available.

---

## 7. Privacy Requirements

### Data Minimization

- **Context** sent for prediction is transient (not logged on server)
- **Signals** contain predicted vs typed words (not full sentences)
- **Device ID** is random UUID (not hardware ID)
- **No GPS, contacts, photos, or other permissions required**

### Password Fields

- Detect `secureTextEntry` (iOS) / `inputType = TYPE_TEXT_VARIATION_PASSWORD` (Android)
- Disable ALL predictions, signals, and network calls
- Show empty prediction strip

### Opt-in Data Sharing

Three tiers (user chooses in settings):
```
Private (default):  Signals only (predicted vs typed, no context)
Help Train:         Signals + full context tokens (+50% miles bonus)
Maximum:            Signals + context + gradients (+100% miles bonus)
```

---

## 8. Platform-Specific Notes

### iOS

- Keyboard Extension (Custom Keyboard target)
- App Group for shared storage: `group.com.gpu.keyboard`
- `RequestsOpenAccess` = YES in Info.plist (for network)
- Memory limit: ~60MB (Extension can be killed at any time)
- Use `UIInputViewController` as base class

### Android

- `InputMethodService` as base class
- `android:isDefault="false"` in manifest
- Settings activity for enable/configure
- `INTERNET` permission required
- `FOREGROUND_SERVICE` for background compute
- Use `SharedPreferences` for persistence (equivalent of App Group)
- `View.SYSTEM_UI_FLAG_*` for fullscreen keyboard support

### Web (future)

- PWA with `contenteditable` overlay
- WebSocket native browser API
- `IndexedDB` for signal queue persistence
- Service Worker for offline T9

---

## 9. Testing Checklist

Before release, verify:

- [ ] Predictions appear within 200ms of typing pause
- [ ] Offline T9 works with airplane mode
- [ ] Reconnects after network loss (up to 10 attempts)
- [ ] Signals upload in background
- [ ] Signal queue persists across app restarts
- [ ] Password fields show no predictions
- [ ] Anti-prediction activates after 3 rejections
- [ ] Language switch works mid-sentence
- [ ] Backspace acceleration works
- [ ] Double-space → period + space + shift
- [ ] Space swipe moves cursor
- [ ] Memory stays under 60MB (iOS) / 100MB (Android)
- [ ] No crashes when Extension is killed by OS
- [ ] Haptic feedback toggles on/off
- [ ] Dark/Light theme follows system

---

## Quick Start for Android Developers

1. Implement `InputMethodService` with ЙЦУКЕН + QWERTY layouts
2. Add WebSocket client to `wss://{HOST}/ws/predict`
3. Build T9 prefix index from `dict-ru.txt` + `dict-en.txt`
4. Add prediction strip above keyboard (3 slots)
5. Implement `SignalCollector` (queue → batch → POST /signals)
6. Test offline → online transition
7. Add settings: language, theme, network mode, ⚡ counter

**Reference implementation:** `keyboard/ios/` in this repository (Swift)

---

## Contact

- **Email:** kustyuka@gmail.com
- **Telegram:** @yuka_k
- **GitHub:** github.com/yukakust/GPU
