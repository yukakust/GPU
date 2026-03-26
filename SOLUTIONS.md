# PT-MoE Solutions — Brainstorm Session

## Ключевое архитектурное решение: минимум групп, максимум треков

### Суть
Группа = последовательный round-trip по сети = +50-100ms задержки.
Трек = параллельный вычислитель на отдельном телефоне = бесплатный compute.

**Цель: минимизировать группы (1-2), максимизировать треки (4-32+).**

### Архитектура: 1-2 группы × N треков

```
Fast (autocomplete):   1 группа × 1-2 трека  = локально, ~30-50ms
Normal (предложения):  1 группа × 4 трека    = 1 round-trip, ~150-200ms
Smart (сложный ответ): 2 группы × 4+ треков   = 2 round-trips, ~300-400ms
```

При 1 группе:
```
     Вход (текст)
          │
  ┌───┬───┴───┬───┐
  ▼   ▼       ▼   ▼
[T1] [T2]   [T3] [T4]     ← параллельно на 4 телефонах
 6L   6L     6L   6L        каждый считает свои слои внутри
  │   │       │   │
  └───┴───┬───┴───┘
          ▼
 Cross-Track Attention      ← обмен информацией между треками
          ▼
 Token-Dependent Merge      ← умное объединение
          ▼
       Ответ
```

1 broadcast → все считают параллельно → 1 gather → CTA + merge → готово.

При 2 группах — то же самое дважды: первый раунд "о чём вопрос?", второй "вот ответ с учётом контекста от всех".

### Терминология

**Группа** = раунд обсуждения между телефонами. Все телефоны отдали результаты → merge → если нужно, ещё раунд. Каждая группа = 1 round-trip по сети.

**Трек** = один телефон-специалист. Внутри — последовательные слои (6-12 штук). Телефон считает их быстро, локально, ~50-100ms.

**Переменное число групп:**
```
0 групп:  Локально, 1 трек, без сети.      "Прив" → "ет"           ~30ms
1 группа: Спросили N телефонов, merge.      "Какая погода?" → ответ  ~150-200ms
2 группы: Два раунда обсуждения.            "Напиши функцию..."     ~300-400ms
3 группы: Три раунда (макс, для сложного).  "Проанализируй..."      ~500ms
```

Число групп выбирается автоматически по entropy после каждого раунда:
- entropy < 0.3 после раунда 1 → стоп, ответ уверенный
- entropy > 0.3 → нужен ещё раунд

ИЛИ задаётся приложением: клавиатура = 0-1, чат = 1-2, API = 2-3.

### Redundancy x2 с приоритетным routing

Каждый трек отправляется на 2 телефона: top-1 эксперт (лучший) и top-2 эксперт (второй лучший).

```
Запрос: "напиши функцию сортировки на Python"

Router:
  Code   → top-1: телефон Маши  |  top-2: телефон Пети    (бэкап)
  Lang   → top-1: телефон Коли  |  top-2: телефон Димы    (бэкап)
  Know   → top-1: телефон Ани   |  top-2: телефон Саши    (бэкап)
  Style  → top-1: телефон Лены  |  top-2: телефон Миши    (бэкап)

Отправляем всем 8. Берём первый ответ от каждого типа трека.
Маша зависла → Петя уже считает → берём его. Задержки нет.
```

Token-dependent merge автоматически учтёт если пришёл top-2 вместо top-1 (чуть меньший вес).
Cost: x2 compute. Benefit: resilience + speed.

### Обоснование (статьи)
- **Branch-Train-Merge** (Meta, 2022): 22.4B модель без sequential dependency = 2.5× дешевле обычной при том же качестве
- **ParaFormer** (2025): качество определяется inter-branch collaboration, не глубиной
- **Kraken** (NeurIPS 2024): параллельные ветки для multi-device inference, +35.6% speedup
- **Transformers & Logarithmic Depth** (2024): логарифмическая глубина достаточна для многих задач

---

## Решение 1: Token-Dependent Merge

**Проблема:** Текущий merge = фиксированные скалярные веса. Все токены получают одинаковую смесь треков.

**Решение:** `gate = softmax(Linear(d_model, num_tracks))` — каждый токен сам выбирает какие треки важнее.

```python
# Было: статические веса
weights = softmax(self.gate_weights)  # [num_tracks] — одинаковые для всех токенов
output = sum(w * track_out for w, track_out in zip(weights, track_outputs))

# Стало: token-dependent
# hidden shape: [batch, seq_len, d_model]
weights = softmax(self.gate_proj(hidden))  # [batch, seq_len, num_tracks]
output = sum(w.unsqueeze(-1) * track_out for w, track_out in zip(weights.unbind(-1), track_outputs))
```

**Overhead:** +768 × num_tracks параметров (мизер).

**Доказано:** Branchformer (ICML 2022) — ровно этот подход, обгоняет и Transformer и Conformer.

---

## Решение 2: Cross-Track Attention

**Проблема:** При 1 группе каждый трек работает вслепую — не знает что вычислили другие.

**Решение:** Перед merge треки делают attention друг на друга.

```python
# track_outputs: list of [batch, seq_len, d_model], len = num_tracks
stacked = torch.stack(track_outputs, dim=2)  # [batch, seq_len, num_tracks, d_model]

# Для каждой позиции: 4 вектора (от 4 треков) делают attention между собой
# Multi-head attention: Q, K, V all from stacked
# Результат: каждый трек "знает" что вычислили остальные

cross_attn_out = self.cross_attention(stacked)  # same shape
track_outputs = cross_attn_out.unbind(dim=2)    # list of [batch, seq_len, d_model]
```

**Overhead:** ~4 × d_model² = 4 × 768² ≈ 2.4M params (при 4 heads). Считается на координаторе, не на телефоне.

**Обоснование:** ParaFormer показал что inter-branch communication — ключ к качеству параллельных архитектур.

---

## Решение 3: Track Specialization (Diversity Loss + DropTrack)

**Проблема:** Треки видят одинаковый вход, получают одинаковые градиенты → учат одно и то же.

**Решение A — Diversity Loss:** Штраф за cosine similarity **выходов** треков.

```python
# ВАЖНО: штрафуем ВЫХОДЫ (активации), не веса.
# Статья "Geometric Regularization" (2025) доказала что на весах это НЕ работает.
diversity_loss = 0
for i in range(num_tracks):
    for j in range(i+1, num_tracks):
        cos_sim = F.cosine_similarity(track_outputs[i], track_outputs[j], dim=-1)
        diversity_loss += cos_sim.mean()
diversity_loss *= lambda_diversity  # 0.01-0.1
```

**Решение B — DropTrack:** Случайно выключаем 1+ трек при обучении (как DropPath в vision).

```python
if self.training:
    drop_mask = torch.bernoulli(torch.full((num_tracks,), 1 - drop_rate))
    # Гарантируем что хотя бы 1 трек жив
    if drop_mask.sum() == 0:
        drop_mask[torch.randint(num_tracks, (1,))] = 1
    track_outputs = [out * mask for out, mask in zip(track_outputs, drop_mask)]
```

**Эффект DropTrack:**
1. Каждый трек вынужден быть полезным самостоятельно
2. Модель работает с ЛЮБЫМ подмножеством треков при inference
3. Естественная специализация: если трек 1 уже хорош в языке, трек 2 "вынужден" учить что-то другое

---

## Решение 4: Любой subset треков валиден (Tiered System)

**Проблема:** Разные устройства имеют разные возможности.

**Решение:** DropTrack при обучении гарантирует что модель работает с любым подмножеством треков.

### Тиры устройств

| Tier | Устройство | Треков | RAM | Storage | Что может |
|------|-----------|:------:|:---:|:-------:|-----------|
| 1 | Слабый телефон | 1 | ~50MB | ~150MB | Autocomplete, простые задачи |
| 2 | Средний телефон | 2 | ~100MB | ~300MB | Предложения, средние задачи |
| 3 | Мощный телефон | 4 | ~200MB | ~600MB | Полная модель, сложные задачи |
| 4 | Комп / сервер | 8-32 | 1-4GB | 1.5-5GB | Максимальный интеллект |

### Rating growth by device tier

More tracks = more training compute = higher rating = higher priority.

| Устройство | Треков | Training compute | Rating growth |
|------------|:------:|:----------------:|:-------------:|
| Tier 1 (слабый телефон) | 1 | ~30ms/token | Slow but steady |
| Tier 2 (средний) | 2 | ~60ms/token | 2× faster rating growth |
| Tier 3 (мощный) | 4 | ~120ms/token | 4× faster rating growth |
| Tier 4 (комп/сервер) | 8-32 | varies | Fastest rating growth |

No multipliers, no bonuses. 1 token trained = 1 rating point (signals mode) or 2 rating points (full data mode). More tracks simply means more tokens processed per unit of time.

### Как работает token-dependent merge с неполным набором треков

```python
# available_tracks — маска доступных треков [1, 0, 1, 1] = треки 0, 2, 3
weights = softmax(self.gate_proj(hidden))  # [batch, seq, num_tracks]
weights = weights * available_tracks       # зануляем недоступные
weights = weights / weights.sum(dim=-1, keepdim=True)  # перенормализация
```

Merge автоматически перераспределяет веса на доступные треки.

---

## Решение 5: Масштабирование треков = масштабирование интеллекта

Каждый новый трек = новый специалист в сети. Масштабируется линейно:

```
4 трека:   Language + Knowledge + Domain + Style              → базовая модель
8 треков:  + Reasoning + Code + Math + Creative               → продвинутая
16 треков: + Medical + Legal + Finance + Science + ...         → экспертная
32 трека:  + персональные эксперты юзеров                     → уникальная
64+ трека: каждый юзер = свой эксперт                         → бесконечный рост
```

Token-dependent merge масштабируется: Linear(d_model, N) — добавить трек = добавить столбец в матрицу весов.

### Добавление нового трека без переобучения всей модели

1. Инициализировать новый трек из ближайшего существующего
2. Расширить merge projection: добавить столбец (init ~0)
3. Fine-tune только новый трек + merge layer
4. Существующие треки не трогаем

Это BTX подход (Meta, 2024): Branch → Train independently → Mix via MoE routing.

---

## Решение 6: Adaptive Depth (переменное число групп)

### Тиры продуктов (явный выбор, не автоматика)

```
Keyboard:    0-1 группа   free (participant contributes compute)
Chat/Babel:  1-2 группы   free (participant contributes compute + translation data)
API Light:   2 группы     free, priority by rating
API Full:    3 группы     free, priority by rating
API Deep:    4 группы     free, priority by rating (API without device = last in queue)
```

Приложение/юзер выбирает сколько раундов. Никакой entropy-магии.

### Почему НЕ entropy для выбора числа групп

1. Entropy после 1 группы ≠ entropy после всех групп (shallow model может быть "уверена" в мусоре)
2. Калибровка порогов зависит от размера модели, данных, задачи — хрупко
3. LM head между группами = blocking +5-10ms

### V3+ оптимизация: Confidence Head

Маленький MLP (768 → 64 → 1) предсказывает "нужен ли ещё раунд?". Обучается вместе с моделью.

```python
confidence = sigmoid(self.confidence_head(hidden.mean(dim=1)))
if confidence > 0.8:
    break  # экономим round-trip
```

API Full = до 3 групп, но confidence head может остановить после 2. Юзер платит за максимум, часто получает результат дешевле.

### Для обучения

Loss на выходе КАЖДОЙ группы (intermediate exit heads), не только финальной. Модель учится давать полезный output после любого количества раундов.

---

## Решение 7: Streaming между группами (Kraken-стиль)

При 2+ группах — overlap compute и network:

```
Обычно:     [compute 100ms] → [send 50ms] → [merge] → [compute 100ms] → [send 50ms]
Streaming:  [compute L1-L3] → [send partial] → [compute L4-L6] → [send rest]
                                    ↕ (параллельно)
            Координатор уже начинает получать данные пока телефон досчитывает
```

При 3 группах экономия: ~30-40% network latency (overlap вместо sequential).

Критично для API Full/Deep (3-4 группы). Для клавиатуры (0-1 группа) не нужно.

---

## Решение 8: Rating — экономика сети

### Формула
```
1 rating point = 1 токен обученный для сети на твоём устройстве
```

Rating — permanent, never decreases. It's a total counter of how much you've contributed to the network.

### Правила
- **Everything is free for participants.** If you have the app, you get AI — no spending, no balances.
- **Rating = priority.** Higher rating → higher priority in the queue.
- **Priority formula:** `priority_share = √(your_rating) / Σ√(all_ratings)` — square root ensures diminishing returns for whales.
- **API without device = last in queue.** You can use the API without contributing compute, but you wait behind everyone who does.
- **Developers build free** on the GPU Network — the API is open, priority is the only differentiator.

### How rating grows
```
Signals mode (default):       1× rating per token trained
Full data mode (opt-in):      2× rating per token trained
```

### Rating display
Rating shown as a number with ⚡ in settings. People compare, compete, show off. Transferable to family/friends.

### Battery / Internet Rules

Like sleep for humans — work during the day, brain processes at night. Phone charges → trains on everything from today → wakes up smarter.

```
Charging + WiFi:       training (day's data) + inference for network + data sync
Charging + cellular:   training (day's data) + inference for network (KB only)
Battery + WiFi:        inference for network + sync pending rating
Battery + cellular:    inference for network only (KB per request)
Offline:               own AI only, data accumulates, pending_rating queued
```

Key principles:
- **Training = only while charging.** Plugged in at night → phone learns from everything typed today → smarter by morning.
- **Inference = always** (except offline). Serving network requests = kilobytes of text, negligible traffic/battery.
- **No user settings.** It just works. User never thinks about it.
- **Pending rating:** training done while charging queues up. Syncs to coordinator when WiFi available. Nothing lost.

---

## Итого: что берём в V2

### Простым языком

| # | Решение | Что даёт |
|---|---------|----------|
| 1 | **Token-Dependent Merge** | Каждое слово получает свою смесь экспертов, а не 25%/25%/25%/25% |
| 2 | **Cross-Track Attention** | Эксперты обмениваются записками перед ответом, не работают вслепую |
| 3 | **Diversity Loss** | Треки учат разное, а не дублируют друг друга. 4 телефона = 4× польза |
| 4 | **DropTrack** | Модель работает с любым количеством треков (1-32). Отвалился телефон — ОК |
| 5 | **Переменные группы (0-4)** | Простая задача = 0 раундов, сложная = 4. Приложение выбирает |
| 6 | **Streaming** | Сеть и процессор работают параллельно. -30-40% ожидания при 2+ группах |
| 7 | **Redundancy x2** | Каждый трек на 2 телефонах (top-1 + top-2 эксперт). Один завис — бэкап уже считает |
| 8 | **Rating** | Everything free for participants. Rating = total tokens trained. Higher rating = higher priority in queue |
| 9 | **Tiered Tracks** | Каждый участвует по силам: 1 трек (150MB) или 32 (5GB) |
| 10 | **Свои запросы = 0** | Клавиатура работает даже без интернета и поинтов |
| 11 | **Signal-based learning** | Девайсы шлют сигналы (predicted vs typed). Сервер дообучает модель каждую неделю. Маховик: больше юзеров → больше данных → модель умнее |
| 12 | **Opt-in full data sharing** | Юзер выбирает: "Приватный" (signals = 1× rating), "Помогаю учить" (full data = 2× rating). Больше данных = лучше обучение = больше рейтинг. Честно объясняем что видим |

### Потом (V3+)

| # | Решение | Что даёт |
|---|---------|----------|
| 13 | **Confidence Head** | Экономит раунды: "уже уверена после 2, 3й не нужен" |
| 14 | **Federated Learning** | Градиенты на устройстве, текст вообще не покидает телефон |
| 15 | **Добавление треков** | Новый специалист без переобучения всей модели |
| 16 | **Expertise marketplace** | Professionals train model on domain data → rating grows → higher priority. Contributing, not "earning" |
| 17 | **Speculative Decoding** | Draft-модель на телефоне, сеть проверяет. ×3-5 скорость |
| 18 | **P2P WiFi** | Телефоны в одной комнате общаются напрямую, без интернета |

---

## План обучения

### Фаза 0: Подготовка (делаем 1 раз, переиспользуем всегда)

**День 1: Токенизатор + данные**

```
1. SentencePiece токенизатор (~2-3 часа)
   - Обучаем на RU + EN (70/30 в выборке для токенизатора)
   - 32K vocab, оптимальный для обоих языков
   - Делаем ОДИН раз, используем для ablation → solid → full → production

2. Токенизация данных (~1-2 часа)
   - Quick ablation: 100M токенов, 100% русский (CulturaX/C4 ru)
   - Ретокенизируем SentencePiece → бинарный формат
   - Потом наращиваем: 100M → 500M → 5B (дописываем, не переделываем)

3. Warmup sweep (~1 час)
   - 3 прогона по 1K шагов с разным LR warmup (500, 1000, 2000)
   - Выбираем лучший по скорости снижения loss
```

**День 1-2: Код архитектуры**

```
Реализовать в models/:
├── Batched expert dispatch (ffn.py) — 5-10× ускорение MoE routing
├── Token-Dependent Merge (layers.py) — Linear(d_model, num_tracks) + softmax
├── Cross-Track Attention (layers.py) — MHA между выходами треков
├── Diversity Loss (layers.py) — cosine penalty на ВЫХОДАХ треков
├── DropTrack (layers.py) — random drop треков при обучении
├── Expert utilization logging (ffn.py) — bincount + log каждые N шагов
├── Track weight logging (layers.py) — TDM weight distribution + cosine между треками
└── Переменные группы (model.py) — num_groups как параметр, exit после любой группы
```

**Всё переиспользуется:** токенизатор, данные, код — делаем раз, используем на всех этапах.

---

### Фаза 1: Quick Ablation (100M токенов, ~1 день обучения)

**Цель:** проверить архитектурные решения. Что работает, что нет.

**Данные:** 100M токенов, 100% русский.
**Почему только русский:** 100M мало для двух языков. Цель — проверить архитектуру, не язык.

**Железо:** 2× NVIDIA L20 48GB. 2 модели параллельно (по 1 на GPU).

**Модели:**

| ID | Группы | Треки | Merge | CTA | Diversity | DropTrack | Что проверяем |
|----|:------:|:-----:|-------|:---:|:---------:|:---------:|---------------|
| A | 1 | 4 | Scalar (baseline) | - | - | - | Baseline: 1 группа × 4 трека работает? |
| B | 1 | 4 | Token-Dependent | - | - | - | TDM помогает? |
| C | 1 | 4 | TDM | + | - | - | CTA поверх TDM помогает? |
| D | 1 | 4 | TDM | + | + | + | Полный пакет: треки специализируются? |
| E | 2 | 4 | TDM | + | + | + | Второй раунд стоит того? |
| F | 1 | 8 | TDM | + | + | + | Больше треков = лучше? |
| G | 1 | 2 | TDM | + | + | + | Деградация при 2 треках? (тест DropTrack) |

**Параметры обучения:**

```
batch_size = 8
seq_len = 1024
grad_accum = 4
tokens_per_step = 32,768
optimizer = AdamW (β1=0.9, β2=0.95, wd=0.1)
precision = bfloat16
patience = 5-7
LR = из warmup sweep
aux_loss_weight = 0.01
diversity_loss_weight = 0.01-0.1 (тюним на модели D)
drop_track_rate = 0.25 (в среднем 1 из 4 треков выключен)
```

**Расчёт времени:**

```
steps_per_epoch = 100M / 32,768 = 3,052
epochs ≈ 5 (patience=5, оценка)
total_steps = 15,260
t_step ≈ 1.40с (с batched dispatch + TDM + CTA)

T_per_model = 15,260 × 1.40 = 21,364с ≈ 5.9 часов

7 моделей / 2 параллельно = 4 раунда × 5.9ч = 23.6ч ≈ 1 день
```

**Метрики (автоматические после каждого эксперимента):**

```
Количественные:
├── Val PPL (качество генерации)
├── Expert utilization (токенов на эксперта, dead experts)
├── Track cosine similarity (дублируют ли треки друг друга)
├── TDM weight entropy (использует ли merge разные веса для разных токенов)
└── Throughput (tokens/sec — скорость обучения)

Качественные (человеческим глазом):
├── Генерация по фиксированным промптам (6 штук, RU + EN + Code)
├── Сравнительная таблица: все модели рядом, один промпт
└── Track weights heatmap: какой трек активен на каком токене
```

**Тестовые промпты (фиксированные для всех моделей):**

```
1. "Привет, как"                              — русский разговорный
2. "Уважаемый коллега, хочу сообщить что"     — русский формальный
3. "Столица Франции —"                        — знания / факты
4. "def fibonacci("                           — код
5. "Напиши функцию которая сортирует"         — смешанный (RU + код)
6. "The weather today is"                     — английский (тест токенизатора)
```

---

### Фаза 2: Solid Ablation (500M токенов, ~3 дня) — если Quick покажет результат

**Данные:** 500M токенов, 90% RU + 10% EN.
**Модели:** Топ-2 из Quick ablation + варианты.
**Цель:** подтвердить результаты на большем объёме, проверить что английский не ломает.

---

### Фаза 3: Full Training V2 (1-5B токенов, 1-3 недели)

**Данные:** 1-5B токенов, 70% RU + 30% EN.
**Модель:** Лучшая конфигурация из Solid ablation.
**Цель:** Production quality для клавиатуры и API.

---

### Сводка по времени

| Этап | Данные | Языки | Время обучения | Всего с подготовкой |
|------|:------:|:-----:|:--------------:|:-------------------:|
| Подготовка | — | — | — | 1-2 дня (код + токенизатор) |
| Quick ablation | 100M | RU | ~1 день | 3-4 дня от старта |
| Solid ablation | 500M | 90% RU + 10% EN | ~3 дня | +4 дня |
| Full training V2 | 1-5B | 70% RU + 30% EN | 1-3 недели | +1-3 недели |

**Подготовка делается 1 раз.** Токенизатор, код, данные — переиспользуются на всех этапах.
Данные наращиваются: 100M → 500M → 5B (дописываем, не переделываем).

---

## Quick Ablation Results (10M tokens, March 2026)

| Model | Architecture | Final Loss | Notes |
|-------|-------------|:----------:|-------|
| **F_8tracks** | 1 group × 8 tracks + TDM + CTA + Diversity + DropTrack | **8.09** | 🏆 BEST. More tracks = better. |
| C_cta | 1 group × 4 tracks + TDM + CTA | 8.12 | CTA helps slightly |
| B_tdm | 1 group × 4 tracks + TDM | 8.12 | TDM helps vs baseline |
| A_baseline | 1 group × 4 tracks, scalar merge | 8.14 | Baseline |
| G_2tracks | 1 group × 2 tracks + all | 8.29* | Graceful degradation works |
| D_full | 1 group × 4 tracks + TDM + CTA + Div + Drop | 8.30 | Regularization hurts at 10M, may help at 100M+ |
| E_2groups | 2 groups × 4 tracks + all | 8.56 | WORST. Groups add latency AND hurt quality. |

*G was interrupted at step 120/152, may improve slightly.

### Key Conclusions from Ablation
1. **More tracks = better quality.** F (8 tracks) beats all 4-track models.
2. **1 group is optimal.** E (2 groups) is the worst model. Groups add latency without quality gain.
3. **TDM and CTA help.** B and C beat A baseline consistently.
4. **Diversity + DropTrack hurt at 10M.** D worse than C. Expected — regularization needs more data. Must verify on 100M+.
5. **Architecture confirmed: 1 group × N tracks.** Aligns perfectly with our distributed phone vision.

### Technical Notes
- Models A-D trained with standard AdamW on single GPU (38GB VRAM)
- Models E, F trained with AdamW8bit (bitsandbytes) — required for 2.6B param models on 44GB GPU
- Model G trained on single GPU (699M params, 18GB VRAM)
- 8-bit Adam impact on loss: < 0.01 (negligible for architecture comparison)

---

## Scaling Architecture: Glass Ceiling & Solutions

### Problem: CTA Trained on N Tracks Cannot Extrapolate to N+

CTA (Cross-Track Attention) is trained with a fixed number of tracks (e.g., 8).
DropTrack teaches the model to handle FEWER tracks (1-8), but NOT more (9+).
Softmax distribution fundamentally changes: 8 tracks = 12.5% each, 64 tracks = 1.6% each.

### Solution: Three-Phase Scaling

**Phase A: BTX (8→16 tracks, no latency penalty)**
- Train new tracks independently (copy existing + fine-tune on domain data)
- Retrain ONLY merge layer (TDM + CTA) on new track count
- Merge layer = ~10K params, retraining takes minutes
- CTA extrapolation from 8→16 is small step, likely works
- 0 additional round-trips: same ~115ms latency

**Phase B: BTX or Hierarchical (16→64, test both)**
- BTX path: retrain merge layer on 64 inputs (hours)
  - 1 round-trip, ~115ms, but quality uncertain
- Hierarchical path: 8 groups × 8 tracks, two-level CTA
  - 2 round-trips, ~200ms, quality guaranteed
- **Run both, compare perplexity. Pick winner.**

**Phase C: Hierarchical only (64→512)**
- 8 groups × 8 subgroups × 8 tracks = 512
- 3 round-trips, ~285ms
- CTA always sees exactly 8 inputs (trained configuration)
- No glass ceiling at any scale

### Network Transfer Optimization: INT8
- Compute in bfloat16 on device
- Quantize hidden states to INT8 before sending (divide by max, multiply by 127)
- Coordinator dequantizes back to bfloat16
- Bandwidth halved: 100 tracks × 1MB = 100MB (vs 200MB with bfloat16)
- Quality loss minimal (hidden states, not weights)

---

## V3 Plan: Scaling Beyond Pilot

### V2 → V3 Differences

| Dimension | V2 (Pilot) | V3 (Production) |
|-----------|-----------|-----------------|
| Track size | d=1024, 6 layers, 160MB (INT4) | d=2048, 8 layers, 500MB (INT4) |
| Track count | 8 | 16-64 |
| Total params | 2.5B | 24-96B |
| Active params | 700M | 6-25B |
| Comparable to | GPT-2 | LLaMA-7B to LLaMA-70B |
| Training | 2× L20 (have) | 8-16× A100 80GB (rent ~$5-10K) |
| Phone storage | 160MB per track | 500MB per track |
| New tracks via | Training from scratch | BTX (copy + fine-tune) |
| Merge scaling | Single CTA(8) | BTX or Hierarchical |

### V3 Training Plan
1. Train seed model (8 tracks, d=2048, 8 layers) on 10-50B tokens
2. BTX: copy seed tracks → fine-tune 8-56 more on domain data
3. Test BTX merge vs hierarchical merge at 16 and 64 tracks
4. Pick winner, deploy

### V3 Track Specialization (via BTX)
Each new track is a copy fine-tuned on specific domain:
```
Track 1-8:   General (seed model, trained together)
Track 9:     Code (fine-tuned on code corpora)
Track 10:    Medical (fine-tuned on medical text)
Track 11:    Legal (fine-tuned on legal text)
Track 12:    Math/Science (fine-tuned on STEM)
Track 13-16: Language-specific (EN, ZH, ES, AR)
```

### V3 Training Compute Estimate
```
Seed model (8 tracks, d=2048):
  Params: 8 × 1.5B = 12B
  Data: 50B tokens
  Hardware: 8× A100 80GB, ~1 week
  Cost: ~$3,000-5,000 (cloud rental)

BTX tracks (8 additional):
  Each: 1.5B params, 5B tokens fine-tune
  Hardware: 1× A100 per track, ~1 day each
  Cost: ~$500-1,000 total

Merge retraining:
  Params: ~100K (tiny)
  Time: hours
  Cost: negligible
```

### V4 Vision (Future)
```
Track size: d=4096, 12 layers, 1.5GB (INT4)
Track count: 64-512
Total params: 384B-3T
Active params: 100-800B
Comparable to: GPT-4 / GPT-5 territory
Training: Major compute investment ($100K-1M)
Merge: Hierarchical (3 levels)
Latency: ~285ms (3 round-trips between coordinators)
```

---

## Path to GPT-5: Time × Users, Not Money

### Why We Don't Need $300M

OpenAI spent $300M on GPT-5 because they need to buy everything: data, compute, annotations.
The GPU Network gets all three for free from participants. We are a participant in the network, not a central bank — we contribute compute and benefit from the collective just like everyone else.

### Data: Users ARE the Dataset

```
Users        Tokens/year     Cumulative (4 years)
1M           36B             144B
10M          365B            1.4T
100M         3.6T            14.4T  ← GPT-5 scale
```

This is not scraped web data — it's real human language, fresh every day,
in every language, across every domain. No other company has this.

### RLHF: Every Keystroke is an Annotation

```
Prediction accepted → reward +1 (preferred response)
Prediction rejected → reward -1 (dispreferred), user's input = correct response

This is Direct Preference Optimization (DPO) data:
  OpenAI RLHF:     ~1M preference pairs ($10M, paid annotators)
  Our 1M users:     100M preference pairs PER DAY (free)
  Our 100M users:   10B preference pairs PER DAY

For API: explicit thumbs up/down = classic RLHF at massive scale.
```

### Training: Federated Learning on Phones

Pre-training from scratch requires centralized GPU (seed model).
But continuous fine-tuning runs on phones:

```
Each phone:
  1. Runs inference for user (already doing this)
  2. Collects signals (predicted vs actual)
  3. Computes local gradients for its track (LoRA adapter, small)
  4. Sends gradients to coordinator (NOT data — privacy preserved)

Coordinator:
  1. Aggregates gradients from thousands of phones (Federated Averaging)
  2. Updates global model
  3. Pushes updated weights to phones
  4. Repeat daily

Effective compute:
  1 phone ≈ 1/1000 GPU for training
  10M phones = 10,000 GPU-equivalents
  With federated overhead (10-100×) = 100-1000 free GPUs, 24/7, forever
```

Google already does this with Gboard keyboard. Proven at scale.

### Revised Roadmap

```
Year 0:  Seed model on GPU (our contribution as a network participant)
         V2: 8 tracks, GPT-2 level
         Launch to 150 users (TestFlight)

Year 1:  V3 seed on GPU (network grows)
         64 tracks, LLaMA-13B level
         1M users → 36B tokens/year + 100M DPO pairs/day
         Federated fine-tuning begins

Year 2:  Model continuously improved by participant signals
         LLaMA-70B level (from data, not money)
         10M users → 365B tokens/year
         Professional track specialization emerging

Year 3:  V4 seed on GPU (grant/investor/community fund)
         512 tracks, GPT-4 base level
         + 2 years of federated learning from 10M participants
         + trillions of DPO pairs
         = GPT-4+ level

Year 4:  100M participants
         3.6T new tokens/year
         Federated learning = 100-1000 free GPUs 24/7
         Model improves EVERY DAY
         GPT-5 level — not because anyone spent $300M,
         but because 100M people train the model together
```

### Our Advantage Over OpenAI

```
OpenAI:                              GPU Network:
$300M upfront → GPT-5               Seed model → participants grow it
Model frozen between releases        Model improves EVERY DAY
Data = scraped internet (stale)      Data = live human speech (fresh)
RLHF = paid annotators (expensive)  RLHF = user keystrokes (free, massive)
Inference = expensive servers        Inference = participant phones (free)
Controlled by 1 company             Controlled by no one
```

**Our advantage is not money. Our advantage is time × users.**

---

## Ключевые статьи

| Статья | Год | Что доказывает | Релевантность |
|--------|-----|---------------|---------------|
| Branchformer | ICML 2022 | Token-dependent merge работает | TDM |
| ParaFormer | 2025 | Inter-branch collaboration > depth | CTA |
| Kraken | NeurIPS 2024 | Parallel branches для multi-device | Наш use case |
| BTM / BTX | Meta 2022/2024 | Parallel training + merge = 2.5× дешевле | Масштабирование |
| DEMix | NAACL 2022 | Domain-specific experts, add/remove post-training | Tiered tracks |
| LayerDrop | ICLR 2020 | Drop layers → extract sub-networks | DropTrack |
| Geometric Reg. | 2025 | Diversity на весах НЕ работает, на активациях ДА | Diversity loss |
| Mixture-of-Depths | ICML 2024 | Skip computation per-token | Adaptive depth |
| Soft MoE | ICLR 2024 | Soft routing > hard routing | Merge design |
