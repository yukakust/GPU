# PT-MoE Research — Problems & Bottlenecks

## 1. АРХИТЕКТУРА: РАСПРЕДЕЛЁННАЯ МОДЕЛЬ НА ТЕЛЕФОНАХ

### Корневая проблема: PT-MoE проиграл dense в эксперименте

Вся архитектура строится на гипотезе "параллельные треки = распределяемость без потери качества". Эксперимент показал обратное: чем больше треков — тем хуже PPL. Это нужно починить **до** V2, иначе фундамент шаткий.

**Почему PT-MoE деградирует (вероятные причины):**

**A. Merge Layer не учит ничего полезного.** 4-12 скалярных весов (softmax) = статическое взвешенное среднее. Модель не может сказать "для этого токена важен трек 2, а для того — трек 4". Все треки учат одно и то же → дублирование вместо специализации.

- **Фикс:** Token-dependent merge: `gate = softmax(Linear(d_model, num_tracks))` — каждый токен получает свой вектор весов. Это +768×num_tracks параметров (мизер), но даёт треку причину специализироваться. Можно пойти дальше: cross-track attention на merge boundary (дорого, но максимально мощно).

**B. Нет механизма специализации треков.** Aux loss балансирует экспертов **внутри** MoE-слоя, но ничего не заставляет **треки** быть разными. Трек 1 и трек 4 видят одинаковый input, получают одинаковые градиенты через merge → сходятся к одному решению.

- **Фикс:** Track diversity loss — штраф за cosine similarity между выходами треков. Или: dropout целых треков при обучении (как DropPath в vision) — заставляет каждый трек быть самодостаточным. Или: разные данные/задачи на разные треки (один на RU, другой на EN, третий на code).

**C. Shallow-wide проигрывает deep-narrow.** 2 группы × 12 треков = 2 последовательных шага трансформации. Трансформер с 2 слоями принципиально слабее 24-слойного — никакое количество параллельных треков это не компенсирует. Глубина = абстракция и рассуждение.

- **Фикс:** Не уходить ниже 6 групп. Оптимальная точка скорее всего 6-8 групп × 2-4 трека. Это всё ещё распределяемо (4 телефона), но сохраняет достаточную глубину. Конфигурация 2×12 и 3×8 — тупиковые.

---

### Проблема координатора

GPU_PLAN Phase 2-4 предполагает что координатор "лёгкий". На самом деле:

| Операция | Размер при V2 (d=1024) | Размер при V4 (d=4096) |
|----------|----------------------|----------------------|
| Embedding | 50K × 1024 = 50M params | 50K × 4096 = 200M params |
| LM Head | 50K × 1024 = 50M params | 50K × 4096 = 200M params |
| Merge (на группу) | мизер | мизер |
| **Итого на координаторе** | **~100M params** | **~400M params** |

Координатор выполняет **треть всех вычислений** и работает на numpy/CPU. Это потолок throughput.

- **Фикс (V2):** Перевести координатор на PyTorch + GPU (хотя бы серверный). Или: embedding/LM head тоже распределить — один телефон считает embedding, другой LM head.
- **Фикс (V4):** Если координатор = устройство юзера, embedding + LM head должны быть квантизированы и запускаться локально на телефоне юзера. Это ~100-200MB INT4 — влезает.

---

### Network protocol

**Float32 по сети — 2× лишнего.** Тензор `1 × seq_len × d_model` при каждом round-trip:
- V2 (d=1024): 1 × 512 × 1024 × 4 = **2MB** per call, float16 = 1MB
- V4 (d=4096): 1 × 512 × 4096 × 4 = **8MB** per call, float16 = 4MB

При 6 групп × 2 направления = 12 передач на токен. V4: **96MB на токен** при float32.

- **Фикс:** Float16 минимум. Лучше: INT8 квантизация hidden states. Ещё лучше: delta encoding.

**Нет валидации ответов от телефонов.** NaN/wrong shape = тихая мусорная генерация.

- **Фикс:** `assert tensor.shape == expected`, `assert isfinite(tensor).all()`, timeout + fallback.

**Нет redundancy.** Самый медленный телефон = скорость группы.

- **Фикс:** Отправлять задачу на N+1 телефонов, брать первые N ответов.

---

### Что реально прокачает распределённость

- **Speculative decoding** — маленькая модель на телефоне юзера генерирует 4-8 draft токенов, сеть проверяет за 1 forward pass. ×3-5 по скорости.
- **Pipeline parallelism** — группа 1 считает токен N, группа 2 одновременно токен N-1. Throughput ×num_groups.
- **P2P WiFi** — MultipeerConnectivity iOS, latency <1ms. 5 телефонов в комнате = полная модель локально.

---

## 2. ОБУЧЕНИЕ: ЧТО ДЕЛАЕМ НЕ ТАК

### Batched expert dispatch — bottleneck #1

`models/ffn.py` — Python for-loop по каждому токену × каждому эксперту. Основная причина почему MoE в 1.7× медленнее dense (5368s vs 3120s).

```python
# СЕЙЧАС: O(batch × seq_len × top_k) Python calls
for i in range(batch * seq_len):
    for j, expert_idx in enumerate(top_indices[i]):
        expert_out = self.experts[expert_idx](token)

# НУЖНО: O(num_experts) batched calls
for expert_idx in range(num_experts):
    mask = (top_indices == expert_idx).any(dim=-1)
    tokens_for_expert = x[mask]
    out[mask] = expert(tokens_for_expert)
```

Разница: **5-10×** на реальных размерах.

### 10M токенов для MoE = недостаточно

MoE требует значительно больше данных чем dense. 10M = 500-2000× меньше нужного. Отрицательный результат может быть целиком из-за недостатка данных.

- **Фикс:** Минимум 100M для ablation, 1B+ для реальных выводов.

### Expert utilization — летим вслепую

Нет логирования: сколько токенов получает каждый эксперт, dead experts, router collapse.

- **Фикс:** 5 строк кода: `expert_counts = torch.bincount(top_indices.flatten(), minlength=num_experts)`

### Early stopping patience=3

MoE PPL нестабильна. Patience=3 может убить обучение при временном spike.

- **Фикс:** Patience=5-7 или EMA smoothing на val_ppl.

### LR warmup не тюнится

Фиксированный `min(2000, max_steps // 10)`.

- **Фикс:** 2-3 коротких прогона с разным warmup.

### Нет ablation study

Один эксперимент, один датасет, один размер. Невозможно понять что вредит.

- **Фикс:** Минимальный ablation:
  1. token-dependent merge vs scalar merge
  2. track diversity loss vs без
  3. batched dispatch vs loop
  4. 10M vs 100M vs 500M токенов

---

## 3. ВСЁ ОСТАЛЬНОЕ

### Безопасность и инфраструктура

| Проблема | Импакт | Фикс |
|----------|--------|------|
| Credentials в публичном репо (IP, SSH paths) | Любой может найти сервера | `.gitignore` + rotate + убрать из git history |
| WebSocket без auth | Мусорные тензоры, DoS | API key в header при handshake |
| Нет rate limiting | Один клиент забивает сервер | `slowapi` или nginx rate limit |
| requirements.txt без pinning | Ломается при обновлении | `pip freeze > requirements.lock` |
| Нет тестов, нет CI | Рефакторинг = рулетка | Smoke test модели + contract test WebSocket |

### iOS клавиатура

| Проблема | Импакт | Фикс |
|----------|--------|------|
| KeyboardViewController 1800 строк | Невозможно поддерживать | Разбить: InputHandler, LayoutManager, PredictionManager |
| Нет client-side debounce | 5-10 WS запросов/сек | `DispatchWorkItem` с 150ms delay |
| SignalCollector без atomic write | Extension убивается → corrupt | Temp file → atomic rename или SQLite |
| AutocorrectEngine O(n) fuzzy | Тормоза на 100K+ словаре | SymSpell — O(1) lookup |
| RealPredictor greedy only | Repetitive предсказания | Подключить TOP_K/TEMPERATURE из config.py |

### Privacy

SignalCollector шлёт `context_ids` — это закодированный текст. Сервер восстановит всё.

- **V2:** Шифровать context_ids, сервер получает только `(predicted_id, actual_id, match: bool)`.
- **V3+:** Градиенты на устройстве, context не шлётся.

---

## МАППИНГ: ЧТО В ПЛАНЕ, ЧТО НЕТ

### Планируем решить (есть в GPU_PLAN / GPU_EVOLUTION)

| Проблема | Где в плане |
|----------|-------------|
| GPT-2 токенизатор | V2 → SentencePiece 32K |
| ONNX без квантизации | INT4 Quantization (Stage 2) |
| KV-cache | KV-Cache Sharing (Stage 2), но только sharing, не базовый |
| ComputeWorker = заглушка | Phase 2: ONNX Runtime Web |
| Signals не используются | Phase 3: federated learning |
| Нет offline predictions | V3: on-device inference |
| Network latency | Pipeline Parallelism + P2P + Speculative |
| Синхронный барьер | Pipeline Parallelism (Stage 2) |

### НЕ планируем решить (пропущено)

| Проблема | Критичность |
|----------|-------------|
| Batched expert dispatch | ВЫСОКАЯ — 5-10× ускорение обучения |
| Token-dependent merge | ВЫСОКАЯ — корень деградации PT-MoE |
| Отрицательные результаты не проанализированы | ВЫСОКАЯ — фундамент |
| Expert utilization logging | ВЫСОКАЯ — 5 строк кода, диагностика |
| Координатор на numpy = bottleneck | СРЕДНЯЯ — потолок throughput |
| Track diversity / специализация | ВЫСОКАЯ — треки учат одно и то же |
| WebSocket без auth | СРЕДНЯЯ — security |
| Binary protocol без валидации | СРЕДНЯЯ — тихие ошибки |
| Float32 по сети | СРЕДНЯЯ — 2× лишних данных |
| KeyboardViewController God Object | СРЕДНЯЯ — maintainability |
| SignalCollector без atomic write | СРЕДНЯЯ — потеря данных |
| RealPredictor greedy only | НИЗКАЯ — UX |
| Нет тестов / CI | СРЕДНЯЯ — quality |
| Credentials в публичном репо | ВЫСОКАЯ — security |
| Privacy: context_ids = текст | ВЫСОКАЯ — reputation risk |
