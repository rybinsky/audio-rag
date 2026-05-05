# ТЗ: Audio RAG со стриминговым LLM-ответом на Triton Inference Server

## 0. TL;DR

Делаем сервис "поговори со своей аудио-коллекцией". Пользователь загружает подкасты / аудиокниги / лекции, система их индексирует (ASR + аудио-эмбеддинги + текстовые эмбеддинги в Qdrant). Дальше пользователь задаёт вопрос текстом, бэкенд делает гибридный поиск (текст + звук), реранкит, скармливает топ-K в LLM и стримит обратно текстовый ответ с цитатами (таймстемп + аудио-фрагмент + сниппет транскрипта). Всё работает на CPU - под Mac M1/M2 в первую очередь, на любом x86 CPU с 16GB+ RAM как запасной вариант. Триггерится через Triton Inference Server, который оркестрирует семь моделей через ensemble (для ingest) и BLS (для query). Цель проекта - продемонстрировать понимание Triton (ensemble vs BLS, decoupled mode, dynamic batching, mixed backends) на realistic ML-системе с тремя модальностями (аудио + текст + LLM).

---

## 1. Цель и продукт

**Что строим.** Inference-сервис на Triton, реализующий полноценный Audio RAG. Два публичных эндпоинта по gRPC:

1. **Ingest endpoint** - принимает аудио-файл с метаданными, разбивает на чанки 20-30 сек, прогоняет через VAD, ASR и эмбеддеры, складывает в Qdrant. Возвращает статус.
2. **Query endpoint** - принимает текстовый вопрос, делает гибридный поиск, стримит ответ LLM токен за токеном, вместе с цитатами.

**Use case в README.** Пользователь индексирует пару сезонов Lex Fridman Podcast, спрашивает "что Karpathy говорил про scaling laws". Получает стриминговый ответ "Karpathy утверждал что..." с тремя карточками-цитатами: таймстемп + сниппет транскрипта + кнопка проигрывания соответствующего отрывка.

**Зачем этот проект.** Демонстрация что автор умеет проектировать Triton-сервер для multi-stage ML-системы, а не только обёртку одной модели. Использует обе ключевые формы оркестрации (ensemble и BLS), корректно применяет decoupled mode для стриминга LLM, грамотно распределяет модели по бэкендам, понимает где реально нужен dynamic batching. Закрывает три темы сразу: аудио + RAG + LLM.

---

## 2. Аудитория и как они увидят что проект работает

Реалистично три типа посетителей репы:

**Рекрутер или нанимающий инженер** (90% посетителей). 30-60 секунд внимания. Видит демо-видео сверху, архитектурную диаграмму, таблицу бенчей. Не запускает.

**Любопытный ML-инженер** (8%). Кликает на HuggingFace Space если есть. Запускает локально через `docker compose up` если есть готовый образ.

**Глубокий learner или форкер** (2%). Читает код, конфиги, документацию.

Все три уровня покрываются одинаковой стратегией distribution что и в предыдущей итерации - см. раздел 11.

---

## 3. Стек и модели

### Серверная сторона
- **Triton Inference Server** 25.x, образ `nvcr.io/nvidia/tritonserver:25.xx-py3`, **ARM64 build для Mac** (есть официальные ARM64 builds, либо через Docker linux/arm64 с эмуляцией если что).
- **Бэкенды**: ONNX Runtime (CPU, опционально CoreML execution provider для буста на M1), Python backend (для llama.cpp LLM и BLS-оркестратора).
- **Транспорт**: gRPC для query (с decoupled streaming), gRPC или HTTP для ingest.
- **Внешняя зависимость**: Qdrant как отдельный контейнер.

### Клиент
- **Python 3.11+**, библиотеки: `tritonclient[grpc]`, `qdrant-client`, опционально `sounddevice` для проигрывания цитат.
- **Веб-интерфейс**: Gradio 4.x с чат-окном и плеером для цитат.

### Модели (всё ONNX, CPU-friendly)

| Модель | Роль | Размер | Backend в Triton |
|---|---|---|---|
| **Silero VAD** | Детекция речи в аудио-чанках | ~16 МБ | ONNX Runtime |
| **Whisper-base** (или small для англ) | ASR транскрипция чанков | ~140 МБ | ONNX Runtime |
| **LAION-CLAP** | Аудио и текст в общее пространство для cross-modal поиска | ~200 МБ | ONNX Runtime (две model-папки: clap_audio и clap_text) |
| **BGE-M3** | Мультиязычные текстовые эмбеддинги транскриптов | ~440 МБ | ONNX Runtime |
| **bge-reranker-v2-m3** | Реранкинг top-50 кандидатов | ~560 МБ | ONNX Runtime |
| **Qwen2.5-3B-Instruct** (Q4_K_M) | LLM для синтеза ответа | ~2.0 ГБ | Python backend + llama.cpp |
| **Qdrant** | Векторная БД (отдельный сервис) | - | внешний |

**Итого моделей в RAM при работе:** ~3.5 ГБ + LLM 2 ГБ + накладные = ~7-8 ГБ. На 16 ГБ M1/M2 помещается с запасом.

### Почему такой набор моделей

- **Whisper** - стандарт ASR, ONNX-готовый (через `optimum`), есть варианты по размеру.
- **CLAP вместо чисто текстового поиска** - даёт возможность искать "по звуку" (например, найти моменты со смехом, музыкой, конкретным голосом), плюс текстовый запрос можно прогонять через CLAP-text-encoder и искать в аудио-индексе - это и есть гибридный multimodal поиск.
- **BGE-M3** - сейчас один из лучших мультиязычных эмбеддеров, поддерживает русский, ONNX-friendly.
- **bge-reranker-v2-m3** - тот же дом, тот же мультиязычный охват, реранкер сильно поднимает recall@K.
- **Qwen2.5-3B-Instruct** - хорошо работает на русском и английском, через llama.cpp на M1 даёт 25-40 токенов/сек, что достаточно для стримингового ответа. Альтернатива - Llama-3.2-3B-Instruct (только английский, но чуть быстрее) или Phi-3-mini-4k (только английский, очень быстрый).
- **Qdrant** - простой, надёжный, поддерживает гибридный поиск из коробки.

---

## 4. Архитектура

### 4.1. Ingest pipeline (ensemble, batch processing)

```
audio_file (PCM 16kHz)
    ↓
[chunker: режет на сегменты 20-30 сек с overlap 2сек] (Python)
    ↓ (для каждого чанка)
[Silero VAD] → если речи нет, чанк выбрасывается
    ↓ (для чанков с речью)
ENSEMBLE:
    ┌─────────────────────────────────────────┐
    │  параллельно:                           │
    │  ├── [Whisper] → transcript             │
    │  └── [CLAP audio_encoder] → audio_vec   │
    └─────────────────────────────────────────┘
    ↓
[BGE-M3 на transcript] → text_vec
    ↓
output: {chunk_id, transcript, audio_vec, text_vec, ts_start, ts_end, source_file}
    ↓
запись в Qdrant (две коллекции: text_index и audio_index)
```

**Почему ensemble а не BLS:** ingest-пайплайн это чистый DAG без условной логики. Triton ensemble идеально подходит для статичных DAG, и его проще написать (просто config.pbtxt с описанием графа).

### 4.2. Query pipeline (BLS, online)

```
text_query (от пользователя)
    ↓
BLS query_orchestrator:
    ├─ Шаг 1: параллельно
    │   ├── [BGE-M3] → query_text_vec
    │   └── [CLAP text_encoder] → query_audio_vec
    │
    ├─ Шаг 2: параллельный kNN-поиск в Qdrant
    │   ├── text_index по query_text_vec → top-50 text-кандидатов
    │   └── audio_index по query_audio_vec → top-50 audio-кандидатов
    │
    ├─ Шаг 3: RRF (reciprocal rank fusion) → объединённый top-50
    │
    ├─ Шаг 4: условная логика
    │   - если top-1 score >> top-2 (большой gap) → пропускаем реранкер
    │   - иначе → [bge-reranker-v2-m3] на top-50 → top-K (K=5)
    │
    ├─ Шаг 5: формируем prompt для LLM:
    │   "Контекст: [chunk_1 transcript] ... [chunk_K transcript]
    │    Вопрос: {query}
    │    Ответь на основе контекста, цитируй [номер чанка]."
    │
    ├─ Шаг 6: [LLM stream] (decoupled) → токены стримятся клиенту
    │
    └─ Шаг 7: после окончания LLM-стрима, отправляем структурированные цитаты
        (chunk_id, transcript, ts_start, ts_end, source_file)
```

**Почему BLS а не ensemble:** есть условная логика (skip-rerank), есть стриминг (LLM в decoupled mode), есть параллельные операции с агрегацией. Ensemble этого не умеет.

### 4.3. Какие Triton-фичи задействованы и зачем

- **Ensemble** для ingest - демонстрация классического статичного DAG-пайплайна.
- **BLS (Business Logic Scripting)** для query - демонстрация условной логики и стриминга в оркестрации.
- **Decoupled mode** на LLM и на BLS - без него стриминга токенов нет.
- **Dynamic batching** на эмбеддерах (BGE-M3, CLAP) - реально ускоряет ingest когда чанков много, и query когда concurrent users.
- **Mixed backends** - ONNX Runtime для всего детерминированного, Python backend для LLM (llama.cpp) и для BLS-оркестратора.
- **Внешний сервис** в пайплайне (Qdrant) - демонстрация как Triton интегрируется с продуктовой инфраструктурой.

---

## 5. Раскладка Triton model repository

```
model_repo/
├── chunker/
│   ├── config.pbtxt                # Python backend: режет аудио на сегменты
│   └── 1/model.py
├── silero_vad/
│   ├── config.pbtxt                # ONNX backend
│   └── 1/model.onnx
├── whisper/
│   ├── config.pbtxt                # ONNX backend
│   └── 1/model.onnx
├── clap_audio/
│   ├── config.pbtxt                # ONNX backend, аудио-энкодер
│   └── 1/model.onnx
├── clap_text/
│   ├── config.pbtxt                # ONNX backend, текстовый энкодер CLAP
│   └── 1/model.onnx
├── bge_text_embed/
│   ├── config.pbtxt                # ONNX backend
│   └── 1/model.onnx
├── reranker/
│   ├── config.pbtxt                # ONNX backend
│   └── 1/model.onnx
├── llm/
│   ├── config.pbtxt                # Python backend + llama.cpp
│   └── 1/
│       ├── model.py
│       └── qwen25-3b-instruct-q4_k_m.gguf
├── ingest_ensemble/
│   ├── config.pbtxt                # ensemble config с описанием DAG
│   └── 1/                          # пустая папка (ensemble не имеет своего кода)
├── query_bls/
│   ├── config.pbtxt                # Python backend, главный оркестратор query
│   └── 1/model.py
```

---

## 6. Конфигурации компонентов

### chunker
- Backend: Python.
- Stateless. Принимает аудио-блоб, возвращает массив чанков с таймстемпами.
- max_batch_size: 1 (резка - per-file операция).

### silero_vad
- Backend: ONNX Runtime.
- max_batch_size: 32. dynamic_batching: enabled, max_queue_delay 1мс.
- Модель крошечная, батчинг почти бесплатен.
- instance_group: count=2, kind=CPU.

### whisper
- Backend: ONNX Runtime.
- max_batch_size: 4 (Whisper жадный до памяти).
- dynamic_batching: enabled, max_queue_delay 100мс (можно подождать ради батча, ASR нерeal-time в нашем сценарии).
- instance_group: count=1.

### clap_audio и clap_text
- Backend: ONNX Runtime.
- max_batch_size: 16.
- dynamic_batching: enabled.
- instance_group: count=1 каждый.

### bge_text_embed
- Backend: ONNX Runtime.
- max_batch_size: 32. Эмбеддеры идеально батчатся.
- dynamic_batching: enabled, max_queue_delay 5мс.

### reranker
- Backend: ONNX Runtime.
- max_batch_size: 16. cross-encoder, прогоняется на (query, candidate) парах, поэтому батч это набор кандидатов одного запроса.

### llm
- Backend: Python.
- decoupled: True (обязательно для стриминга токенов).
- max_batch_size: 1 (llama.cpp на CPU не любит батчинг).
- instance_group: count=2, kind=CPU (два параллельных LLM для concurrent users; на M1 16GB это уже впритык по RAM, может потребоваться 1).
- Внутри model.py - инициализация llama.cpp с Qwen2.5-3B GGUF-файлом, метод execute стримит токены через `pb_utils.InferenceResponseSender`.

### ingest_ensemble
- Тип: ensemble.
- Описание DAG в config.pbtxt: chunker → vad → (whisper, clap_audio в параллель) → bge_text_embed.
- Выход: транскрипты + текстовые векторы + аудио-векторы + таймстемпы для всех речевых чанков.

### query_bls
- Backend: Python.
- decoupled: True.
- max_batch_size: 0 (батчинг внутри подмоделей).
- instance_group: count=4, kind=CPU.

---

## 7. Логика BLS-оркестратора (псевдокод query_bls/1/model.py)

```python
async def execute(request):
    query_text = parse_input(request, "QUERY")
    response_sender = request.get_response_sender()

    # Шаг 1: параллельно эмбеддим запрос двумя моделями
    text_emb_future = async_infer("bge_text_embed", query_text)
    clap_emb_future = async_infer("clap_text", query_text)
    query_text_vec = await text_emb_future
    query_audio_vec = await clap_emb_future

    # Шаг 2: параллельный kNN в Qdrant (внешний клиент)
    text_results, audio_results = await asyncio.gather(
        qdrant.search("text_index", query_text_vec, top_k=50),
        qdrant.search("audio_index", query_audio_vec, top_k=50),
    )

    # Шаг 3: RRF fusion
    fused = rrf_fuse(text_results, audio_results, k=60)[:50]

    # Шаг 4: условный реранкер
    if score_gap_large(fused):
        top_k = fused[:5]
    else:
        rerank_inputs = [(query_text, c.transcript) for c in fused]
        rerank_scores = await async_infer("reranker", rerank_inputs)
        top_k = sort_by_score(fused, rerank_scores)[:5]

    # Шаг 5: формируем prompt
    context = format_context(top_k)
    prompt = build_rag_prompt(query_text, context)

    # Шаг 6: стримим LLM-ответ
    llm_stream = async_infer_stream("llm", prompt)
    async for token in llm_stream:
        response_sender.send(
            output_token=token,
            output_type="answer_chunk",
            is_final=False,
        )

    # Шаг 7: отправляем цитаты структурированно
    response_sender.send(
        output_citations=serialize_citations(top_k),
        output_type="citations",
        is_final=False,
    )
    response_sender.send(flags=TRITONSERVER_RESPONSE_COMPLETE_FINAL)
```

---

## 8. Streaming-протокол клиент-сервер

### Ingest (одиночный, не стриминговый)

| Tensor | Dtype | Shape | Описание |
|---|---|---|---|
| `INPUT_AUDIO` | FP32 | `[1, N]` | PCM 16kHz mono |
| `INPUT_SOURCE_ID` | BYTES | `[1, 1]` | Идентификатор источника (имя файла/URL) |
| `INPUT_METADATA_JSON` | BYTES | `[1, 1]` | Произвольные мета (автор, дата, язык) |

Возвращает: статус, число обработанных чанков, ошибки.

### Query (стриминговый, decoupled)

**Запрос:**
| Tensor | Dtype | Shape | Описание |
|---|---|---|---|
| `INPUT_QUERY` | BYTES | `[1, 1]` | Вопрос пользователя |
| `INPUT_TOP_K` | UINT32 | `[1, 1]` | Сколько цитат вернуть, default 5 |
| `INPUT_LANGUAGE` | BYTES | `[1, 1]` | "en"/"ru" - влияет на промпт |

**Ответы (стрим):**
| Tensor | Dtype | Shape | Описание |
|---|---|---|---|
| `OUTPUT_TYPE` | BYTES | `[1, 1]` | "answer_chunk" / "citations" / "error" |
| `OUTPUT_PAYLOAD` | BYTES | `[1, 1]` | для answer_chunk - токен/слово; для citations - JSON со списком цитат |
| `OUTPUT_IS_FINAL` | BOOL | `[1, 1]` | True для последнего сообщения |

Завершается флагом `TRITONSERVER_RESPONSE_COMPLETE_FINAL`.

---

## 9. Клиент

### CLI-клиент

```
audio-rag-cli ingest --file podcast.mp3 --source "Lex #421"
audio-rag-cli ingest --dir ./podcasts/
audio-rag-cli ask "what did Karpathy say about scaling laws?"
audio-rag-cli ask --query "scaling laws" --play-citations
```

Что должен делать:
- `ingest` - читает файл/директорию, шлёт в Triton, показывает прогресс.
- `ask` - открывает gRPC-стрим, печатает ответ token-by-token в терминал, затем выводит цитаты и (если флаг `--play-citations`) проигрывает каждую через sounddevice.

Логирование: TTFT (time-to-first-token), end-to-end latency, tokens/sec, retrieval recall если есть ground truth.

### Gradio веб-интерфейс

- Левая панель: загрузка файлов, список проиндексированных источников.
- Правая панель: чат с историей.
- При получении ответа - стримово отображается текст, под ним появляются карточки-цитаты с кнопкой play, при клике - аудио проигрывается с правильного таймстемпа.

---

## 10. Distribution

### Уровень 0: README

В самом верху:
1. **Демо-видео** 30-60 секунд: индексация одного эпизода Lex Fridman → задаётся вопрос → стримово печатается ответ → карточки цитат → клик на карточку → играет аудио из подкаста с того самого момента.
2. **Архитектурная диаграмма** в mermaid (две: ingest и query отдельно).
3. **Таблица бенчей**: TTFT, end-to-end latency, retrieval recall@10, throughput на ingest.
4. Папка `examples/` с одним мини-датасетом для воспроизведения.

### Уровень 1: HuggingFace Space (CPU-only)

HF Spaces бесплатно даёт CPU-инстансы (16GB RAM, 8 vCPU). Этого хватает для всего нашего пайплайна, потому что нет GPU-зависимых компонентов.

В Space:
- Pre-indexed мини-коллекция (например, 5 эпизодов Lex Fridman, ~20 часов).
- Запрет на ingest пользовательских файлов (или жёсткий лимит) - чтобы не убить Space.
- Только query через Gradio.
- Прямая ссылка из README "Try it live".

### Уровень 2: Docker compose локально

В README ровно две команды:

```bash
git clone <repo> && cd <repo>
docker compose up
```

Сразу поднимает Triton + Qdrant + Gradio фронтенд на `localhost:7860`. Качает пре-собранный образ с моделями включённой через GitHub Container Registry (ghcr.io). Без билда. Без сборки моделей.

### Уровень 3: Полная сборка с нуля

`docs/build_from_source.md` - как скачать веса, экспортнуть всё в ONNX, скачать GGUF для LLM, собрать docker-образ. Для тех кто хочет понять или адаптировать под свои модели.

---

## 11. Hardware compatibility matrix

| Платформа | RAM | Поддерживается | Ingest скорость | Query TTFT | Query tokens/sec |
|---|---|---|---|---|---|
| Mac M1/M2 8GB | 8GB | ⚠️ только query, без LLM или с малым Q | - | - | - |
| Mac M1/M2 16GB | 16GB | ✅ | ~5x realtime | ~600мс | ~25-40 |
| Mac M1 Pro / M2 Pro 16GB+ | 16-32GB | ✅ | ~8x realtime | ~400мс | ~35-55 |
| Mac M3/M4 любая | 16GB+ | ✅ | ~10x realtime | ~350мс | ~40-60 |
| x86 CPU (16+ cores) 32GB | 32GB | ✅ | ~6x realtime | ~500мс | ~30-50 |
| GPU (любая NVIDIA) | 8GB+ VRAM | ✅ + опционально перевод LLM на vLLM | заметно быстрее | ~150мс | 80+ |
| HF Space free CPU | 16GB | ✅ (pre-indexed only) | - | ~800мс | ~20 |

"Realtime" в столбце ingest скорости - сколько часов аудио обрабатывается за один час wallclock.

---

## 12. Бенчмарки

### 12.1. Latency запроса
- TTFT (time-to-first-token): от send запроса до прихода первого токена.
- End-to-end: до прихода последнего токена + цитат.
- p50, p95, p99 на 100 запросах.
- Цель: TTFT p50 <600мс на M1 16GB.

### 12.2. Throughput ingest
- Сколько часов аудио обрабатывается в час wallclock.
- Замерить на 5 разных длительностей (10мин, 30мин, 1ч, 2ч, 4ч).
- Цель: не менее 5x realtime на M1 16GB.

### 12.3. Качество retrieval
- Собрать руками или через LLM 100 пар (query, ground_truth_chunk_id) на маленьком тестовом корпусе.
- Recall@1, Recall@10, MRR.
- Сравнить три режима: только text-search, только audio-search (CLAP), гибрид.
- Показать что гибрид лучше каждого по отдельности.

### 12.4. Качество ответа
- Качественно: 20 примеров вопрос-ответ в README с кратким комментарием.
- Опционально количественно: LLM-as-a-judge на 50 примерах с GPT-4 или Claude как arbiter, метрики faithfulness и answer_relevance (как в Ragas).

### 12.5. Concurrency scaling
- 1, 2, 4, 8 одновременных запросов через perf_analyzer.
- Показать как растёт TTFT.
- Цель: при concurrency=4 TTFT p50 не выше 1с.

### 12.6. Сравнение с baseline
- Тот же пайплайн но без Triton (просто Python-скрипт, всё последовательно).
- Показать выигрыш от батчинга и параллелизма.

### 12.7. Тестовый датасет
- 5 эпизодов Lex Fridman Podcast (~25 часов, в открытом доступе).
- Опционально: 3-5 русскоязычных подкастов для проверки русского.

---

## 13. Структура репозитория

```
.
├── README.md
├── LICENSE
├── .github/workflows/
│   ├── docker-build.yml            # авто-сборка образов на push tag
│   └── tests.yml
├── docker-compose.yml              # локальный dev стек: Triton + Qdrant + frontend
├── Dockerfile.server               # Triton с моделями
├── Dockerfile.frontend             # Gradio
├── docs/
│   ├── architecture.md
│   ├── triton-deep-dive.md         # детальный разбор каждого config.pbtxt и почему так
│   ├── benchmarks.md
│   ├── build_from_source.md
│   └── citations-and-prompt.md     # как форматируем prompt и цитаты
├── model_repo/                     # см. раздел 5
├── scripts/
│   ├── download_models.sh          # все веса с HuggingFace
│   ├── export_to_onnx.py           # whisper, bge, clap, reranker
│   ├── prepare_qdrant.py           # инициализация коллекций
│   └── run_benchmarks.sh
├── client/
│   ├── pyproject.toml
│   ├── audio_rag_client/
│   │   ├── cli.py
│   │   ├── core.py                 # gRPC-логика
│   │   └── audio.py                # проигрывание цитат
├── frontend/
│   ├── app.py                      # Gradio
│   └── requirements.txt
├── benchmarks/
│   ├── benchmark_latency.py
│   ├── benchmark_throughput.py
│   ├── benchmark_retrieval.py
│   └── results/
├── examples/
│   ├── sample_audio/               # один короткий эпизод для воспроизведения
│   └── sample_queries.json
├── hf_space/
│   ├── app.py
│   ├── Dockerfile
│   └── pre_indexed/                # снэпшот Qdrant с проиндексированной мини-коллекцией
└── tests/
    ├── test_components.py
    ├── test_ingest_ensemble.py
    ├── test_query_bls.py
    └── test_e2e.py
```

---

## 14. Пошаговый план

### Неделя 1: модели и эксперименты в чистом Python

- День 1-2: скачать все модели с HuggingFace, прогнать на одной аудио-записи в чистом Python (без Triton). Получить транскрипт, аудио-вектор, текстовый-вектор, прогнать через эмбеддер. Понять что всё работает на M1.
- День 3: ONNX-экспорт моделей (whisper через `optimum.exporters.onnx`, BGE и reranker аналогично, CLAP - через свой скрипт или готовые ONNX из HF). Проверить что ONNX-выходы совпадают с PyTorch (cosine > 0.99).
- День 4: настроить Qdrant локально, написать ingest-скрипт в чистом Python, проиндексировать тестовый эпизод. Убедиться что поиск даёт осмысленные результаты.
- День 5: написать query-скрипт в чистом Python (без Triton): эмбеддинг → kNN → реранкер → llama.cpp с Qwen → ответ. Проверить что end-to-end работает.
- День 6-7: бенчмарки baseline в чистом Python для последующего сравнения. Отладка и улучшение качества промпта.

### Неделя 2: переезд на Triton, ingest pipeline

- День 1: поднять Triton-сервер на M1 в Docker (linux/arm64). Убедиться что простейшая ONNX-модель отвечает.
- День 2: написать config.pbtxt для каждой ONNX-модели. Загрузить, проверить через `tritonclient`.
- День 3: написать chunker как Python backend, написать ensemble config для ingest pipeline.
- День 4-5: end-to-end тест ingest через Triton: подаём аудио, получаем все эмбеддинги и транскрипты, кладём в Qdrant. Сверить что результат идентичен chistому Python-скрипту.
- День 6: тюнинг dynamic batching, прогон ingest на 5 часах аудио, замер throughput.
- День 7: тесты, починка.

### Неделя 3: query pipeline и LLM streaming

- День 1: написать LLM-модель в Python backend с llama.cpp. Сначала non-streaming (вернуть весь ответ сразу). Проверить корректность.
- День 2-3: переключить LLM в decoupled mode со стримингом токенов. Проверить через простой клиент.
- День 4-5: написать BLS-оркестратор query (без условной логики реранкера сначала). End-to-end query через Triton: вопрос → стриминговый ответ.
- День 6: добавить условную логику skip-rerank, добавить отправку цитат после стрима.
- День 7: тесты concurrency, фикс проблем с decoupled.

### Неделя 4: клиент, фронт, бенчи

- День 1-2: CLI-клиент. Команды ingest и ask.
- День 3-4: Gradio-фронтенд. Чат с цитатами и плеером.
- День 5: запись демо-видео.
- День 6: все бенчи из раздела 12, оформление в таблицы и графики.
- День 7: e2e-тесты.

### Неделя 5: distribution и полировка

- День 1-2: Dockerfile для Triton-образа (с моделями внутри), Dockerfile для frontend, docker-compose.
- День 3: GitHub Actions workflow для сборки и пуша в ghcr.io по тегу.
- День 4-5: HuggingFace Space - адаптация под бесплатный CPU-инстанс, pre-indexed snapshot Qdrant.
- День 6: README - архитектура, бенчи, видео, ссылки на Space и Docker, hardware matrix.
- День 7: docs/, финальная вычитка, публикация.

**Итого: 5 недель в плотном режиме.** При 4-часовом дне - 6-7 недель.

---

## 15. Deliverables

К концу проекта в репе должно быть:

1. ✅ Working Triton-сервер, поднимающийся через `docker compose up`.
2. ✅ Pre-built docker image на ghcr.io.
3. ✅ CLI-клиент, установимый через `pip install`.
4. ✅ Gradio веб-интерфейс.
5. ✅ HuggingFace Space с живой демкой и pre-indexed коллекцией.
6. ✅ Демо-видео в README (30-60 сек).
7. ✅ Архитектурные диаграммы (mermaid + PNG).
8. ✅ Полные бенчмарки: retrieval, latency, throughput, concurrency.
9. ✅ Hardware compatibility matrix.
10. ✅ Документация: architecture.md, triton-deep-dive.md, benchmarks.md, build_from_source.md.
11. ✅ Юнит-тесты + e2e-тест.
12. ✅ examples/ с воспроизводимым мини-датасетом.

---

## 16. Критерии приёмки

Проект считается готовым когда:

1. На свежем Mac M1/M2 16GB команда `git clone && docker compose up` поднимает рабочий сервис за <5 минут (без учёта первого pull образа).
2. CLI команда `audio-rag-cli ask "..."` отдаёт стриминговый ответ с TTFT p50 <600мс.
3. Retrieval recall@10 на 100 тестовых запросах >0.7.
4. README содержит все секции из раздела 13: видео, диаграммы, бенчи, quickstart, hardware matrix.
5. HuggingFace Space работает, доступен по ссылке.
6. Code review самим собой: query_bls/1/model.py читается, имеет комментарии, видна структура.
7. Юнит-тесты проходят, GitHub Actions зелёные.

---

## 17. Out of scope

- Дообучение или файнтюн моделей - все веса с HuggingFace.
- Поддержка нескольких пользователей с раздельными коллекциями (multi-tenancy) - все индексируют в общий индекс.
- Аутентификация, биллинг, rate limiting - демонстрационный проект.
- Голосовой ввод запроса (Whisper на запрос) - можно добавить как однодневный бонус, но не основная фича.
- TTS-озвучка ответа - не нужна, ответ текстовый.
- Distributed Qdrant, шардирование, репликация - один инстанс хватает.
- Мобильное приложение / нативный десктоп клиент - только Gradio в браузере.
- Real-time индексация стрима подкаста - только batch ingest файлов.
- Translation между языками запроса и контента - запрос и контент должны быть на одном языке (хотя BGE-M3 кроссязычен и частично работает).

---

## 18. Риски и митигация

### Риск 1: Triton ARM64 на Mac работает с проблемами
**Вероятность**: средняя. **Импакт**: блокер.
**Митигация**: проверить на день 1 первой недели что официальный или сообщественный Triton ARM64 image поднимается на M1/M2. Если нет - использовать через эмуляцию linux/amd64 (медленнее, но работает) или собрать Triton из исходников под ARM64. Fallback: переехать на cloud CPU-инстанс (Hetzner ARM64, AWS Graviton) - дешёво, $5-10/месяц.

### Риск 2: Qwen2.5-3B на M1 слишком медленный
**Вероятность**: низкая (метрики llama.cpp хорошие). **Импакт**: средний.
**Митигация**: даунгрейд на Llama-3.2-1B или Qwen2.5-1.5B. Качество ответов на RAG падает но не критически.

### Риск 3: CLAP cross-modal поиск даёт мусор
**Вероятность**: средняя. CLAP хорошо ищет музыку и звуковые сцены, для произносимой речи он слабее.
**Импакт**: средний - hybrid search не даст ожидаемого буста.
**Митигация**: если CLAP-вклад окажется отрицательным - в финальном пайплайне опционально отключаем audio-side поиск для речевых запросов, оставляем чисто текстовый. В README объясняем когда CLAP помогает (запросы про музыку, эмоциональный тон, звуковые события) и когда нет (фактологические запросы про содержание речи).

### Риск 4: Decoupled streaming через Python backend нестабилен
**Вероятность**: низкая. **Импакт**: средний.
**Митигация**: следовать примерам из официальной документации Triton Python backend, не импровизировать. Тесты на edge cases (клиент отвалился, ошибка LLM посередине) пишутся в первую неделю стриминговой работы.

### Риск 5: HF Space упирается в 16GB RAM
**Вероятность**: средняя.
**Импакт**: HF Space не работает.
**Митигация**: для Space заменить Qwen2.5-3B на Llama-3.2-1B (700MB вместо 2GB), отключить реранкер, ограничить top-K в Qdrant. Качество в Space будет хуже чем в локальной версии, это честно отмечается в README.

### Риск 6: Время разработки превысит 5 недель
**Вероятность**: высокая.
**Импакт**: проект остаётся недоделанным.
**Митигация**: чёткая приоритизация. Если выкидывать - первым Gradio-фронт (CLI достаточно для демо), потом часть бенчей, потом HF Space. README, Docker и видео - неприкасаемое.

---

## 19. Полезные ссылки

- Triton docs - ensemble: https://github.com/triton-inference-server/server/blob/main/docs/user_guide/architecture.md#ensemble-models
- Triton docs - BLS: https://github.com/triton-inference-server/python_backend#business-logic-scripting
- Triton docs - decoupled: https://github.com/triton-inference-server/server/blob/main/docs/user_guide/decoupled_models.md
- Triton ARM64 builds: https://github.com/triton-inference-server/server (releases с тегом arm64)
- Whisper ONNX: https://huggingface.co/onnx-community/whisper-base
- LAION-CLAP: https://github.com/LAION-AI/CLAP
- BGE-M3: https://huggingface.co/BAAI/bge-m3
- bge-reranker-v2-m3: https://huggingface.co/BAAI/bge-reranker-v2-m3
- Qwen2.5-3B GGUF: https://huggingface.co/Qwen/Qwen2.5-3B-Instruct-GGUF
- Qdrant docs: https://qdrant.tech/documentation/
- llama-cpp-python (Python bindings для llama.cpp): https://github.com/abetlen/llama-cpp-python
- Lex Fridman Podcast RSS (для тестового корпуса): https://lexfridman.com/podcast/

---

## 20. Открытые вопросы

Эти решения оставляем на момент когда упрёмся в них:

1. Хранить ли raw audio где-то рядом с Qdrant (для воспроизведения цитат) или достаточно ссылки на исходный файл с offset? По умолчанию - ссылка + offset, raw audio юзер хранит сам.
2. Размер чанка ingest (20с vs 30с vs adaptive по тишинам) - подбираем экспериментально по recall.
3. Окно overlap между чанками (0 vs 2с vs 5с) - влияет на retrieval, экспериментально.
4. Скармливать ли LLM полные транскрипты top-K чанков или их сокращённые версии - решим по реальной длине промпта и качеству ответов.
5. Делать ли свой prompt-template per язык (en/ru) или один универсальный - решим по результатам на смешанных запросах.
6. RRF параметр k (60 - стандартное значение, но может потребоваться тюнинг) - подберём на validation set.
