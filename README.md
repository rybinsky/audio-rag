# audio-rag

**Audio RAG** — прототип системы для индексации подкастов и ответов на вопросы по аудио с использованием Triton Inference Server и LLM.

## Возможности

- 🎧 Ингест подкастов через Triton (ASR + chunking + индексация)
- 📝 Текстовые и аудио-вопросы через Triton
- 🤖 Генерация ответов с помощью LLM (Qwen2.5-1.5B-Instruct)
- 🔍 Поиск релевантных фрагментов с цитатами
- 🐳 Docker Compose для быстрого запуска
- ⚙️ Конфигурация через Hydra

## Быстрый старт

### 1. Клонирование и установка зависимостей

```bash
git clone <repo-url>
cd audio-rag

# Создать виртуальное окружение
python3 -m venv .venv
source .venv/bin/activate  # Linux/macOS
# или .venv\Scripts\activate на Windows

# Установить зависимости
pip install -e .
```

### 2. Запуск Triton сервера

```bash
# Собрать и запустить Docker контейнер с Triton
docker-compose up -d triton

# Проверить что сервер готов
curl http://localhost:8000/v2/health/ready
```

Ожидаемый ответ: пустой ответ с HTTP 200.

### 3. Проверка статуса моделей

```bash
# Посмотреть логи запуска (загрузка моделей занимает 2-3 минуты)
docker-compose logs -f triton
```

Ожидаемый вывод:
```
"successfully loaded 'asr_whisper'"
"successfully loaded 'ingest_bls'"
"successfully loaded 'query_bls'"
"successfully loaded 'llm_qwen'"
```

## Использование

### Ингест подкаста

Загрузить аудиофайл подкаста в индекс:

```bash
# Через Docker (файл должен быть доступен в /workspace)
docker-compose exec triton python3 main.py triton-ingest-podcast \
  --source my-podcast \
  --audio-file /workspace/tests/Подкаст.mp3

# Или локально (файл на хост-машине)
python3 main.py triton-ingest-podcast \
  --source my-podcast \
  --audio-file ./tests/Подкаст.mp3
```

Параметры:
- `--source` — идентификатор источника (используется для цитат)
- `--audio-file` — путь к аудиофайлу (MP3, M4A, WAV, OGG, AAC)
- `--transcript-file` — (опционально) готовый транскрипт

### Задать вопрос (текст)

```bash
docker-compose exec triton python3 main.py triton-ask "О чём говорилось в подкасте?"
```

### Задать вопрос (аудио)

```bash
docker-compose exec triton python3 main.py triton-ask-audio \
  --question-audio-file /workspace/tests/Вопрос.m4a
```

Параметры:
- `--top-k` — количество релевантных чанков (по умолчанию 5)

### Пример ответа с LLM

```
Resolved question transcript: О чём говорилось в подкасте?

В подкасте обсуждали курс доллара. На 5 мая 2022 года курс доллара 
составлял 75 рублей и 50 копеек к российскому рублю.

Citations:
- my-podcast [0:18]: На сегодняшний день 5 мая 2022 года курс доллара...
```

## Структура проекта

```
.
├── audio_rag/              # Основной код
│   ├── __init__.py
│   ├── chunking.py         # Нарезка текста на чанки
│   ├── cli.py              # CLI команды
│   ├── config.py           # Загрузка конфигурации (Hydra)
│   ├── embeddings.py       # Эмбеддинги (хеширование)
│   ├── models.py           # Pydantic модели данных
│   ├── service.py          # Бизнес-логика
│   ├── settings.py         # Настройки приложения
│   ├── store.py            # JSONL хранилище чанков
│   └── triton_client.py    # Клиент для Triton
├── conf/                   # Конфигурационные файлы Hydra
│   └── config.yaml
├── model_repo/             # Triton model repository
│   ├── asr_whisper/        # ASR модель (faster-whisper)
│   ├── ingest_bls/         # Ingest pipeline (Python backend)
│   ├── query_bls/          # Query pipeline (Python backend)
│   └── llm_qwen/           # LLM модель (Qwen2.5-1.5B-Instruct)
├── tests/                  # Тестовые аудиофайлы
├── tmp/                    # Временные файлы (хранилище чанков)
├── sitecustomize.py        # Исправление sys.path для Triton
├── Dockerfile.triton       # Docker образ Triton
├── docker-compose.yml      # Docker Compose конфигурация
├── requirements-triton.txt # Зависимости для Triton
├── main.py                 # Точка входа CLI
└── README.md
```

## Конфигурация

Конфигурация управляется через Hydra. Основной файл: `conf/config.yaml`.

### Переменные окружения

| Переменная | Описание | По умолчанию |
|------------|----------|--------------|
| `AUDIO_RAG_STORE_PATH` | Путь к файлу хранилища чанков | `./tmp/chunks.jsonl` |
| `AUDIO_RAG_ASR_MODEL_SIZE` | Размер модели Whisper | `tiny` |
| `AUDIO_RAG_ASR_COMPUTE_TYPE` | Тип вычислений ASR | `int8` |
| `AUDIO_RAG_ASR_DEVICE` | Устройство для ASR | `cpu` |
| `AUDIO_RAG_ASR_VAD_FILTER` | Включить VAD фильтрацию | `true` |
| `AUDIO_RAG_USE_LLM` | Использовать LLM для ответов | `true` |
| `AUDIO_RAG_LLM_MODEL` | Модель LLM | `Qwen/Qwen2.5-1.5B-Instruct` |
| `AUDIO_RAG_LLM_DEVICE` | Устройство для LLM | `cpu` |
| `AUDIO_RAG_LLM_MAX_TOKENS` | Максимум токенов в ответе | `512` |

### Переопределение через CLI

```bash
python3 main.py triton-ask "Вопрос" \
  retrieval.default_top_k=10
```

## Архитектура

### Компоненты системы

```
┌─────────────────────────────────────────────────────────────────┐
│                      Triton Server                               │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌───────────┐ │
│  │ asr_whisper │ │ ingest_bls  │ │  query_bls  │ │ llm_qwen  │ │
│  │ (ASR)       │ │ (Python)    │ │ (Python)    │ │ (LLM)     │ │
│  └─────────────┘ └─────────────┘ └─────────────┘ └───────────┘ │
└─────────────────────────────────────────────────────────────────┘
         │                │                │              │
         ▼                ▼                ▼              ▼
┌─────────────────────────────────────────────────────────────────┐
│            JSONL Chunk Store (tmp/chunks.jsonl)                  │
└─────────────────────────────────────────────────────────────────┘
```

### Ingest Pipeline

```
Аудиофайл → ASR (Whisper) → Транскрипт → Chunking → Индексация
```

### Query Pipeline

```
Вопрос (текст/аудио) 
    → ASR (если аудио) 
    → Retrieval (поиск релевантных чанков) 
    → LLM (генерация ответа) 
    → Ответ с цитатами
```

### Модели Triton

| Модель | Тип | Назначение |
|--------|-----|------------|
| `asr_whisper` | Python backend | Распознавание речи (faster-whisper) |
| `ingest_bls` | Python backend | Индексация подкастов |
| `query_bls` | Python backend | Обработка запросов, retrieval |
| `llm_qwen` | Python backend | Генерация ответов (Qwen2.5-1.5B) |

## Текущие ограничения

Проект является **прототипом** и содержит упрощения:

| Компонент | Текущее решение | План по ТЗ |
|-----------|-----------------|------------|
| ASR | faster-whisper (Python backend) | ONNX Whisper |
| Эмбеддинги | Хеширование текста | BGE-M3 + CLAP |
| Хранилище | JSONL файл | Qdrant |
| Retrieval | Простой поиск по хешу | Vector search + Reranker |
| LLM | ✅ Qwen2.5-1.5B-Instruct | Qwen с стримингом |
| Orchestration | Python backend | Ensemble + BLS |

## Разработка

### Запуск тестов

```bash
pip install -e ".[dev]"
pytest tests/
```

### Линтинг

```bash
ruff check audio_rag/
mypy audio_rag/
```

### Сборка Docker образа

```bash
docker-compose build --no-cache triton
```

### Очистка

```bash
# Остановить контейнеры и удалить volumes
docker-compose down -v

# Очистить __pycache__
find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null

# Очистить хранилище чанков
rm -f tmp/chunks.jsonl
```

## Устранение неполадок

### Ошибка: ModuleNotFoundError: No module named 'packaging'

Эта ошибка связана с конфликтом DALI backend в Triton. Решение уже включено в проект через `sitecustomize.py`.

Если проблема возникает, убедитесь что:
1. `sitecustomize.py` существует в корне проекта
2. Dockerfile.triton копирует его в `/usr/local/lib/python3.12/dist-packages/`

### Ошибка: EOFError: EOF read where object expected

Повреждены `.pyc` файлы. Очистите кэш:

```bash
find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null
find . -type f -name "*.pyc" -delete 2>/dev/null
docker-compose build --no-cache triton
```

### LLM модель не загружается

Проверьте логи:

```bash
docker-compose logs triton | grep -E "(llm_qwen|failed|error)" -i
```

Возможные причины:
1. Недостаточно памяти (требуется ~4GB RAM для Qwen2.5-1.5B)
2. Превышен timeout загрузки модели

Временное решение — отключить LLM:
```yaml
# docker-compose.yml
AUDIO_RAG_USE_LLM: "false"
```

### Ответы шаблонные вместо LLM

Проверьте:
1. Загружена ли модель: `curl http://localhost:8000/v2/models/llm_qwen`
2. Включён ли LLM: `docker-compose exec triton env | grep AUDIO_RAG_USE_LLM`

## Системные требования

- Docker Desktop 4.0+
- Docker Compose 2.0+
- 8GB RAM минимум (для LLM)
- 10GB свободного места на диске (для моделей)

## План развития

См. `TZ_audio_rag.md` для полного технического задания.

### Ближайшие шаги

1. ✅ **LLM**: добавлен Qwen2.5-1.5B для генерации ответов
2. **Эмбеддинги**: заменить хеширование на BGE-M3 для текста и CLAP для аудио
3. **Хранилище**: миграция с JSONL на Qdrant для векторного поиска
4. **Стриминг**: добавить стриминг токенов от LLM
5. **ONNX**: миграция моделей в ONNX формат для оптимизации

## Лицензия

MIT