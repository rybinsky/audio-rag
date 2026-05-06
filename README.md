# Audio RAG

**Audio RAG** — A prototype system for podcast indexing and audio-based Q&A using Triton Inference Server and LLM.

[README на русском](#audio-rag-на-русском) | [English](#audio-rag)

---

## Features

- 🎧 Podcast ingestion via Triton (ASR + chunking + indexing)
- 📝 Text and audio questions via Triton
- 🤖 Answer generation with LLM (Qwen2.5-0.5B-Instruct)
- 🔍 Relevant fragment search with citations
- 🐳 Docker Compose for quick deployment
- ⚙️ Hydra-based configuration
- 🗄️ Qdrant vector database for efficient similarity search
- 🎯 BGE-M3 embeddings for multilingual support

## Quick Start

### 1. Clone and Install Dependencies

```bash
git clone <repo-url>
cd audio-rag

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate  # Linux/macOS
# or .venv\Scripts\activate on Windows

# Install dependencies
pip install -e .
```

### 2. Configure Environment Variables

```bash
# Copy example environment file
cp .env.example .env

# Edit .env with your settings (optional, defaults work for local development)
```

### 3. Start Triton Server

```bash
# Build and start Docker containers
docker-compose up -d

# Check if server is ready
curl http://localhost:8000/v2/health/ready
```

Expected response: empty response with HTTP 200.

### 4. Verify Model Status

```bash
# View startup logs (model loading takes 2-3 minutes)
docker-compose logs -f triton
```

Expected output:
```
"successfully loaded 'asr_whisper'"
"successfully loaded 'bge_embedder'"
"successfully loaded 'reranker'"
"successfully loaded 'ingest_bls'"
"successfully loaded 'query_bls'"
"successfully loaded 'llm_qwen'"
```

## Usage

### Ingest a Podcast

```bash
python main.py triton-ingest-podcast \
  --source my-podcast \
  --audio-file ./path/to/podcast.mp3
```

Expected output:
```json
{
  "status": "success",
  "chunks_count": 15,
  "source_id": "my-podcast"
}
```

### Ask a Question (Text)

```bash
python main.py triton-ask "What was discussed in the podcast?"
```

### Ask a Question (Audio)

```bash
python main.py triton-ask-audio --question-audio-file ./question.mp3
```

### Example Answer with LLM

```
Resolved question transcript: What was discussed in the podcast?

Based on the transcript, the podcast discussed the current exchange rate
of the dollar to the Russian ruble as of May 5, 2022...

Citations:
- my-podcast [0:18] score=0.847: На сегодняшний день 5 мая 2022 года...
```

## Project Structure

```
audio-rag/
├── audio_rag/           # Main package
│   ├── embedders/       # BGE-M3 and Triton embedders
│   ├── stores/          # Qdrant and JSONL stores
│   ├── service.py       # Core business logic
│   ├── cli.py          # Command-line interface
│   └── config.py       # Configuration loading
├── model_repo/         # Triton model repository
│   ├── asr_whisper/    # Whisper ASR model
│   ├── bge_embedder/   # BGE-M3 embeddings
│   ├── reranker/       # BGE reranker
│   ├── ingest_bls/     # Ingestion pipeline
│   ├── query_bls/      # Query pipeline
│   └── llm_qwen/       # LLM for answers
├── conf/               # Hydra configuration
├── tests/              # Test suite
├── docker-compose.yml  # Docker services
├── Dockerfile.triton   # Triton container
└── .env.example        # Environment variables template
```

## Configuration

Configuration is managed through Hydra. Main file: `conf/config.yaml`.

### Environment Variables

Key environment variables (see `.env.example` for full list):

| Variable | Description | Default |
|----------|-------------|---------|
| `QDRANT_HOST` | Qdrant server hostname | `localhost` |
| `QDRANT_PORT` | Qdrant server port | `6333` |
| `TRITON_SERVER` | Set to "true" inside Triton | `false` |
| `AUDIO_RAG_ASR_MODEL_SIZE` | Whisper model size | `tiny` |
| `AUDIO_RAG_LLM_MODEL` | LLM model name | `Qwen/Qwen2.5-0.5B-Instruct` |

**Important:** When running inside Docker Compose, set `QDRANT_HOST=qdrant` (the service name).

### Docker Compose Configuration

The `docker-compose.yml` automatically sets the correct environment variables:
- Triton container uses `QDRANT_HOST=qdrant` to connect to Qdrant service
- Local client uses `QDRANT_HOST=localhost` (default)

## Architecture

### System Components

1. **ASR (Whisper)** - Speech-to-text for audio files
2. **Embedder (BGE-M3)** - Text embeddings for semantic search
3. **Vector Store (Qdrant)** - Efficient similarity search
4. **Reranker (BGE-Reranker)** - Improved search relevance
5. **LLM (Qwen)** - Natural language answer generation

### Ingest Pipeline

```
Audio File → ASR (Whisper) → Transcript → Chunking → Embedding (BGE-M3) → Qdrant
```

### Query Pipeline

```
Question → Embedding → Qdrant Search → Reranking → LLM Generation → Answer
```

### Triton Models

| Model | Purpose | Hardware |
|-------|---------|----------|
| `asr_whisper` | Speech recognition | CPU/GPU |
| `bge_embedder` | Text embeddings | CPU/GPU |
| `reranker` | Search reranking | CPU/GPU |
| `ingest_bls` | Ingestion orchestration | CPU |
| `query_bls` | Query orchestration | CPU |
| `llm_qwen` | Answer generation | CPU/GPU |

## Development

### Running Tests

```bash
# Run all tests
pytest tests/

# Run specific test
pytest tests/test_mvp.py -v
```

**Note:** Tests require Qdrant to be running (`docker-compose up -d qdrant`).

### Code Quality

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run linting
flake8 audio_rag/ tests/
mypy audio_rag/
```

### Building Docker Image

```bash
docker-compose build triton
```

### Clean Up

```bash
# Stop all services
docker-compose down

# Remove volumes (clears all data)
docker-compose down -v

# Remove orphaned images
docker image prune
```

## Troubleshooting

### Error: Connection refused to Qdrant

**Symptom:** `ConnectionRefusedError: [Errno 111] Connection refused`

**Solution:** 
1. Ensure Qdrant is running: `docker-compose ps`
2. Check environment variables: `QDRANT_HOST` should be `qdrant` in Docker, `localhost` locally
3. Verify Qdrant health: `curl http://localhost:6333/collections`

### Error: QdrantClient has no attribute 'search'

**Symptom:** `AttributeError: 'QdrantClient' object has no attribute 'search'`

**Solution:** This is fixed in the current version. The code now uses `query_points()` API compatible with qdrant-client 1.16+.

### Error: ModuleNotFoundError: No module named 'packaging'

**Symptom:** Error when loading models in Triton.

**Solution:** This is fixed in the current Dockerfile via sitecustomize.py.

### LLM Model Not Loading

**Symptom:** Out of memory or slow startup.

**Solution:**
1. Use smaller model: `AUDIO_RAG_LLM_MODEL=Qwen/Qwen2.5-0.5B-Instruct`
2. Disable LLM: `AUDIO_RAG_USE_LLM=false`

### Receiving Generic Answers Instead of LLM

**Symptom:** Answers are template-based, not generated by LLM.

**Solution:**
1. Verify LLM is enabled: `AUDIO_RAG_USE_LLM=true`
2. Check LLM logs: `docker-compose logs triton | grep llm_qwen`

## System Requirements

- **Python:** 3.9+
- **Docker:** 20.10+
- **Docker Compose:** 2.0+
- **RAM:** 4GB minimum, 8GB recommended
- **Disk:** 10GB for models and data

## Recent Updates

### 2025-01-06
- ✅ Fixed Qdrant connection with environment variables in config.yaml
- ✅ Updated to qdrant-client 1.16+ API (query_points)
- ✅ Implemented query_bls model execute method
- ✅ Added .env.example with comprehensive documentation
- ✅ Removed unused variables from tests
- ✅ Added test audio files to .gitignore

## Roadmap

- [ ] Telegram bot integration
- [ ] Web UI
- [ ] Multi-language support
- [ ] Batch ingestion
- [ ] Custom embedding models
- [ ] RAG evaluation metrics

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests: `pytest tests/`
5. Submit a pull request

## License

MIT License - see [LICENSE](LICENSE) for details.

---

# Audio RAG на русском

**Audio RAG** — прототип системы для индексации подкастов и ответов на вопросы по аудио с использованием Triton Inference Server и LLM.

## Возможности

- 🎧 Ингест подкастов через Triton (ASR + chunking + индексация)
- 📝 Текстовые и аудио-вопросы через Triton
- 🤖 Генерация ответов с помощью LLM (Qwen2.5-0.5B-Instruct)
- 🔍 Поиск релевантных фрагментов с цитатами
- 🐳 Docker Compose для быстрого запуска
- ⚙️ Конфигурация через Hydra
- 🗄️ Qdrant векторная БД для эффективного поиска
- 🎯 BGE-M3 эмбеддинги с поддержкой мультиязычности

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

### 2. Настройка переменных окружения

```bash
# Скопировать пример файла окружения
cp .env.example .env

# Отредактировать .env при необходимости (опционально, дефолты работают для локальной разработки)
```

### 3. Запуск Triton сервера

```bash
# Собрать и запустить Docker контейнеры
docker-compose up -d

# Проверить что сервер готов
curl http://localhost:8000/v2/health/ready
```

Ожидаемый ответ: пустой ответ с HTTP 200.

### 4. Проверка статуса моделей

```bash
# Посмотреть логи запуска (загрузка моделей занимает 2-3 минуты)
docker-compose logs -f triton
```

Ожидаемый вывод:
```
"successfully loaded 'asr_whisper'"
"successfully loaded 'bge_embedder'"
"successfully loaded 'reranker'"
"successfully loaded 'ingest_bls'"
"successfully loaded 'query_bls'"
"successfully loaded 'llm_qwen'"
```

## Использование

### Ингест подкаста

```bash
python main.py triton-ingest-podcast \
  --source my-podcast \
  --audio-file ./path/to/podcast.mp3
```

Ожидаемый результат:
```json
{
  "status": "success",
  "chunks_count": 15,
  "source_id": "my-podcast"
}
```

### Задать вопрос (текст)

```bash
python main.py triton-ask "О чем говорилось в подкасте?"
```

### Задать вопрос (аудио)

```bash
python main.py triton-ask-audio --question-audio-file ./question.mp3
```

### Пример ответа с LLM

```
Resolved question transcript: О чем говорилось в подкасте?

Основываясь на транскрипте, в подкасте обсуждался текущий курс доллара
к российскому рублю по состоянию на 5 мая 2022 года...

Citations:
- my-podcast [0:18] score=0.847: На сегодняшний день 5 мая 2022 года...
```

## Структура проекта

```
audio-rag/
├── audio_rag/           # Основной пакет
│   ├── embedders/       # BGE-M3 и Triton эмбеддеры
│   ├── stores/          # Qdrant и JSONL хранилища
│   ├── service.py       # Основная бизнес-логика
│   ├── cli.py          # Консольный интерфейс
│   └── config.py       # Загрузка конфигурации
├── model_repo/         # Репозиторий моделей Triton
│   ├── asr_whisper/    # Whisper ASR модель
│   ├── bge_embedder/   # BGE-M3 эмбеддинги
│   ├── reranker/       # BGE реранкер
│   ├── ingest_bls/     # Пайплайн ингеста
│   ├── query_bls/      # Пайплайн запросов
│   └── llm_qwen/       # LLM для ответов
├── conf/               # Конфигурация Hydra
├── tests/              # Тесты
├── docker-compose.yml  # Docker сервисы
├── Dockerfile.triton   # Triton контейнер
└── .env.example        # Шаблон переменных окружения
```

## Конфигурация

Конфигурация управляется через Hydra. Основной файл: `conf/config.yaml`.

### Переменные окружения

Основные переменные окружения (см. `.env.example` для полного списка):

| Переменная | Описание | По умолчанию |
|------------|----------|--------------|
| `QDRANT_HOST` | Имя хоста Qdrant | `localhost` |
| `QDRANT_PORT` | Порт Qdrant | `6333` |
| `TRITON_SERVER` | Установите "true" внутри Triton | `false` |
| `AUDIO_RAG_ASR_MODEL_SIZE` | Размер модели Whisper | `tiny` |
| `AUDIO_RAG_LLM_MODEL` | Имя модели LLM | `Qwen/Qwen2.5-0.5B-Instruct` |

**Важно:** При запуске внутри Docker Compose установите `QDRANT_HOST=qdrant` (имя сервиса).

### Конфигурация Docker Compose

Файл `docker-compose.yml` автоматически устанавливает правильные переменные окружения:
- Контейнер Triton использует `QDRANT_HOST=qdrant` для подключения к сервису Qdrant
- Локальный клиент использует `QDRANT_HOST=localhost` (по умолчанию)

## Архитектура

### Компоненты системы

1. **ASR (Whisper)** - Распознавание речи из аудио файлов
2. **Embedder (BGE-M3)** - Текстовые эмбеддинги для семантического поиска
3. **Vector Store (Qdrant)** - Эффективный поиск по сходству
4. **Reranker (BGE-Reranker)** - Улучшение релевантности поиска
5. **LLM (Qwen)** - Генерация ответов на естественном языке

### Пайплайн Ingest

```
Аудио файл → ASR (Whisper) → Транскрипт → Chunking → Embedding (BGE-M3) → Qdrant
```

### Пайплайн Query

```
Вопрос → Embedding → Поиск в Qdrant → Reranking → Генерация LLM → Ответ
```

### Модели Triton

| Модель | Назначение | Оборудование |
|--------|------------|--------------|
| `asr_whisper` | Распознавание речи | CPU/GPU |
| `bge_embedder` | Текстовые эмбеддинги | CPU/GPU |
| `reranker` | Реранкинг поиска | CPU/GPU |
| `ingest_bls` | Оркестрация ингеста | CPU |
| `query_bls` | Оркестрация запросов | CPU |
| `llm_qwen` | Генерация ответов | CPU/GPU |

## Разработка

### Запуск тестов

```bash
# Запустить все тесты
pytest tests/

# Запустить конкретный тест
pytest tests/test_mvp.py -v
```

**Примечание:** Для тестов требуется запущенный Qdrant (`docker-compose up -d qdrant`).

### Качество кода

```bash
# Установить dev зависимости
pip install -e ".[dev]"

# Запустить линтинг
flake8 audio_rag/ tests/
mypy audio_rag/
```

### Сборка Docker образа

```bash
docker-compose build triton
```

### Очистка

```bash
# Остановить все сервисы
docker-compose down

# Удалить volumes (очищает все данные)
docker-compose down -v

# Удалить потерянные образы
docker image prune
```

## Устранение неполадок

### Ошибка: Connection refused к Qdrant

**Симптом:** `ConnectionRefusedError: [Errno 111] Connection refused`

**Решение:**
1. Убедитесь, что Qdrant запущен: `docker-compose ps`
2. Проверьте переменные окружения: `QDRANT_HOST` должен быть `qdrant` в Docker, `localhost` локально
3. Проверьте здоровье Qdrant: `curl http://localhost:6333/collections`

### Ошибка: QdrantClient has no attribute 'search'

**Симптом:** `AttributeError: 'QdrantClient' object has no attribute 'search'`

**Решение:** Исправлено в текущей версии. Код теперь использует API `query_points()`, совместимый с qdrant-client 1.16+.

### Ошибка: ModuleNotFoundError: No module named 'packaging'

**Симптом:** Ошибка при загрузке моделей в Triton.

**Решение:** Исправлено в текущем Dockerfile через sitecustomize.py.

### Модель LLM не загружается

**Симптом:** Нехватка памяти или медленный запуск.

**Решение:**
1. Используйте модель поменьше: `AUDIO_RAG_LLM_MODEL=Qwen/Qwen2.5-0.5B-Instruct`
2. Отключите LLM: `AUDIO_RAG_USE_LLM=false`

### Ответы шаблонные вместо LLM

**Симптом:** Ответы основаны на шаблонах, не генерируются LLM.

**Решение:**
1. Проверьте, что LLM включен: `AUDIO_RAG_USE_LLM=true`
2. Проверьте логи LLM: `docker-compose logs triton | grep llm_qwen`

## Системные требования

- **Python:** 3.9+
- **Docker:** 20.10+
- **Docker Compose:** 2.0+
- **RAM:** 4GB минимум, 8GB рекомендуется
- **Диск:** 10GB для моделей и данных

## Последние обновления

### 2025-01-06
- ✅ Исправлено подключение к Qdrant через переменные окружения в config.yaml
- ✅ Обновлено до API qdrant-client 1.16+ (query_points)
- ✅ Реализован метод execute модели query_bls
- ✅ Добавлен .env.example с подробной документацией
- ✅ Удалены неиспользуемые переменные из тестов
- ✅ Тестовые аудиофайлы добавлены в .gitignore

## План развития

- [ ] Интеграция с Telegram ботом
- [ ] Web UI
- [ ] Поддержка мультиязычности
- [ ] Batch ингест
- [ ] Кастомные модели эмбеддингов
- [ ] Метрики оценки RAG

## Вклад в проект

Приветствуются любые вклады! Пожалуйста:

1. Сделайте форк репозитория
2. Создайте ветку для функции
3. Внесите изменения
4. Запустите тесты: `pytest tests/`
5. Отправьте pull request

## Лицензия

MIT License - см. [LICENSE](LICENSE) для деталей.