# audio-rag

`audio-rag` — прототип Audio RAG для небольших подкастов с **Triton-first** сценарием работы:

1. поднимается Triton в Docker,
2. пользователь передаёт аудио подкаста (`.mp3`, `.m4a`, `.wav`, `.ogg`, `.aac`),
3. Triton транскрибирует аудио через `faster-whisper`,
4. транскрипт режется на чанки и индексируется в локальный store,
5. пользователь передаёт аудио-вопрос,
6. Triton снова использует ASR,
7. система возвращает текстовый ответ с цитатами и ссылкой на исходный аудиофайл.

Проект сейчас решает локальный end-to-end сценарий `audio -> Triton -> text answer`. Это не production-версия и не полный стек из финального ТЗ, но уже runnable вертикальный срез, который можно поднимать, тестировать и развивать.

## Возможности

- ingest подкаста через Triton: `triton-ingest-podcast`
- текстовый вопрос через Triton: `triton-ask`
- аудио-вопрос через Triton: `triton-ask-audio`
- локальный transcript-first fallback без Triton
- ASR на базе `faster-whisper`
- JSONL-хранилище чанков
- цитаты с `source_id`, offsets и `audio_path`
- автоматическое преобразование локальных путей в `/workspace/...` для Triton-контейнера
- конфигурация через Hydra

## Текущие ограничения

Сейчас в проекте намеренно упрощены:

- retrieval: локальный hashing embedder вместо production embeddings
- storage: JSONL вместо Qdrant
- orchestration: Python backend в Triton без ensemble graph и без reranker
- generation: ответ формируется из найденного контекста без LLM-стриминга

## Структура проекта

```text
.
├── audio_rag/
│   ├── chunking.py
│   ├── cli.py
│   ├── config.py
│   ├── embeddings.py
│   ├── models.py
│   ├── service.py
│   ├── settings.py
│   ├── store.py
│   └── triton_client.py
├── conf/
│   └── config.yaml
├── model_repo/
│   ├── asr_whisper/
│   ├── ingest_bls/
│   └── query_bls/
├── tests/
├── Dockerfile.triton
├── docker-compose.yml
├── main.py
├── pyproject.toml
├── requirements-dev.txt
└── requirements-triton.txt
```

## Требования

### Для локального клиента

- Python 3.9+
- `venv`
- `pip`

### Для Triton

- Docker
- Docker Compose

## Установка локального окружения

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-dev.txt
```

Это установит:

- сам проект в editable-режиме
- `hydra-core`
- `omegaconf`
- `pytest`

## Конфигурация через Hydra

Базовый runtime-конфиг лежит в:

```text
conf/config.yaml
```

В конфиг вынесены:

- размеры чанков
- параметры retrieval
- параметры store
- HTTP-настройки Triton client
- пути хоста и контейнера
- параметры ASR

Ключевые секции:

- `chunking`
- `embedding`
- `retrieval`
- `transcript`
- `metadata`
- `store`
- `triton_http`
- `triton_server`
- `asr`

## Сборка и запуск Triton

```bash
docker compose up --build triton
```

Compose:

- собирает `Dockerfile.triton`
- ставит runtime-зависимости из `requirements-triton.txt`
- публикует порты `8000`, `8001`, `8002`
- монтирует проект в `/workspace`
- монтирует model repository в `/models`
- сохраняет индекс в `./tmp/chunks.jsonl`
- кеширует Whisper-модель в volume `triton_cache`

## Проверка готовности Triton

```bash
curl http://localhost:8000/v2/health/live
curl http://localhost:8000/v2/health/ready
```

Проверка моделей:

```bash
curl http://localhost:8000/v2/models/asr_whisper
curl http://localhost:8000/v2/models/ingest_bls
curl http://localhost:8000/v2/models/query_bls
```

## Основной пользовательский сценарий

### 1. Индексация подкаста

```bash
python main.py triton-ingest-podcast \
  --source my-podcast \
  --audio-file tests/Подкаст.mp3
```

CLI автоматически преобразует путь в контейнерный вид `/workspace/tests/Подкаст.mp3`, если файл лежит внутри репозитория.

Пример ответа:

```json
{
  "source_id": "my-podcast",
  "indexed_chunks": 1,
  "audio_path": "/workspace/tests/Подкаст.mp3",
  "transcript_origin": "asr_whisper"
}
```

### 2. Аудио-вопрос

```bash
python main.py triton-ask-audio \
  --question-audio-file tests/Вопрос.m4a
```

Пример ответа:

```text
Resolved question transcript: Какой курс доллара сегодня?
По локальному индексу лучший контекст для запроса 'Какой курс доллара сегодня?': ...

Citations:
- my-podcast [0:18] score=0.320: ... | audio=/workspace/tests/Подкаст.mp3
```

### 3. Текстовый вопрос

```bash
python main.py triton-ask "О чём говорилось в подкасте?"
```

## Локальный режим без Triton

### Индексация готового транскрипта

```bash
python main.py ingest-text \
  --source sample \
  --file examples/sample_transcript.txt
```

### Индексация подкаста с sidecar transcript

```bash
python main.py ingest-podcast \
  --source sample \
  --audio-file tests/Подкаст.mp3 \
  --transcript-file tests/Подкаст.transcript.txt
```

### Локальный вопрос

```bash
python main.py ask "what was said about retrieval?"
python main.py ask-audio --question-audio-file tests/Вопрос.m4a --question-transcript-file tests/Вопрос.question.txt
```

## Triton-модели

### `asr_whisper`

Назначение: ASR для входного аудио.

Использует:

- `faster-whisper`
- `AUDIO_RAG_ASR_MODEL_SIZE`
- `AUDIO_RAG_ASR_COMPUTE_TYPE`
- `AUDIO_RAG_ASR_DEVICE`
- `AUDIO_RAG_ASR_VAD_FILTER`

### `ingest_bls`

Назначение: ingest подкаста.

Поведение:

- принимает `source_id`, `audio_path`, optional `transcript_path`, metadata
- если транскрипт не передан, вызывает `asr_whisper`
- индексирует результат через service layer

### `query_bls`

Назначение: query по уже загруженным данным.

Поведение:

- принимает текст запроса или аудио-вопрос
- при audio-вопросе вызывает `asr_whisper`
- делает retrieval в JSONL store
- возвращает `answer`, `resolved_query_text`, `citations`

## Тесты

```bash
python -m unittest tests.test_mvp tests.test_audio_workflow tests.test_triton_client
pytest -q
```

## Кодстайл и принципы текущей версии

- конфигурация вынесена в Hydra
- runtime-параметры не разбросаны по файлам
- сервисный слой отделён от Triton transport слоя
- клиентский слой не требует ручного `/workspace/...` для файлов внутри репозитория
- удалены `from __future__ import annotations`
- удалены динамические `__import__(...)`

## Что логично делать дальше

1. заменить hashing embedder на нормальные embeddings,
2. вынести store в Qdrant,
3. добавить reranking,
4. подключить LLM для нормальной генерации ответа,
5. добавить UI и/или HTTP API поверх CLI.
