<div align="center">

# 🎙️ Audio RAG

**RAG System for Audio Content with Triton Inference Server**

*Transcribe, index, and query audio content with AI-powered search*

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Docker](https://img.shields.io/badge/docker-required-blue.svg)](https://www.docker.com/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

[Features](#-features) • [Quick Start](#-quick-start) • [Usage](#-usage) • [Architecture](#-architecture) • [Documentation](#-documentation)

</div>

---

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#-features)
- [Demo](#-demo)
- [Requirements](#-requirements)
- [Quick Start](#-quick-start)
- [Usage](#-usage)
  - [Ingest Audio](#ingest-audio)
  - [Ask Questions](#ask-questions`.
- [Architecture](#-architecture)
- [Project Structure](#-project-structure)
- [Configuration](#-configuration)
- [Development](#-development)
- [Troubleshooting](#-troubleshooting)
- [Roadmap](#-roadmap)
- [Contributing](#-contributing)
- [License](#-license)
- [Acknowledgments](#-acknowledgments)

---

## Overview

**Audio RAG** is a production-ready system for indexing and querying audio content using Retrieval-Augmented Generation (RAG). Built on NVIDIA Triton Inference Server, it provides:

- 🎯 **Speech-to-Text** - Whisper-powered transcription with timestamps
- 🔍 **Semantic Search** - BGE-M3 multilingual embeddings with Qdrant vector DB
- 🤖 **AI Answers** - LLM-generated responses with source citations
- 🚀 **Production Ready** - Docker-based deployment with Triton inference server

Perfect for podcasts, interviews, lectures, and any audio content you want to make searchable and queryable.

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| 🎤 **Audio Transcription** | Whisper ASR with word-level timestamps |
| 🌍 **Multilingual** | Support for 99+ languages via BGE-M3 embeddings |
| 📊 **Vector Search** | Qdrant-powered similarity search |
| 🔄 **Reranking** | BGE-Reranker for improved relevance |
| 💬 **LLM Answers** | Contextual answers with citations |
| 🐳 **Docker Ready** | One-command deployment |
| ⚡ **Triton Powered** | Scalable inference serving |
| 🎛️ **Configurable** | Hydra-based configuration |

---

## Demo

### Ingest a Podcast

```bash
python main.py triton-ingest-podcast \
  --source my-podcast \
  --audio-file ./podcast.mp3
```

```json
{
  "status": "success",
  "chunks_count": 15,
  "source_id": "my-podcast"
}
```

### Ask Questions

```bash
python main.py triton-ask "What was discussed about exchange rates?"
```

```
Resolved question: What was discussed about exchange rates?

The current exchange rate as of May 5, 2022 is 75.50 rubles per US dollar.

Citations:
• my-podcast [0:18] score=0.847
  "На сегодняшний день 5 мая 2022 года курс доллара составляет 
   75 рублей и 50 копеек к российскому рублю."
```

---

## Requirements

### Minimum

| Requirement | Version |
|-------------|---------|
| Python | 3.9+ |
| Docker | 20.10+ |
| Docker Compose | 2.0+ |
| RAM | 4 GB |
| Disk | 10 GB |
| CPU | 2 cores |

### Recommended

| Requirement | Version |
|-------------|---------|
| RAM | 8 GB+ |
| CPU | 4 cores+ |
| GPU | NVIDIA with CUDA (optional) |
| Disk | 20 GB SSD |

---

## 🚀 Quick Start

### 1. Clone & Install

```bash
# Clone the repository
git clone <repo-url>
cd audio-rag

# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate  # Linux/macOS
# or: .venv\Scripts\activate  # Windows

# Install dependencies
pip install -e .
```

### 2. Configure Environment

```bash
# Copy environment template
cp .env.example .env

# Edit if needed (defaults work for local development)
# nano .env
```

### 3. Start Services

```bash
# Build and start all services (Qdrant + Triton)
docker-compose up -d

# Wait for models to load (3-10 min on first run)
# Models are downloaded from Hugging Face (~2-3GB total)
docker-compose logs -f triton
```

**Expected output when all models are ready:**

```
✓ successfully loaded 'asr_whisper'
✓ successfully loaded 'bge_embedder'
✓ successfully loaded 'reranker'
✓ successfully loaded 'ingest_bls'
✓ successfully loaded 'query_bls'
✓ successfully loaded 'llm_qwen'
```

Press `Ctrl+C` to stop following logs.

### 4. Verify Setup

```bash
# Check Triton server health
curl http://localhost:8000/v2/health/ready

# Check Qdrant health
curl http://localhost:6333/collections
```

Both should return HTTP 200. You're ready to go! 🎉

---

## Usage

### Ingest Audio

Import audio content into the vector database:

```bash
python main.py triton-ingest-podcast \
  --source <source-id> \
  --audio-file ./path/to/audio.mp3
```

**Parameters:**

| Parameter | Description |
|-----------|-------------|
| `--source` | Unique identifier for the audio source |
| `--audio-file` | Path to audio file (MP3, WAV, M4A, etc.) |

**Output:**

```json
{
  "status": "success",
  "chunks_count": 15,
  "source_id": "my-podcast"
}
```

### Ask Questions

#### Text Question

```bash
python main.py triton-ask "What topics were discussed?"
```

#### Audio Question (Voice Input)

```bash
python main.py triton-ask-audio --question-audio-file ./question.mp3
```

**Response format:**

```
Resolved question transcript: [Transcribed question]

[LLM-generated answer based on context]

Citations:
- source_id [start:end] score=0.XXX
  "Relevant text snippet from audio..."
```

---

## Architecture

### System Components

```
┌─────────────────────────────────────────────────────────────────┐
│                        TRITON SERVER                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐             │
│  │ ASR Whisper │  │ BGE-M3 Emb. │  │   Reranker  │             │
│  │   (Audio→   │  │  (Text→     │  │ (Relevance  │             │
│  │    Text)    │  │  Vectors)   │  │  Scoring)   │             │
│  └─────────────┘  └─────────────┘  └─────────────┘             │
│                                                                 │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐             │
│  │ Ingest BLS  │  │  Query BLS  │  │  LLM Qwen   │             │
│  │ (Ingestion  │  │  (Query     │  │ (Answer     │             │
│  │  Pipeline)  │  │  Pipeline)  │  │ Generation) │             │
│  └─────────────┘  └─────────────┘  └─────────────┘             │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
                    ┌─────────────────┐
                    │     Qdrant      │
                    │  Vector Store   │
                    └─────────────────┘
```

### Data Flow

#### Ingestion Pipeline

```
Audio File ──▶ Whisper ASR ──▶ Transcript
                                    │
                                    ▼
                              Chunking (120 words)
                                    │
                                    ▼
                              BGE-M3 Embeddings
                                    │
                                    ▼
                               Qdrant Store
```

#### Query Pipeline

```
Question ──▶ BGE-M3 Embed ──▶ Qdrant Search
                                      │
                                      ▼
                                Reranker
                                      │
                                      ▼
                              LLM Qwen
                                      │
                                      ▼
                               Answer + Citations
```

### Models

| Model | Type | Size | Purpose |
|-------|------|------|---------|
| Whisper | ASR | tiny/base/small/medium | Speech-to-text transcription |
| BGE-M3 | Embedding | 568M | Multilingual text embeddings (1024 dim) |
| BGE-Reranker-v2-m3 | Reranker | 560M | Cross-encoder relevance scoring |
| Qwen2.5-0.5B-Instruct | LLM | 0.5B | Contextual answer generation |

---

## Project Structure

```
audio-rag/
├── audio_rag/              # Main Python package
│   ├── embedders/          # Text embedding implementations
│   │   ├── bge.py         # BGE-M3 local embedder
│   │   ├── triton_bge.py  # BGE-M3 Triton client
│   │   └── hashing.py     # Deterministic embedder (testing)
│   ├── stores/            # Vector store implementations
│   │   ├── qdrant_store.py  # Qdrant vector database
│   │   └── jsonl_store.py   # JSONL file-based store
│   ├── service.py         # Core RAG business logic
│   ├── cli.py             # Command-line interface
│   ├── config.py          # Configuration loader
│   ├── factories.py       # Component factories
│   └── utils/             # Utilities
│       └── logging.py     # Logging configuration
│
├── model_repo/            # Triton model repository
│   ├── asr_whisper/       # Whisper ASR model
│   ├── bge_embedder/      # BGE-M3 embedding model
│   ├── reranker/          # BGE reranker model
│   ├── ingest_bls/        # Ingestion orchestration
│   ├── query_bls/         # Query orchestration
│   └── llm_qwen/          # LLM answer generation
│
├── conf/                  # Hydra configuration
│   └── config.yaml        # Main configuration
│
├── tests/                 # Test suite
│   ├── test_mvp.py        # Integration tests
│   ├── test_triton_client.py  # Triton client tests
│   └── test_audio_workflow.py # Audio workflow tests
│
├── docker-compose.yml     # Docker services
├── Dockerfile.triton      # Triton container
├── .env.example           # Environment template
├── pyproject.toml         # Project metadata
└── requirements-*.txt     # Dependencies
```

---

## Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `QDRANT_HOST` | Qdrant server hostname | `localhost` |
| `QDRANT_PORT` | Qdrant server port | `6333` |
| `TRITON_SERVER` | Set to `"true"` inside Triton | `false` |
| `AUDIO_RAG_ASR_MODEL_SIZE` | Whisper model (`tiny`/`base`/`small`/`medium`) | `tiny` |
| `AUDIO_RAG_ASR_DEVICE` | ASR device (`cpu`/`cuda`) | `cpu` |
| `AUDIO_RAG_LLM_MODEL` | LLM model name | `Qwen/Qwen2.5-0.5B-Instruct` |
| `AUDIO_RAG_LLM_DEVICE` | LLM device (`cpu`/`cuda`) | `cpu` |
| `AUDIO_RAG_LLM_MAX_TOKENS` | Max tokens for response | `512` |
| `AUDIO_RAG_USE_LLM` | Enable/disable LLM | `true` |

### Hydra Configuration

Edit `conf/config.yaml` for advanced settings:

```yaml
chunking:
  chunk_words: 120        # Words per chunk
  overlap_words: 24       # Overlap between chunks

retrieval:
  default_top_k: 5        # Number of results to retrieve

qdrant:
  collection_name: audio_rag_chunks
  vector_size: 1024       # BGE-M3 embedding dimension

bge:
  model_name: BAAI/bge-m3
  device: cpu
  max_length: 512
```

### GPU Support

For GPU acceleration, install [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) and update `docker-compose.yml`:

```yaml
services:
  triton:
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    environment:
      AUDIO_RAG_ASR_DEVICE: cuda
      AUDIO_RAG_LLM_DEVICE: cuda
```

---

## Development

### Running Tests

```bash
# Start Qdrant (required for tests)
docker-compose up -d qdrant

# Run all tests
pytest tests/ -v

# Run specific test
pytest tests/test_mvp.py -v

# Run with coverage
pytest tests/ --cov=audio_rag
```

### Code Quality

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Format code
black audio_rag/ tests/

# Lint
flake8 audio_rag/ tests/

# Type check
mypy audio_rag/
```

### Docker Commands

```bash
# Build Triton image
docker-compose build triton

# Rebuild without cache
docker-compose build --no-cache triton

# View logs
docker-compose logs -f triton

# Stop all services
docker-compose down

# Remove volumes (clears all data)
docker-compose down -v
```

---

## Troubleshooting

### Models Not Loading

**Symptoms:** Triton logs show model loading errors or timeouts.

**Solutions:**

1. Check disk space (models require ~3GB):
   ```bash
   df -h
   ```

2. Check internet connection (models download from Hugging Face):
   ```bash
   curl -I https://huggingface.co
   ```

3. Check logs for errors:
   ```bash
   docker-compose logs triton | grep -i error
   ```

4. Restart with clean state:
   ```bash
   docker-compose down -v && docker-compose up -d
   ```

### Connection Refused to Qdrant

**Symptoms:** `ConnectionRefusedError: [Errno 111] Connection refused`

**Solutions:**

1. Check Qdrant is running:
   ```bash
   docker-compose ps qdrant
   ```

2. Verify Qdrant health:
   ```bash
   curl http://localhost:6333/collections
   ```

3. Check environment:
   - Inside Docker: `QDRANT_HOST=qdrant`
   - Local client: `QDRANT_HOST=localhost`

### Out of Memory

**Symptoms:** Container crashes or becomes unresponsive.

**Solutions:**

1. Use smaller models:
   ```yaml
   AUDIO_RAG_ASR_MODEL_SIZE: tiny
   AUDIO_RAG_LLM_MODEL: Qwen/Qwen2.5-0.5B-Instruct
   ```

2. Disable LLM:
   ```yaml
   AUDIO_RAG_USE_LLM: "false"
   ```

3. Increase Docker memory allocation (4GB+ recommended)

### Slow Model Loading

**Symptoms:** Models take >15 minutes to load.

**This is normal on first run** - models download from Hugging Face (~3GB). Subsequent starts are faster.

Check download progress:
```bash
docker-compose logs triton | grep "Loading"
```

---

## Roadmap

### Short Term

- [ ] Multiple audio format support (WAV, M4A, FLAC)
- [ ] Batch ingestion for multiple files
- [ ] REST API endpoint
- [ ] Improved error messages

### Medium Term

- [ ] Telegram bot integration
- [ ] Web UI for podcast management
- [ ] Multi-language UI
- [ ] RAG evaluation metrics

### Long Term

- [ ] Custom embedding model support
- [ ] Audio segment playback
- [ ] Speaker diarization
- [ ] Real-time transcription
- [ ] Cloud deployment guides

---

## Contributing

Contributions are welcome! Please follow these steps:

1. **Fork** the repository
2. **Create** a feature branch (`git checkout -b feature/amazing-feature`)
3. **Make** your changes
4. **Run** tests (`pytest tests/ -v`)
5. **Commit** changes (`git commit -m 'Add amazing feature'`)
6. **Push** to branch (`git push origin feature/amazing-feature`)
7. **Open** a Pull Request

### Code Style

- Follow PEP 8 guidelines
- Use type hints
- Write docstrings for all functions
- Keep functions under 50 lines
- Add tests for new features

### Reporting Issues

Please include:
- Python version
- Docker version
- Operating system
- Steps to reproduce
- Expected vs actual behavior
- Relevant logs

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Acknowledgments

- [NVIDIA Triton Inference Server](https://github.com/triton-inference-server/server) - Scalable model serving
- [OpenAI Whisper](https://github.com/openai/whisper) - Speech recognition
- [BAAI BGE-M3](https://huggingface.co/BAAI/bge-m3) - Multilingual embeddings
- [Qwen Team](https://huggingface.co/Qwen) - Qwen LLM models
- [Qdrant](https://qdrant.tech/) - Vector database
- [Hydra](https://hydra.cc/) - Configuration framework

---

<div align="center">

**[⬆ Back to Top](#️-audio-rag)**

Made with ❤️ for the audio AI community

</div>
