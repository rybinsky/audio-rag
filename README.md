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

# Wait for models to load (takes 3-10 minutes on first run)
# Models are downloaded from Hugging Face (~2-3GB total)
docker-compose logs -f triton
```

**Expected output when all models are ready:**
```
"successfully loaded 'asr_whisper'"
"successfully loaded 'bge_embedder'"
"successfully loaded 'reranker'"
"successfully loaded 'ingest_bls'"
"successfully loaded 'query_bls'"
"successfully loaded 'llm_qwen'"
```

Press `Ctrl+C` to stop following logs once models are loaded.

### 4. Verify Setup

```bash
# Check Triton server health
curl http://localhost:8000/v2/health/ready

# Check Qdrant health
curl http://localhost:6333/collections
```

Both should return HTTP 200.

---

## Usage Examples

### Ingest a Podcast

```bash
python main.py triton-ingest-podcast \
  --source my-podcast \
  --audio-file ./path/to/podcast.mp3
```

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
python main.py triton-ask "What was discussed in the podcast?"
```

#### Audio Question

```bash
python main.py triton-ask-audio --question-audio-file ./question.mp3
```

### Example Output

```
Resolved question transcript: What is the current exchange rate?

The current exchange rate as of May 5, 2022 is 75.50 rubles per US dollar.

Citations:
- my-podcast [0:18] score=0.847: На сегодняшний день 5 мая 2022 года курс доллара составляет 75 рублей и 50 копеек к российскому рублю.
```

**Components of the response:**
- **Resolved question** - The actual question (transcribed if audio)
- **Answer** - LLM-generated response based on context
- **Citations** - Source audio segments with:
  - Source ID
  - Timestamp range
  - Relevance score
  - Text snippet

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
│   │   ├── qdrant.py      # Qdrant vector database
│   │   └── jsonl.py       # JSONL file-based store
│   ├── service.py         # Core RAG business logic
│   ├── cli.py            # Command-line interface
│   ├── config.py         # Configuration loader
│   ├── factories.py      # Component factories
│   ├── triton_client.py  # Triton HTTP client
│   └── utils/            # Utilities
│       └── logging.py    # Logging configuration
│
├── model_repo/            # Triton model repository
│   ├── asr_whisper/      # Whisper ASR model
│   ├── bge_embedder/     # BGE-M3 embedding model
│   ├── reranker/         # BGE reranker model
│   ├── ingest_bls/       # Ingestion orchestration
│   ├── query_bls/        # Query orchestration
│   └── llm_qwen/         # LLM answer generation
│
├── conf/                  # Hydra configuration files
│   └── config.yaml       # Main configuration
│
├── tests/                 # Test suite
│   ├── test_mvp.py       # Integration tests
│   ├── test_triton_client.py  # Triton client tests
│   ├── Подкаст.mp3       # Test podcast
│   └── Вопрос.mp3        # Test question audio
│
├── docker-compose.yml     # Docker services definition
├── Dockerfile.triton      # Triton container build
├── .env.example           # Environment variables template
├── requirements-triton.txt # Python dependencies for Triton
└── pyproject.toml         # Project metadata and dependencies
```

---

## Configuration

### Environment Variables

Key configuration via environment variables:

| Variable | Description | Default |
|----------|-------------|---------|
| `QDRANT_HOST` | Qdrant server hostname | `localhost` |
| `QDRANT_PORT` | Qdrant server port | `6333` |
| `TRITON_SERVER` | Set to `"true"` inside Triton container | `false` |
| `AUDIO_RAG_ASR_MODEL_SIZE` | Whisper model size (`tiny`, `base`, `small`, `medium`) | `tiny` |
| `AUDIO_RAG_ASR_DEVICE` | ASR device (`cpu`, `cuda`) | `cpu` |
| `AUDIO_RAG_LLM_MODEL` | LLM model name | `Qwen/Qwen2.5-0.5B-Instruct` |
| `AUDIO_RAG_LLM_DEVICE` | LLM device (`cpu`, `cuda`) | `cpu` |
| `AUDIO_RAG_LLM_MAX_TOKENS` | Max tokens for LLM response | `512` |
| `AUDIO_RAG_USE_LLM` | Enable/disable LLM generation | `true` |

### Docker Compose Configuration

The `docker-compose.yml` automatically configures:

- **Qdrant** - Vector database on port 6333
- **Triton** - Inference server on port 8000 (HTTP), 8001 (gRPC), 8002 (metrics)
- **Networking** - Containers communicate via service names
- **Volumes** - Persistent storage for Qdrant data

**Important:** Models are **not** cached locally. They are downloaded from Hugging Face on each container restart. This ensures you always have the latest versions.

### Hydra Configuration

Advanced configuration via `conf/config.yaml`:

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

---

## Architecture

### System Components

| Component | Model | Purpose |
|-----------|-------|---------|
| **ASR** | Whisper (tiny) | Speech-to-text transcription |
| **Embedder** | BGE-M3 | Multilingual text embeddings (1024 dim) |
| **Vector Store** | Qdrant | Similarity search and storage |
| **Reranker** | BGE-Reranker-v2-m3 | Improve search relevance |
| **LLM** | Qwen2.5-0.5B-Instruct | Generate contextual answers |

### Triton Models

Each model runs as a separate Triton service:

| Model | Type | Description |
|-------|------|-------------|
| `asr_whisper` | Python backend | Faster-Whisper ASR |
| `bge_embedder` | Python backend | Sentence-Transformers embeddings |
| `reranker` | Python backend | Cross-encoder reranking |
| `ingest_bls` | Python backend | Orchestrates ingestion pipeline |
| `query_bls` | Python backend | Orchestrates query pipeline |
| `llm_qwen` | Python backend | Transformers text generation |

### Data Flow

#### Ingestion Flow

```
┌─────────────┐     ┌──────────────┐     ┌──────────────┐
│ Audio File  │────▶│ ASR Whisper  │────▶│  Transcript  │
└─────────────┘     └──────────────┘     └──────────────┘
                                                │
                                                ▼
┌─────────────┐     ┌──────────────┐     ┌──────────────┐
│   Qdrant    │◀────│ BGE Embedder │◀────│   Chunking   │
└─────────────┘     └──────────────┘     └──────────────┘
```

#### Query Flow

```
┌─────────────┐     ┌──────────────┐     ┌──────────────┐
│   Question  │────▶│ BGE Embedder │────▶│ Qdrant Search│
└─────────────┘     └──────────────┘     └──────────────┘
                                                │
                                                ▼
┌─────────────┐     ┌──────────────┐     ┌──────────────┐
│    Answer   │◀────│   LLM Qwen   │◀────│   Reranker   │
└─────────────┘     └──────────────┘     └──────────────┘
```

---

## Development

### Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_mvp.py -v

# Run with coverage
pytest tests/ --cov=audio_rag
```

**Note:** Tests require Qdrant to be running:
```bash
docker-compose up -d qdrant
```

### Code Quality

```bash
# Install development dependencies
pip install -e ".[dev]"

# Run linting
flake8 audio_rag/ tests/

# Run type checking
mypy audio_rag/

# Format code
black audio_rag/ tests/
```

### Building Docker Image

```bash
# Build Triton image
docker-compose build triton

# Force rebuild without cache
docker-compose build --no-cache triton
```

### Clean Up

```bash
# Stop all services
docker-compose down

# Remove volumes (clears all data)
docker-compose down -v

# Remove all containers and images
docker-compose down --rmi all -v
```

---

## Troubleshooting

### Models Not Loading

**Symptom:** Triton logs show model loading errors or timeouts.

**Solutions:**
1. **Check disk space** - Models require ~3GB
   ```bash
   df -h
   ```

2. **Check internet connection** - Models download from Hugging Face
   ```bash
   curl -I https://huggingface.co
   ```

3. **Check logs for specific errors**
   ```bash
   docker-compose logs triton | grep -i error
   ```

4. **Restart with clean state**
   ```bash
   docker-compose down -v
   docker-compose up -d
   ```

### Connection Refused to Qdrant

**Symptom:** `ConnectionRefusedError: [Errno 111] Connection refused`

**Solutions:**
1. **Check Qdrant is running**
   ```bash
   docker-compose ps qdrant
   ```

2. **Verify Qdrant health**
   ```bash
   curl http://localhost:6333/collections
   ```

3. **Check environment variables**
   - Inside Docker: `QDRANT_HOST=qdrant`
   - Local client: `QDRANT_HOST=localhost`

### LLM Not Generating Answers

**Symptom:** Receiving template answers instead of LLM-generated responses.

**Solutions:**
1. **Verify LLM is enabled**
   ```bash
   # In docker-compose.yml
   AUDIO_RAG_USE_LLM: "true"
   ```

2. **Check LLM model status**
   ```bash
   docker-compose logs triton | grep llm_qwen
   ```

3. **Check LLM is loaded**
   ```bash
   curl http://localhost:8000/v2/models/llm_qwen
   ```

4. **View LLM logs**
   ```bash
   docker-compose logs triton | grep "LLM request"
   ```

### Out of Memory

**Symptom:** Container crashes or becomes unresponsive.

**Solutions:**
1. **Use smaller models**
   ```yaml
   # In docker-compose.yml
   AUDIO_RAG_ASR_MODEL_SIZE: tiny
   AUDIO_RAG_LLM_MODEL: Qwen/Qwen2.5-0.5B-Instruct
   ```

2. **Disable LLM**
   ```yaml
   AUDIO_RAG_USE_LLM: "false"
   ```

3. **Increase Docker memory** - Allocate at least 4GB to Docker

### Slow Model Loading

**Symptom:** Models take >15 minutes to load.

**Solutions:**
1. **First run downloads models** - Expected behavior, wait for completion
2. **Slow internet** - Models download from Hugging Face (~3GB)
3. **Check download progress**
   ```bash
   docker-compose logs triton | grep "Loading model"
   ```

### ModuleNotFoundError: packaging

**Symptom:** Error when loading models in Triton.

**Solution:** This is fixed in current version via `sitecustomize.py` in the Docker image.

---

## System Requirements

### Minimum Requirements

- **Python:** 3.9 or higher
- **Docker:** 20.10 or higher
- **Docker Compose:** 2.0 or higher
- **RAM:** 4GB minimum
- **Disk:** 10GB free space
- **CPU:** 2 cores

### Recommended Requirements

- **RAM:** 8GB or more
- **CPU:** 4 cores or more
- **GPU:** NVIDIA GPU with CUDA support (optional, speeds up inference)
- **Disk:** 20GB SSD

### GPU Support (Optional)

For GPU acceleration, install NVIDIA Container Toolkit and modify `docker-compose.yml`:

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
```

---

## Recent Updates

### 2025-01-08

**Major Changes:**
- ✅ **Removed local model caching** - Models now download from Hugging Face on each restart
- ✅ **Fixed LLM response handling** - Properly validates LLM outputs
- ✅ **Improved logging** - Consistent, production-ready logging across all models
- ✅ **Removed unused code** - Cleaned up imports and print statements

**Bug Fixes:**
- Fixed `'NoneType' object has no attribute 'as_numpy'` error in LLM
- Fixed Qdrant connection with environment variables
- Updated to qdrant-client 1.16+ API

**Improvements:**
- Better error handling in query pipeline
- Cleaner codebase with no unused imports
- Production-ready logging format

### 2025-01-06

- ✅ Fixed Qdrant connection with environment variables in config.yaml
- ✅ Updated to qdrant-client 1.16+ API (query_points)
- ✅ Implemented query_bls model execute method
- ✅ Added .env.example with comprehensive documentation
- ✅ Removed unused variables from tests
- ✅ Added test audio files to .gitignore

---

## Roadmap

### Short Term

- [ ] Add support for multiple audio formats (WAV, M4A, FLAC)
- [ ] Implement batch ingestion for multiple files
- [ ] Add API endpoint for web integration
- [ ] Improve error messages and user feedback

### Medium Term

- [ ] Telegram bot integration
- [ ] Web UI for podcast management
- [ ] Multi-language support (UI and responses)
- [ ] RAG evaluation metrics

### Long Term

- [ ] Custom embedding model support
- [ ] Audio segment playback
- [ ] Speaker diarization
- [ ] Real-time transcription
- [ ] Cloud deployment guides (AWS, GCP, Azure)

---

## Contributing

We welcome contributions! Please follow these steps:

1. **Fork** the repository
2. **Create** a feature branch (`git checkout -b feature/amazing-feature`)
3. **Make** your changes
4. **Run** tests (`pytest tests/ -v`)
5. **Commit** your changes (`git commit -m 'Add amazing feature'`)
6. **Push** to the branch (`git push origin feature/amazing-feature`)
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

- [Triton Inference Server](https://github.com/triton-inference-server/server) - NVIDIA
- [Whisper](https://github.com/openai/whisper) - OpenAI
- [BGE-M3](https://huggingface.co/BAAI/bge-m3) - BAAI
- [Qwen2.5](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct) - Qwen Team
- [Qdrant](https://qdrant.tech/) - Vector database
- [Hydra](https://hydra.cc/) - Configuration framework