---
title: Audio RAG Demo
emoji: 🎙️
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: 4.0.0
app_file: app.py
pinned: false
license: mit
python_version: 3.12
---

# 🎙️ Audio RAG Demo

This is a demo of the Audio RAG system - a Retrieval-Augmented Generation application for audio content (podcasts, interviews, etc.).

## Features

- **Upload Podcasts**: Ingest audio files with transcripts into a searchable index
- **Ask Questions**: Query your audio content using natural language
- **Get Citations**: Receive answers with precise timestamps and source references

## How to Use

### 1. Upload a Podcast

1. Go to the "Upload Podcast" tab
2. Upload an audio file (.mp3, .wav, .m4a, etc.)
3. Upload a transcript file (.txt) with the spoken content
4. Provide a unique source ID (e.g., "my-podcast-episode-1")
5. Click "Ingest Podcast" to process and index

### 2. Ask Questions

1. Go to the "Ask Questions" tab
2. Type your question about the uploaded content
3. Adjust the number of citations if needed
4. Click "Ask" to get an answer with relevant citations

### 3. View Statistics

Check the "Statistics" tab to see:
- Number of indexed chunks
- List of sources
- Store location

## Technical Stack

This demo uses:
- **Embeddings**: BAAI/bge-m3 (running locally)
- **Reranker**: BAAI/bge-reranker-v2-m3
- **Vector Store**: JSONL (lightweight, no external database needed)
- **UI Framework**: Gradio

## Requirements

The demo requires:
- **Python 3.12+** (Python 3.13 supported with pyaudioop)
- Audio file (.mp3, .wav, .m4a, etc.)
- Transcript file (.txt) with the same content as the audio

**Note**: The transcript must be manually provided. This demo does not include automatic speech recognition (ASR) to keep resource usage low for the free tier.

## Limitations

- **No ASR**: You need to provide transcripts manually
- **CPU-only**: Models run on CPU (slower but free)
- **Small batch size**: Optimized for free tier memory limits
- **Temporary storage**: Data is not persisted between sessions
- **Memory**: May need to upgrade to CPU upgrade for larger models

## Deployment to Hugging Face Spaces

### Quick Deploy

1. Create a new Space on Hugging Face
2. Choose "Gradio" as the SDK
3. Upload these files:
   - `app.py` - Main application
   - `requirements_hf.txt` - Rename to `requirements.txt`
   - `README_HF.md` - Rename to `README.md`
   - `conf/config_hf.yaml` - Configuration
   - `audio_rag/` - Package directory
   - `runtime.txt` - (optional) Python version specification

**Important**: If you encounter audioop errors with Python 3.13, either:
- Use Python 3.12 (recommended)
- Or ensure `pyaudioop` is in requirements.txt (already included)

### Environment Variables (Optional)

You can set these environment variables:
- `AUDIO_RAG_CONFIG_NAME`: Config file name (default: "config_hf")
- `AUDIO_RAG_STORE_PATH`: Path to store JSONL database (default: "/tmp/audio_rag_store.jsonl")

### Deployment Script

Use the provided deployment script for easy deployment:

```bash
# Using Python script
python scripts/deploy_hf.py --username YOUR_USERNAME --token YOUR_HF_TOKEN

# Or using bash script
./scripts/deploy_hf.sh --username YOUR_USERNAME --token YOUR_HF_TOKEN
```

See `scripts/` directory for more details.

## Local Development

To run locally:

```bash
# Install dependencies
pip install -r requirements_hf.txt

# Run the app
python app.py
```

## Project Structure

```
audio-rag/
├── app.py                    # Gradio demo application
├── requirements_hf.txt       # Dependencies for HF Spaces
├── README_HF.md             # This file
├── conf/
│   └── config_hf.yaml       # Configuration for HF Spaces
└── audio_rag/               # Main package
    ├── __init__.py
    ├── config.py
    ├── factories.py
    ├── service.py
    ├── settings.py
    ├── chunking.py
    ├── models.py
    ├── embedders/
    ├── stores/
    └── utils/
```

## About

Audio RAG is a system that enables semantic search and question-answering over audio content by:
1. Chunking transcripts into manageable segments
2. Creating dense embeddings using BGE-M3
3. Storing vectors in a searchable index
4. Retrieving relevant segments using semantic similarity
5. Reranking results for better accuracy

## Links

- [GitHub Repository](https://github.com/yourusername/audio-rag)
- [Full Documentation](https://github.com/yourusername/audio-rag#readme)

## License

MIT License - see LICENSE file for details.

## Acknowledgments

- BAAI for the excellent BGE-M3 and BGE-Reranker models
- Hugging Face for hosting this demo
- Gradio team for the simple UI framework