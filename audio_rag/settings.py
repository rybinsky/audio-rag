from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class ChunkingSettings:
    chunk_words: int = 120
    overlap_words: int = 24


@dataclass
class EmbeddingSettings:
    dimension: int = 256
    token_pattern: str = r"[A-Za-zА-Яа-яЁё0-9_]+"


@dataclass
class RetrievalSettings:
    default_top_k: int = 5
    max_preview_words: int = 24
    no_results_answer: str = (
        "Не нашёл релевантных фрагментов в локальном индексе. "
        "Сначала загрузите транскрипт или расширьте корпус."
    )
    source_preview_prefix: str = "Источники"


@dataclass
class TranscriptSettings:
    encoding: str = "utf-8"
    podcast_suffix: str = ".transcript.txt"
    question_suffix: str = ".question.txt"


@dataclass
class MetadataSettings:
    audio_path_key: str = "audio_path"
    transcript_path_key: str = "transcript_path"
    transcript_origin_key: str = "transcript_origin"
    ingest_mode_key: str = "ingest_mode"
    ingest_mode_transcript: str = "transcript"
    ingest_mode_podcast: str = "podcast"
    ingest_mode_triton: str = "triton_ingest_bls"


@dataclass
class StoreSettings:
    app_dirname: str = ".audio_rag"
    filename: str = "chunks.jsonl"


@dataclass
class TritonHttpSettings:
    base_url: str = "http://localhost:8000"
    infer_endpoint_template: str = "/v2/models/{model_name}/infer"
    json_content_type: str = "application/json"
    model_ingest_name: str = "ingest_bls"
    model_query_name: str = "query_bls"
    model_asr_name: str = "asr_whisper"
    host_project_root: Optional[str] = None
    container_project_root: str = "/workspace"


@dataclass
class TritonServerSettings:
    model_repository_path: str = "/models"
    store_path: str = "/data/chunks.jsonl"
    asr_model_size: str = "tiny"
    asr_compute_type: str = "int8"
    cache_dir: str = "/root/.cache"


@dataclass
class AsrSettings:
    model_size: str = "tiny"
    compute_type: str = "int8"
    device: str = "cpu"
    vad_filter: bool = True


@dataclass
class PathSettings:
    workspace_dir: str = "tmp"


@dataclass
class AppSettings:
    chunking: ChunkingSettings = field(default_factory=ChunkingSettings)
    embedding: EmbeddingSettings = field(default_factory=EmbeddingSettings)
    retrieval: RetrievalSettings = field(default_factory=RetrievalSettings)
    transcript: TranscriptSettings = field(default_factory=TranscriptSettings)
    metadata: MetadataSettings = field(default_factory=MetadataSettings)
    store: StoreSettings = field(default_factory=StoreSettings)
    triton_http: TritonHttpSettings = field(default_factory=TritonHttpSettings)
    triton_server: TritonServerSettings = field(default_factory=TritonServerSettings)
    asr: AsrSettings = field(default_factory=AsrSettings)
    paths: PathSettings = field(default_factory=PathSettings)


def default_data_dir(settings: AppSettings) -> Path:
    return Path.home() / settings.store.app_dirname


def default_store_path(settings: AppSettings) -> Path:
    return default_data_dir(settings) / settings.store.filename
