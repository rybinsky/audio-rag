from pathlib import Path
from typing import Optional

from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf

from .settings import (
    AppSettings,
    AsrSettings,
    BGESettings,
    ChunkingSettings,
    EmbedderSettings,
    MetadataSettings,
    PathSettings,
    QdrantSettings,
    RerankerSettings,
    RetrievalSettings,
    StoreSettings,
    TranscriptSettings,
    TritonEmbedderSettings,
    TritonHttpSettings,
    TritonServerSettings,
)


def load_settings() -> AppSettings:
    config_dir = Path(__file__).resolve().parent.parent / "conf"
    project_root = config_dir.parent.resolve()
    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()
    with initialize_config_dir(version_base=None, config_dir=str(config_dir)):
        config = compose(config_name="config")
    config_dict = OmegaConf.to_container(config, resolve=True)

    settings = AppSettings(
        chunking=ChunkingSettings(**config_dict["chunking"]),
        retrieval=RetrievalSettings(**config_dict["retrieval"]),
        transcript=TranscriptSettings(**config_dict["transcript"]),
        metadata=MetadataSettings(**config_dict["metadata"]),
        store=StoreSettings(**config_dict["store"]),
        embedder=EmbedderSettings(**config_dict.get("embedder", {})),
        triton_http=TritonHttpSettings(**config_dict["triton_http"]),
        triton_server=TritonServerSettings(**config_dict["triton_server"]),
        asr=AsrSettings(**config_dict["asr"]),
        paths=PathSettings(**config_dict["paths"]),
        qdrant=QdrantSettings(**config_dict["qdrant"]),
        bge=BGESettings(**config_dict.get("bge", {})),
        reranker=RerankerSettings(**config_dict.get("reranker", {})),
        triton_embedder=TritonEmbedderSettings(**config_dict["triton_embedder"]),
    )
    settings.triton_http.host_project_root = str(project_root)
    return settings
