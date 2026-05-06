"""Factories for creating embedders and stores based on configuration."""

from pathlib import Path


from .embedders import BaseEmbedder, TritonBGEEmbedder
from .settings import AppSettings
from .stores import BaseStore, JsonlChunkStore, QdrantChunkStore
from .triton_reranker import TritonReranker


def create_embedder(settings: AppSettings, triton_url: str = "localhost:8000") -> BaseEmbedder:
    """Create embedder using Triton BGE.

    Args:
        settings: Application settings
        triton_url: Triton server URL

    Returns:
        TritonBGEEmbedder instance
    """
    return TritonBGEEmbedder(
        settings=settings.triton_embedder,
        triton_url=triton_url,
    )


def create_store(settings: AppSettings, explicit_path: Path = None) -> BaseStore:
    """Create store based on configuration.

    Args:
        settings: Application settings
        explicit_path: Optional explicit path for JSONL store

    Returns:
        BaseStore instance based on configuration:
        - QdrantChunkStore if store.type == "qdrant"
        - JsonlChunkStore if store.type == "jsonl"

    Raises:
        ValueError: If store type is unknown
    """
    store_type = settings.store.type.lower().strip()

    if store_type == "qdrant":
        return QdrantChunkStore(settings.qdrant)

    elif store_type == "jsonl":
        from .settings import default_data_dir

        if explicit_path is not None:
            store_path = explicit_path.expanduser().resolve()
        else:
            store_path = default_data_dir(settings) / settings.store.filename
        return JsonlChunkStore(store_path)

    else:
        raise ValueError(
            f"Unknown store type: {store_type}. "
            f"Supported types: 'qdrant', 'jsonl'"
        )


def create_reranker(settings: AppSettings, triton_url: str = "localhost:8000") -> TritonReranker:
    """Create reranker using Triton.

    Args:
        settings: Application settings
        triton_url: Triton server URL

    Returns:
        TritonReranker instance
    """
    return TritonReranker(
        settings=settings.triton_embedder,
        triton_url=triton_url,
    )
