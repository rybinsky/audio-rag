"""Factories for creating embedders and stores based on configuration."""

import os
from pathlib import Path
from typing import TYPE_CHECKING

from .embedders import BaseEmbedder, BGEEmbedder
from .settings import AppSettings
from .stores import BaseStore, JsonlChunkStore, QdrantChunkStore

# Check if we're running inside Triton server
# Inside Triton, we should use direct implementations (not Triton clients)
IS_TRITON_SERVER = os.environ.get("TRITON_SERVER", "false").lower() == "true"


def create_embedder(settings: AppSettings, triton_url: str = "localhost:8000") -> BaseEmbedder:
    """Create embedder based on configuration.

    Supports multiple embedder types:
    - hashing: HashingTextEmbedder (deterministic, for testing)
    - bge: BGE-M3 embedder (local or Triton-based)

    Args:
        settings: Application settings
        triton_url: Triton server URL (used only for BGE on client)

    Returns:
        BaseEmbedder instance

    Raises:
        ImportError: If tritonclient is not available on client
    """
    embedder_type = settings.embedder.type.lower().strip()

    if embedder_type == "hashing":
        # Hashing embedder for testing/deterministic embeddings
        from .embedders.hashing import HashingTextEmbedder
        return HashingTextEmbedder(embedding_dim=settings.embedder.embedding_dim)

    elif embedder_type == "bge":
        # BGE-M3 embedder (local or Triton-based)
        if IS_TRITON_SERVER:
            # Inside Triton server: use direct implementation
            return BGEEmbedder(settings.bge)
        else:
            # On client: use Triton client
            # Import here to avoid tritonclient dependency in Triton server
            from .embedders.triton_bge import TritonBGEEmbedder
            return TritonBGEEmbedder(
                settings=settings.triton_embedder,
                triton_url=triton_url,
            )

    else:
        raise ValueError(
            f"Unknown embedder type: {embedder_type}. "
            f"Supported types: 'hashing', 'bge'"
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


def create_reranker(settings: AppSettings, triton_url: str = "localhost:8000"):
    """Create reranker based on environment.

    Inside Triton server: uses SearchReranker (direct implementation).
    On client: uses TritonReranker (calls Triton through HTTP API).

    Args:
        settings: Application settings
        triton_url: Triton server URL (used only on client)

    Returns:
        Reranker instance

    Raises:
        ImportError: If tritonclient is not available on client
    """
    if IS_TRITON_SERVER:
        # Inside Triton server: use direct implementation
        from .reranker import SearchReranker
        return SearchReranker(settings.reranker)
    else:
        # On client: use Triton client
        # Import here to avoid tritonclient dependency in Triton server
        from .triton_reranker import TritonReranker
        return TritonReranker(
            settings=settings.triton_embedder,
            triton_url=triton_url,
        )
