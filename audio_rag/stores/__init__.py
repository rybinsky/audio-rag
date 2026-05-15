"""Stores package for chunk storage backends."""

from .base import BaseStore
from .jsonl_store import JsonlChunkStore

# Conditional import for Qdrant (optional dependency)
try:
    from .qdrant_store import QdrantChunkStore
except ImportError:
    QdrantChunkStore = None  # Not available if qdrant-client is not installed

__all__ = [
    "BaseStore",
    "JsonlChunkStore",
    "QdrantChunkStore",
]
