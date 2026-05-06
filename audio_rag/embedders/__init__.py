"""Embedders package for text embeddings."""

from .base import BaseEmbedder
from .bge import BGEEmbedder

__all__ = [
    "BaseEmbedder",
    "BGEEmbedder",
]
