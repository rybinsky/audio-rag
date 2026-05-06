"""Abstract base class for chunk stores."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from ..models import Chunk, SearchResult


class BaseStore(ABC):
    """Abstract base class for chunk storage backends."""

    @property
    @abstractmethod
    def path(self) -> str:
        """Return storage path or connection string.

        Returns:
            Storage path for JSONL or connection string for Qdrant
        """
        pass

    @abstractmethod
    def add_chunks(self, chunks: Iterable[Chunk]) -> None:
        """Add chunks to the store.

        Args:
            chunks: Iterable of Chunk objects to add
        """
        pass

    @abstractmethod
    def load_chunks(self) -> List[Chunk]:
        """Load all chunks from the store.

        Returns:
            List of all stored Chunk objects
        """
        pass

    @abstractmethod
    def count_chunks(self) -> int:
        """Count total number of chunks in the store.

        Returns:
            Number of chunks
        """
        pass

    @abstractmethod
    def list_sources(self) -> List[str]:
        """List all unique source IDs in the store.

        Returns:
            Sorted list of unique source IDs
        """
        pass

    @abstractmethod
    def clear(self) -> None:
        """Delete all chunks from the store."""
        pass

    @abstractmethod
    def search(
        self,
        query_embedding: List[float],
        top_k: int,
        source_filter: Optional[str] = None,
    ) -> List[SearchResult]:
        """Search for similar chunks using vector similarity.

        Args:
            query_embedding: Query vector
            top_k: Number of results to return
            source_filter: Optional filter by source_id

        Returns:
            List of SearchResult objects sorted by score (descending)
        """
        pass

    @abstractmethod
    def delete_by_source(self, source_id: str) -> None:
        """Delete all chunks from a specific source.

        Args:
            source_id: Source ID to delete
        """
        pass
