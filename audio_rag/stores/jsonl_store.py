"""JSONL-based chunk store for local development."""

import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from ..models import Chunk, SearchResult
from .base import BaseStore


class JsonlChunkStore(BaseStore):
    """JSONL-based chunk store for simple local storage."""

    def __init__(self, path: Path) -> None:
        """Initialize JSONL store.

        Args:
            path: Path to the JSONL file
        """
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)

    @property
    def path(self) -> str:
        """Return storage path.

        Returns:
            Path to the JSONL file as string
        """
        return str(self._path)

    def add_chunks(self, chunks: Iterable[Chunk]) -> None:
        """Add chunks to the JSONL store.

        Args:
            chunks: Iterable of Chunk objects to add
        """
        with self._path.open("a", encoding="utf-8") as file:
            for chunk in chunks:
                file.write(json.dumps(self._serialize_chunk(chunk), ensure_ascii=False) + "\n")

    def load_chunks(self) -> List[Chunk]:
        """Load all chunks from the JSONL store.

        Returns:
            List of all stored Chunk objects
        """
        if not self._path.exists():
            return []
        chunks: List[Chunk] = []
        with self._path.open("r", encoding="utf-8") as file:
            for line in file:
                payload = json.loads(line)
                chunks.append(self._deserialize_chunk(payload))
        return chunks

    def count_chunks(self) -> int:
        """Count total number of chunks.

        Returns:
            Number of chunks in the store
        """
        return len(self.load_chunks())

    def list_sources(self) -> List[str]:
        """List all unique source IDs.

        Returns:
            Sorted list of unique source IDs
        """
        return sorted({chunk.source_id for chunk in self.load_chunks()})

    def clear(self) -> None:
        """Delete all chunks from the store."""
        if self._path.exists():
            self._path.unlink()

    def search(
        self,
        query_embedding: List[float],
        top_k: int,
        source_filter: Optional[str] = None,
    ) -> List[SearchResult]:
        """Search for similar chunks using cosine similarity.

        Args:
            query_embedding: Query vector
            top_k: Number of results to return
            source_filter: Optional filter by source_id

        Returns:
            List of SearchResult objects sorted by score (descending)
        """
        scored_results: List[SearchResult] = []

        for chunk in self.load_chunks():
            # Apply source filter if provided
            if source_filter and chunk.source_id != source_filter:
                continue

            if not chunk.embedding:
                continue
            if len(query_embedding) != len(chunk.embedding):
                continue

            # Calculate cosine similarity
            score = sum(left * right for left, right in zip(query_embedding, chunk.embedding))
            scored_results.append(SearchResult(chunk=chunk, score=score))

        # Sort by score descending
        scored_results.sort(key=lambda result: result.score, reverse=True)
        return scored_results[:top_k]

    def delete_by_source(self, source_id: str) -> None:
        """Delete all chunks from a specific source.

        Args:
            source_id: Source ID to delete
        """
        # Load all chunks
        all_chunks = self.load_chunks()

        # Filter out chunks from the specified source
        remaining_chunks = [chunk for chunk in all_chunks if chunk.source_id != source_id]

        # Clear the store
        self.clear()

        # Re-add remaining chunks
        if remaining_chunks:
            self.add_chunks(remaining_chunks)

    @staticmethod
    def _serialize_chunk(chunk: Chunk) -> Dict:
        """Serialize a Chunk to a dictionary.

        Args:
            chunk: Chunk object to serialize

        Returns:
            Dictionary representation of the chunk
        """
        return {
            "chunk_id": chunk.chunk_id,
            "source_id": chunk.source_id,
            "text": chunk.text,
            "start_offset": chunk.start_offset,
            "end_offset": chunk.end_offset,
            "metadata": chunk.metadata,
            "embedding": chunk.embedding,
        }

    @staticmethod
    def _deserialize_chunk(payload: Dict) -> Chunk:
        """Deserialize a dictionary to a Chunk.

        Args:
            payload: Dictionary with chunk data

        Returns:
            Chunk object
        """
        return Chunk(
            chunk_id=payload["chunk_id"],
            source_id=payload["source_id"],
            text=payload["text"],
            start_offset=payload["start_offset"],
            end_offset=payload["end_offset"],
            metadata=payload.get("metadata", {}),
            embedding=payload.get("embedding", []),
        )
