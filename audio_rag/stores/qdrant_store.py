"""Qdrant-based vector store for audio RAG chunks."""

import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.models import Distance, VectorParams

from ..models import Chunk, SearchResult
from ..settings import QdrantSettings
from .base import BaseStore


class QdrantChunkStore(BaseStore):
    """Qdrant-based vector store for efficient similarity search."""

    def __init__(self, settings: QdrantSettings) -> None:
        self._settings = settings
        self._client = QdrantClient(
            host=settings.host,
            port=settings.port,
        )
        self._collection_ensured = False  # Lazy initialization

    @property
    def path(self) -> str:
        """Return connection string for compatibility with CLI."""
        return f"qdrant://{self._settings.host}:{self._settings.port}/{self._settings.collection_name}"

    def _ensure_collection(self) -> None:
        """Create collection if it doesn't exist.

        Uses lazy initialization and retry logic to handle connection issues.
        """
        if self._collection_ensured:
            return

        max_retries = 5
        retry_delay = 1.0

        for attempt in range(max_retries):
            try:
                collections = self._client.get_collections().collections
                collection_names = [c.name for c in collections]

                if self._settings.collection_name not in collection_names:
                    self._client.create_collection(
                        collection_name=self._settings.collection_name,
                        vectors_config=VectorParams(
                            size=self._settings.vector_size,
                            distance=Distance.COSINE,
                        ),
                    )
                self._collection_ensured = True
                return
            except Exception as e:
                if attempt == max_retries - 1:
                    raise
                time.sleep(retry_delay)
                retry_delay *= 2  # Exponential backoff

    def add_chunks(self, chunks: Iterable[Chunk]) -> None:
        """Add chunks to the Qdrant collection."""
        self._ensure_collection()  # Lazy initialization
        points = []
        for chunk in chunks:
            if not chunk.embedding:
                continue

            # Create point ID from chunk_id (use hash for numeric ID)
            point_id = abs(hash(chunk.chunk_id)) % (2**63)

            # Create payload with all chunk data
            payload = {
                "chunk_id": chunk.chunk_id,
                "source_id": chunk.source_id,
                "text": chunk.text,
                "start_offset": chunk.start_offset,
                "end_offset": chunk.end_offset,
                "metadata": chunk.metadata,
            }

            points.append(
                models.PointStruct(
                    id=point_id,
                    vector=chunk.embedding,
                    payload=payload,
                )
            )

        if points:
            self._client.upsert(
                collection_name=self._settings.collection_name,
                points=points,
            )

    def load_chunks(self) -> List[Chunk]:
        """Load all chunks from Qdrant (for compatibility, use sparingly)."""
        self._ensure_collection()  # Lazy initialization
        chunks = []

        # Scroll through all points
        offset = None
        while True:
            results, offset = self._client.scroll(
                collection_name=self._settings.collection_name,
                limit=100,
                offset=offset,
                with_payload=True,
                with_vectors=True,
            )

            for point in results:
                chunk = self._point_to_chunk(point)
                chunks.append(chunk)

            if offset is None:
                break

        return chunks

    def count_chunks(self) -> int:
        """Count total number of chunks in the collection."""
        self._ensure_collection()  # Lazy initialization
        result = self._client.count(collection_name=self._settings.collection_name)
        return result.count

    def list_sources(self) -> List[str]:
        """List all unique source IDs in the collection."""
        self._ensure_collection()  # Lazy initialization
        # Use scroll to get all unique source_ids
        sources = set()
        offset = None

        while True:
            results, offset = self._client.scroll(
                collection_name=self._settings.collection_name,
                limit=100,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )

            for point in results:
                if point.payload and "source_id" in point.payload:
                    sources.add(point.payload["source_id"])

            if offset is None:
                break

        return sorted(sources)

    def clear(self) -> None:
        """Delete all chunks from the collection."""
        self._ensure_collection()  # Ensure collection exists first
        # Delete the collection and recreate it
        self._client.delete_collection(collection_name=self._settings.collection_name)
        self._collection_ensured = False
        self._ensure_collection()

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
        self._ensure_collection()  # Lazy initialization

        # Build filter if source_filter is provided
        query_filter = None
        if source_filter:
            query_filter = models.Filter(
                must=[
                    models.FieldCondition(
                        key="source_id",
                        match=models.MatchValue(value=source_filter),
                    )
                ]
            )

        # Perform vector search using new query_points API
        response = self._client.query_points(
            collection_name=self._settings.collection_name,
            query=query_embedding,
            limit=top_k,
            query_filter=query_filter,
            with_payload=True,
            with_vectors=True,
        )

        # Convert to SearchResult objects
        search_results = []
        for point in response.points:
            chunk = self._point_to_chunk(point)
            score = point.score
            search_results.append(SearchResult(chunk=chunk, score=score))

        return search_results

    def delete_by_source(self, source_id: str) -> None:
        """Delete all chunks from a specific source.

        Args:
            source_id: Source ID to delete
        """
        self._ensure_collection()  # Lazy initialization
        self._client.delete(
            collection_name=self._settings.collection_name,
            points_selector=models.FilterSelector(
                filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="source_id",
                            match=models.MatchValue(value=source_id),
                        )
                    ]
                )
            ),
        )

    @staticmethod
    def _point_to_chunk(point: models.ScoredPoint) -> Chunk:
        """Convert a Qdrant point to a Chunk object."""
        payload = point.payload or {}

        # Get embedding from vector
        embedding = []
        if point.vector:
            if isinstance(point.vector, list):
                embedding = point.vector
            elif isinstance(point.vector, dict):
                # Named vectors case
                embedding = point.vector.get("", [])

        return Chunk(
            chunk_id=payload.get("chunk_id", ""),
            source_id=payload.get("source_id", ""),
            text=payload.get("text", ""),
            start_offset=payload.get("start_offset", 0),
            end_offset=payload.get("end_offset", 0),
            metadata=payload.get("metadata", {}),
            embedding=embedding,
        )
