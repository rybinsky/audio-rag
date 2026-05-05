import json
from pathlib import Path
from typing import Dict, Iterable, List

from .models import Chunk, SearchResult


class JsonlChunkStore:
    def __init__(self, path: Path) -> None:
        self._path = path
        self._path.parent.mkdir(parents=True, exist_ok=True)

    @property
    def path(self) -> Path:
        return self._path

    def add_chunks(self, chunks: Iterable[Chunk]) -> None:
        with self._path.open("a", encoding="utf-8") as file:
            for chunk in chunks:
                file.write(json.dumps(self._serialize_chunk(chunk), ensure_ascii=False) + "\n")

    def load_chunks(self) -> List[Chunk]:
        if not self._path.exists():
            return []
        chunks: List[Chunk] = []
        with self._path.open("r", encoding="utf-8") as file:
            for line in file:
                payload = json.loads(line)
                chunks.append(self._deserialize_chunk(payload))
        return chunks

    def count_chunks(self) -> int:
        return len(self.load_chunks())

    def list_sources(self) -> List[str]:
        return sorted({chunk.source_id for chunk in self.load_chunks()})

    def clear(self) -> None:
        if self._path.exists():
            self._path.unlink()

    def search(self, query_embedding: List[float], top_k: int) -> List[SearchResult]:
        scored_results: List[SearchResult] = []
        for chunk in self.load_chunks():
            if not chunk.embedding:
                continue
            if len(query_embedding) != len(chunk.embedding):
                continue
            score = sum(left * right for left, right in zip(query_embedding, chunk.embedding))
            scored_results.append(SearchResult(chunk=chunk, score=score))
        scored_results.sort(key=lambda result: result.score, reverse=True)
        return scored_results[:top_k]

    @staticmethod
    def _serialize_chunk(chunk: Chunk) -> Dict:
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
        return Chunk(
            chunk_id=payload["chunk_id"],
            source_id=payload["source_id"],
            text=payload["text"],
            start_offset=payload["start_offset"],
            end_offset=payload["end_offset"],
            metadata=payload.get("metadata", {}),
            embedding=payload.get("embedding", []),
        )
