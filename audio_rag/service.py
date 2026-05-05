import uuid
from pathlib import Path
from typing import Dict, List, Optional

from .chunking import chunk_text
from .embeddings import HashingTextEmbedder
from .models import Chunk, Citation, QueryAnswer, SearchResult
from .settings import AppSettings
from .store import JsonlChunkStore


class AudioRAGService:
    """Transcript-first service layer for ingest and retrieval."""

    def __init__(self, store: JsonlChunkStore, embedder: HashingTextEmbedder, settings: AppSettings) -> None:
        self._store = store
        self._embedder = embedder
        self._settings = settings

    @property
    def store(self) -> JsonlChunkStore:
        return self._store

    def ingest_transcript(
        self,
        *,
        source_id: str,
        transcript: str,
        metadata: Optional[Dict] = None,
    ) -> List[Chunk]:
        cleaned_source_id = source_id.strip()
        cleaned_transcript = transcript.strip()
        if not cleaned_source_id:
            raise ValueError("source_id must not be empty")
        if not cleaned_transcript:
            raise ValueError("transcript must not be empty")

        normalized_metadata = dict(metadata or {})
        normalized_metadata.setdefault(
            self._settings.metadata.ingest_mode_key,
            self._settings.metadata.ingest_mode_transcript,
        )

        chunks: List[Chunk] = []
        for text, start_offset, end_offset in chunk_text(cleaned_transcript, self._settings.chunking):
            chunk_id = f"{cleaned_source_id}-{uuid.uuid4().hex[:8]}"
            chunks.append(
                Chunk(
                    chunk_id=chunk_id,
                    source_id=cleaned_source_id,
                    text=text,
                    start_offset=start_offset,
                    end_offset=end_offset,
                    metadata=normalized_metadata,
                    embedding=self._embedder.encode(text),
                )
            )

        self._store.add_chunks(chunks)
        return chunks

    def ingest_podcast(
        self,
        *,
        source_id: str,
        audio_path: Path,
        transcript_path: Optional[Path] = None,
        metadata: Optional[Dict] = None,
    ) -> List[Chunk]:
        resolved_audio_path = audio_path.expanduser().resolve()
        if not resolved_audio_path.exists():
            raise FileNotFoundError(f"audio file not found: {resolved_audio_path}")

        resolved_transcript_path = self._resolve_sidecar_path(
            file_path=resolved_audio_path,
            explicit_path=transcript_path,
            suffix=self._settings.transcript.podcast_suffix,
        )
        transcript = resolved_transcript_path.read_text(encoding=self._settings.transcript.encoding)
        normalized_metadata = dict(metadata or {})
        normalized_metadata[self._settings.metadata.audio_path_key] = str(resolved_audio_path)
        normalized_metadata[self._settings.metadata.transcript_path_key] = str(resolved_transcript_path)
        normalized_metadata[self._settings.metadata.ingest_mode_key] = self._settings.metadata.ingest_mode_podcast
        return self.ingest_transcript(source_id=source_id, transcript=transcript, metadata=normalized_metadata)

    def ask(self, query: str, *, top_k: Optional[int] = None) -> QueryAnswer:
        cleaned_query = query.strip()
        if not cleaned_query:
            raise ValueError("query must not be empty")
        effective_top_k = top_k or self._settings.retrieval.default_top_k
        if effective_top_k <= 0:
            raise ValueError("top_k must be > 0")

        query_embedding = self._embedder.encode(cleaned_query)
        results = self._store.search(query_embedding, top_k=effective_top_k)
        if not results:
            return QueryAnswer(
                answer=self._settings.retrieval.no_results_answer,
                citations=[],
                resolved_query_text=cleaned_query,
            )

        answer = self._build_answer(cleaned_query, results)
        citations = [self._to_citation(result) for result in results]
        return QueryAnswer(answer=answer, citations=citations, resolved_query_text=cleaned_query)

    def ask_audio(
        self,
        *,
        question_audio_path: Path,
        transcript_path: Optional[Path] = None,
        top_k: Optional[int] = None,
    ) -> QueryAnswer:
        resolved_audio_path = question_audio_path.expanduser().resolve()
        if not resolved_audio_path.exists():
            raise FileNotFoundError(f"question audio file not found: {resolved_audio_path}")

        resolved_transcript_path = self._resolve_sidecar_path(
            file_path=resolved_audio_path,
            explicit_path=transcript_path,
            suffix=self._settings.transcript.question_suffix,
        )
        question_text = resolved_transcript_path.read_text(encoding=self._settings.transcript.encoding).strip()
        answer = self.ask(question_text, top_k=top_k)
        answer.resolved_query_text = question_text
        return answer

    def _build_answer(self, query: str, results: List[SearchResult]) -> str:
        snippets = []
        for index, result in enumerate(results, start=1):
            snippet = self._shorten(result.chunk.text)
            snippets.append(f"[{index}] {snippet}")
        joined_sources = "; ".join(snippets)
        source_names = ", ".join(sorted({result.chunk.source_id for result in results}))
        prefix = self._settings.retrieval.source_preview_prefix
        return f"По локальному индексу лучший контекст для запроса '{query}': {joined_sources}. {prefix}: {source_names}."

    def _shorten(self, text: str) -> str:
        words = text.split()
        if len(words) <= self._settings.retrieval.max_preview_words:
            return text
        return " ".join(words[: self._settings.retrieval.max_preview_words]) + " …"

    def _to_citation(self, result: SearchResult) -> Citation:
        audio_key = self._settings.metadata.audio_path_key
        return Citation(
            chunk_id=result.chunk.chunk_id,
            source_id=result.chunk.source_id,
            snippet=self._shorten(result.chunk.text),
            start_offset=result.chunk.start_offset,
            end_offset=result.chunk.end_offset,
            score=result.score,
            audio_path=str(result.chunk.metadata.get(audio_key, "")),
        )

    @staticmethod
    def _resolve_sidecar_path(file_path: Path, explicit_path: Optional[Path], suffix: str) -> Path:
        if explicit_path is not None:
            resolved_path = explicit_path.expanduser().resolve()
        else:
            resolved_path = file_path.with_suffix("")
            resolved_path = resolved_path.with_name(file_path.stem + suffix)
        if not resolved_path.exists():
            raise FileNotFoundError(f"transcript file not found: {resolved_path}")
        return resolved_path
