from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class Chunk:
    chunk_id: str
    source_id: str
    text: str
    start_offset: int
    end_offset: int
    metadata: Dict[str, Any] = field(default_factory=dict)
    embedding: List[float] = field(default_factory=list)


@dataclass
class SearchResult:
    chunk: Chunk
    score: float


@dataclass
class Citation:
    chunk_id: str
    source_id: str
    snippet: str
    start_offset: int
    end_offset: int
    score: float
    audio_path: str = ""


@dataclass
class QueryAnswer:
    answer: str
    citations: List[Citation]
    resolved_query_text: str = ""
