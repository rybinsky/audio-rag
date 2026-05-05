from typing import List, Tuple

from .settings import ChunkingSettings


class ChunkingError(ValueError):
    """Raised when chunking parameters are invalid."""


ChunkBounds = Tuple[str, int, int]


def chunk_text(text: str, settings: ChunkingSettings) -> List[ChunkBounds]:
    normalized_text = text.strip()
    if not normalized_text:
        return []
    if settings.chunk_words <= 0:
        raise ChunkingError("chunk_words must be > 0")
    if settings.overlap_words < 0:
        raise ChunkingError("overlap_words must be >= 0")
    if settings.overlap_words >= settings.chunk_words:
        raise ChunkingError("overlap_words must be smaller than chunk_words")

    words = normalized_text.split()
    step = settings.chunk_words - settings.overlap_words
    chunks: List[ChunkBounds] = []

    for start in range(0, len(words), step):
        end = min(start + settings.chunk_words, len(words))
        chunk_words_slice = words[start:end]
        if not chunk_words_slice:
            break
        chunks.append((" ".join(chunk_words_slice), start, end))
        if end >= len(words):
            break

    return chunks
