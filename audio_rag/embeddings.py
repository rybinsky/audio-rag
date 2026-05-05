import hashlib
import math
import re
from typing import List

from .settings import EmbeddingSettings


class HashingTextEmbedder:
    """Deterministic lightweight embedder for local MVP development."""

    def __init__(self, settings: EmbeddingSettings) -> None:
        if settings.dimension <= 0:
            raise ValueError("dimension must be > 0")
        self._dimension = settings.dimension
        self._token_pattern = re.compile(settings.token_pattern)

    def encode(self, text: str) -> List[float]:
        vector = [0.0] * self._dimension
        for token in self._tokenize(text):
            index = self._bucket_index(token)
            vector[index] += 1.0
        return self._normalize(vector)

    @staticmethod
    def cosine_similarity(left: List[float], right: List[float]) -> float:
        if len(left) != len(right):
            raise ValueError("Vectors must have the same length")
        return sum(a * b for a, b in zip(left, right))

    def _bucket_index(self, token: str) -> int:
        digest = hashlib.sha256(token.encode("utf-8")).digest()
        return int.from_bytes(digest[:8], byteorder="big") % self._dimension

    @staticmethod
    def _normalize(vector: List[float]) -> List[float]:
        norm = math.sqrt(sum(value * value for value in vector))
        if norm == 0.0:
            return vector
        return [value / norm for value in vector]

    def _tokenize(self, text: str) -> List[str]:
        return [match.group(0).lower() for match in self._token_pattern.finditer(text)]
