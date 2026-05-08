"""Abstract base class for all embedders."""

from abc import ABC, abstractmethod
from typing import List


class BaseEmbedder(ABC):
    """Abstract base class for text and audio embedders.

    All embedder implementations must inherit from this class and implement
    the required abstract methods.
    """

    @abstractmethod
    def encode(self, text: str) -> List[float]:
        """Encode a single text to embedding vector.

        Args:
            text: Text to encode

        Returns:
            Embedding vector as list of floats
        """
        pass

    @abstractmethod
    def encode_batch(self, texts: List[str]) -> List[List[float]]:
        """Encode multiple texts to embedding vectors.

        Args:
            texts: List of texts to encode

        Returns:
            List of embedding vectors
        """
        pass

    @abstractmethod
    def get_embedding_dimension(self) -> int:
        """Get the dimension of embedding vectors.

        Returns:
            Dimension of embedding vectors
        """
        pass

    @staticmethod
    def cosine_similarity(left: List[float], right: List[float]) -> float:
        """Calculate cosine similarity between two vectors.

        Args:
            left: First vector
            right: Second vector

        Returns:
            Cosine similarity score between 0 and 1

        Raises:
            ValueError: If vectors have different lengths
        """
        if len(left) != len(right):
            raise ValueError("Vectors must have the same length")

        dot_product = sum(a * b for a, b in zip(left, right))
        norm_left = sum(a * a for a in left) ** 0.5
        norm_right = sum(b * b for b in right) ** 0.5

        if norm_left == 0 or norm_right == 0:
            return 0.0

        return dot_product / (norm_left * norm_right)
