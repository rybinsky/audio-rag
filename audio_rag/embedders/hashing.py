"""Hashing-based text embedder for deterministic embeddings."""

import hashlib
from typing import List

from .base import BaseEmbedder


class HashingTextEmbedder(BaseEmbedder):
    """Simple embedder that creates deterministic embeddings using hashing.

    This embedder creates fixed-size embeddings by hashing the input text
    and converting the hash to a normalized vector. Useful for testing,
    caching, or scenarios where semantic understanding is not required.
    """

    def __init__(self, embedding_dim: int = 1024) -> None:
        """Initialize hashing embedder.

        Args:
            embedding_dim: Dimension of the output embedding vectors
        """
        self._embedding_dim = embedding_dim

    def encode(self, text: str) -> List[float]:
        """Encode a single text to embedding vector using hashing.

        Args:
            text: Text to encode

        Returns:
            Embedding vector as list of floats
        """
        # Create hash of the text
        hash_bytes = hashlib.sha256(text.encode('utf-8')).digest()

        # Expand hash to fill the embedding dimension
        # Use multiple iterations of hashing to generate enough bytes
        embedding = []
        hash_input = text.encode('utf-8')

        while len(embedding) < self._embedding_dim:
            # Generate more hash bytes
            hash_bytes = hashlib.sha256(hash_input).digest()
            # Convert bytes to float values in range [-1, 1]
            for i in range(0, len(hash_bytes) - 1, 2):
                if len(embedding) >= self._embedding_dim:
                    break
                # Combine two bytes into a float value
                value = (hash_bytes[i] - 128) / 128.0
                embedding.append(float(value))
            # Update hash input for next iteration
            hash_input = hash_bytes + hash_input

        # Normalize the vector
        norm = sum(x * x for x in embedding) ** 0.5
        if norm > 0:
            embedding = [x / norm for x in embedding]

        return embedding[:self._embedding_dim]

    def encode_batch(self, texts: List[str]) -> List[List[float]]:
        """Encode multiple texts to embedding vectors.

        Args:
            texts: List of texts to encode

        Returns:
            List of embedding vectors
        """
        return [self.encode(text) for text in texts]

    def get_embedding_dimension(self) -> int:
        """Get the dimension of embedding vectors.

        Returns:
            Dimension of embedding vectors
        """
        return self._embedding_dim
