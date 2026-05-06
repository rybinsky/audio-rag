"""BGE-M3 embedder using sentence-transformers library."""

from typing import List

from sentence_transformers import SentenceTransformer

from ..settings import BGESettings
from .base import BaseEmbedder


class BGEEmbedder(BaseEmbedder):
    """BGE-M3 embedder using sentence-transformers library.

    This is a local implementation of BGE-M3 embeddings that loads the model
    directly, suitable for use within Triton server models.
    """

    def __init__(self, settings: BGESettings) -> None:
        """Initialize BGE-M3 embedder.

        Args:
            settings: BGE settings containing model configuration
        """
        self._settings = settings
        self._model = SentenceTransformer(
            settings.model_name,
            device=settings.device,
        )
        self._max_length = settings.max_length

    def encode(self, text: str) -> List[float]:
        """Encode a single text to embedding vector.

        Args:
            text: Text to encode

        Returns:
            Embedding vector as list of floats
        """
        embeddings = self.encode_batch([text])
        return embeddings[0]

    def encode_batch(self, texts: List[str]) -> List[List[float]]:
        """Encode multiple texts to embedding vectors.

        Args:
            texts: List of texts to encode

        Returns:
            List of embedding vectors
        """
        if not texts:
            return []

        # Generate embeddings using sentence-transformers
        embeddings = self._model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=self._settings.normalize_embeddings,
        )

        # Convert to list format
        return embeddings.tolist()

    def get_embedding_dimension(self) -> int:
        """Get the dimension of the embedding vectors.

        Returns:
            Dimension of embedding vectors (1024 for BGE-M3)
        """
        return 1024  # BGE-M3 uses 1024-dimensional embeddings
