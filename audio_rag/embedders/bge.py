"""BGE-M3 embedder using FlagEmbedding library."""

from typing import List

from FlagEmbedding import BGEM3FlagEmbedder

from ..settings import BGESettings
from .base import BaseEmbedder


class BGEEmbedder(BaseEmbedder):
    """BGE-M3 embedder using FlagEmbedding library.

    This is a local implementation of BGE-M3 embeddings that loads the model
    directly, suitable for use within Triton server models.
    """

    def __init__(self, settings: BGESettings) -> None:
        """Initialize BGE-M3 embedder.

        Args:
            settings: BGE settings containing model configuration
        """
        self._settings = settings
        self._model = BGEM3FlagEmbedder(
            model_name=settings.model_name,
            device=settings.device,
            use_fp16=settings.device == "cuda",
        )
        self._max_length = settings.max_length
        self._batch_size = settings.batch_size

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

        # BGEM3FlagEmbedder returns a dict with 'dense_vecs' key
        result = self._model.encode(
            texts,
            batch_size=self._batch_size,
            max_length=self._max_length,
        )

        # Extract dense embeddings and convert to list format
        embeddings = result["dense_vecs"].tolist()

        return embeddings

    def get_embedding_dimension(self) -> int:
        """Get the dimension of the embedding vectors.

        Returns:
            Dimension of embedding vectors (1024 for BGE-M3)
        """
        return 1024  # BGE-M3 uses 1024-dimensional embeddings
