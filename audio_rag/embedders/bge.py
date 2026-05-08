"""BGE-M3 embedder using sentence-transformers library."""

import time
from pathlib import Path
from typing import List

from sentence_transformers import SentenceTransformer

from ..settings import BGESettings
from ..utils.logging import get_logger, log_model_loading, log_model_loaded
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
        self._logger = get_logger(__name__)

        # Log model loading
        log_model_loading(self._logger, settings.model_name, None)
        start_time = time.time()

        self._model = SentenceTransformer(
            settings.model_name,
            device=settings.device,
        )
        self._max_length = settings.max_length

        # Log successful loading
        load_time = time.time() - start_time
        log_model_loaded(self._logger, settings.model_name, load_time)

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

        # Log encoding request
        self._logger.debug(f"Encoding {len(texts)} texts with {self._settings.model_name}")

        start_time = time.time()

        # Generate embeddings using sentence-transformers
        embeddings = self._model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=self._settings.normalize_embeddings,
        )

        # Log encoding time
        encode_time = time.time() - start_time
        self._logger.debug(f"Encoded {len(texts)} texts in {encode_time:.3f}s")

        # Convert to list format
        return embeddings.tolist()

    def get_embedding_dimension(self) -> int:
        """Get the dimension of the embedding vectors.

        Returns:
            Dimension of embedding vectors (1024 for BGE-M3)
        """
        return 1024  # BGE-M3 uses 1024-dimensional embeddings
