"""BGE-M3 embedder that calls BGE through Triton Inference Server."""

from typing import List

import numpy as np
import tritonclient.http as httpclient

from ..settings import TritonEmbedderSettings
from .base import BaseEmbedder


class TritonBGEEmbedder(BaseEmbedder):
    """BGE-M3 embedder that calls BGE model through Triton Inference Server.

    This implementation uses Triton Inference Server to generate embeddings,
    allowing for better resource management and model sharing across processes.
    """

    def __init__(
        self,
        settings: TritonEmbedderSettings,
        triton_url: str = "localhost:8000",
    ) -> None:
        """Initialize Triton BGE embedder.

        Args:
            settings: Triton embedder settings
            triton_url: Triton server URL (default: localhost:8000)
        """
        self._settings = settings
        self._triton_url = triton_url

        # Remove scheme from URL if present (tritonclient doesn't accept http://)
        if triton_url.startswith("http://"):
            triton_url = triton_url[7:]  # Remove "http://"
        elif triton_url.startswith("https://"):
            triton_url = triton_url[8:]  # Remove "https://"

        self._client = httpclient.InferenceServerClient(url=triton_url)
        self._model_name = settings.bge_model_name

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

        # Prepare input tensor
        input_texts = np.array(texts, dtype=object)
        input_tensor = httpclient.InferInput("INPUT_TEXT", input_texts.shape, "BYTES")
        input_tensor.set_data_from_numpy(input_texts)

        # Call Triton model
        result = self._client.infer(
            model_name=self._model_name,
            inputs=[input_tensor],
        )

        # Get output embedding
        embeddings = result.as_numpy("OUTPUT_EMBEDDING")

        # Convert to list format
        return embeddings.tolist()

    def get_embedding_dimension(self) -> int:
        """Get the dimension of the embedding vectors.

        Returns:
            Dimension of embedding vectors (1024 for BGE-M3)
        """
        return 1024  # BGE-M3 uses 1024-dimensional embeddings
