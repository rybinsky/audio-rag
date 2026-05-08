"""Reranker that calls reranker through Triton Inference Server."""

from typing import List, Optional

import numpy as np
import tritonclient.http as httpclient

from .models import SearchResult
from .settings import TritonEmbedderSettings


class TritonReranker:
    """Reranker that calls reranker model through Triton Inference Server.

    This implementation uses Triton Inference Server to rerank search results,
    allowing for better resource management and model sharing across processes.
    """

    def __init__(
        self,
        settings: TritonEmbedderSettings,
        triton_url: str = "localhost:8000",
    ) -> None:
        """Initialize Triton reranker.

        Args:
            settings: Triton embedder settings (contains reranker_model_name)
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
        self._model_name = getattr(settings, "reranker_model_name", "reranker")

    def rerank_texts(
        self,
        query: str,
        texts: List[str],
        top_k: Optional[int] = None,
    ) -> List[tuple[int, float]]:
        """Rerank texts and return indices with scores.

        Args:
            query: The search query
            texts: List of texts to rerank
            top_k: Number of results to return

        Returns:
            List of (original_index, score) tuples sorted by score
        """
        if not texts:
            return []

        # Prepare input tensors
        query_tensor = httpclient.InferInput("INPUT_QUERY", [1], "BYTES")
        query_tensor.set_data_from_numpy(np.array([query.encode("utf-8")], dtype=object))

        texts_tensor = httpclient.InferInput("INPUT_TEXTS", [len(texts)], "BYTES")
        texts_array = np.array([t.encode("utf-8") for t in texts], dtype=object)
        texts_tensor.set_data_from_numpy(texts_array)

        inputs = [query_tensor, texts_tensor]

        # Add top_k if provided
        if top_k is not None:
            top_k_tensor = httpclient.InferInput("INPUT_TOP_K", [1], "INT32")
            top_k_tensor.set_data_from_numpy(np.array([top_k], dtype=np.int32))
            inputs.append(top_k_tensor)

        # Call Triton model
        result = self._client.infer(
            model_name=self._model_name,
            inputs=inputs,
        )

        # Get output indices and scores
        indices = result.as_numpy("OUTPUT_INDICES")
        scores = result.as_numpy("OUTPUT_SCORES")

        # Convert to list of tuples
        return [(int(idx), float(score)) for idx, score in zip(indices, scores)]

    def rerank(
        self,
        query: str,
        results: List[SearchResult],
        top_k: Optional[int] = None,
    ) -> List[SearchResult]:
        """Rerank search results based on query relevance.

        Args:
            query: The search query
            results: List of search results to rerank
            top_k: Number of results to return

        Returns:
            Re-ranked list of SearchResult objects sorted by relevance
        """
        if not results:
            return []

        # Extract texts from results
        texts = [result.chunk.text for result in results]

        # Get reranked indices and scores
        reranked = self.rerank_texts(query, texts, top_k)

        # Create new SearchResult objects with reranked scores
        reranked_results = []
        for original_index, score in reranked:
            original_result = results[original_index]
            reranked_results.append(
                SearchResult(
                    chunk=original_result.chunk,
                    score=float(score),
                )
            )

        return reranked_results
