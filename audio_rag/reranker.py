"""Search reranker using sentence-transformers library."""

import time
from typing import List, Optional, Tuple

from sentence_transformers import CrossEncoder

from .settings import RerankerSettings
from .utils.logging import get_logger, log_model_loading, log_model_loaded


class SearchReranker:
    """Search reranker using BGE reranker model.

    This is a local implementation of reranking that loads the model directly,
    suitable for use within Triton server models.
    """

    def __init__(self, settings: RerankerSettings) -> None:
        """Initialize reranker.

        Args:
            settings: Reranker settings containing model configuration
        """
        self._settings = settings
        self._logger = get_logger(__name__)

        # Log model loading
        log_model_loading(self._logger, settings.model_name, None)
        start_time = time.time()

        self._reranker = CrossEncoder(
            settings.model_name,
            max_length=settings.max_length,
            device=settings.device,
        )

        # Log successful loading
        load_time = time.time() - start_time
        log_model_loaded(self._logger, settings.model_name, load_time)

    def rerank_texts(
        self,
        query: str,
        texts: List[str],
        top_k: Optional[int] = None,
    ) -> List[Tuple[int, float]]:
        """Rerank texts based on relevance to query.

        Args:
            query: The search query
            texts: List of texts to rerank
            top_k: Number of top results to return (optional)

        Returns:
            List of (original_index, score) tuples sorted by score descending
        """
        if not texts:
            return []

        # Log reranking request
        self._logger.debug(f"Reranking {len(texts)} texts for query: {query[:50]}...")

        start_time = time.time()

        # Create query-text pairs for reranker
        pairs = [[query, text] for text in texts]

        # Get scores from reranker
        scores = self._reranker.predict(pairs)

        # Log reranking time
        rerank_time = time.time() - start_time
        self._logger.debug(f"Reranked {len(texts)} texts in {rerank_time:.3f}s")

        # Handle single text case (predict returns float instead of list)
        if isinstance(scores, float):
            scores = [scores]

        # Create list of (index, score) tuples
        indexed_scores = list(enumerate(scores))

        # Sort by score descending
        indexed_scores.sort(key=lambda x: x[1], reverse=True)

        # Apply top_k limit if specified
        if top_k is not None and top_k > 0:
            indexed_scores = indexed_scores[:top_k]

        return indexed_scores

    def rerank(
        self,
        query: str,
        results: List["SearchResult"],
        top_k: Optional[int] = None,
    ) -> List["SearchResult"]:
        """Rerank search results based on query relevance.

        Args:
            query: The search query
            results: List of SearchResult objects to rerank
            top_k: Number of results to return (optional)

        Returns:
            Re-ranked list of SearchResult objects sorted by relevance
        """
        if not results:
            return []

        # Import here to avoid circular dependency
        from .models import SearchResult

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
