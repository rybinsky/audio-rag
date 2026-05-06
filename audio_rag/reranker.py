"""Search reranker using sentence-transformers library."""

from typing import List, Optional, Tuple

from sentence_transformers import CrossEncoder

from .settings import RerankerSettings


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
        self._reranker = CrossEncoder(
            settings.model_name,
            max_length=settings.max_length,
            device=settings.device,
        )

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

        # Create query-text pairs for reranker
        pairs = [[query, text] for text in texts]

        # Get scores from reranker
        scores = self._reranker.predict(pairs)

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
