"""Base reranker interface to avoid tritonclient dependency in service.py."""

from abc import ABC, abstractmethod
from typing import List, Optional, Tuple


class BaseReranker(ABC):
    """Abstract base class for search result rerankers.

    This interface allows service.py to work with different reranker
    implementations without importing tritonclient.
    """

    @abstractmethod
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
        pass

    @abstractmethod
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
        pass
