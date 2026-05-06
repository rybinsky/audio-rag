import tempfile
import unittest
from pathlib import Path

from audio_rag.config import load_settings
from audio_rag.embedders import BGEEmbedder
from audio_rag.reranker import SearchReranker
from audio_rag.service import AudioRAGService
from audio_rag.stores import QdrantChunkStore


SAMPLE_TRANSCRIPT = (
    "Scaling laws help estimate how model quality changes with more compute and data. "
    "Karpathy also said that data quality matters as much as raw scale for useful systems."
)
EMPTY_INDEX_ANSWER_PREFIX = "Не нашёл релевантных фрагментов"
TEST_SOURCE_ID = "unit-test-source"
SEARCH_QUERY = "What did Karpathy say about scaling laws?"


class AudioRAGMVPTest(unittest.TestCase):
    def test_ingest_and_ask_returns_citations(self) -> None:
        settings = load_settings()
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Use Qdrant for vector storage, BGE-M3 for embeddings, and Reranker for better results
            # Note: Qdrant must be running for tests to work
            service = AudioRAGService(
                store=QdrantChunkStore(settings.qdrant),
                embedder=BGEEmbedder(settings.bge),
                reranker=SearchReranker(settings.reranker),
                settings=settings,
            )

            chunks = service.ingest_transcript(
                source_id=TEST_SOURCE_ID,
                transcript=SAMPLE_TRANSCRIPT,
                metadata={"lang": "en"},
            )
            answer = service.ask(SEARCH_QUERY, top_k=3)

            self.assertGreaterEqual(len(chunks), 1)
            self.assertTrue(answer.answer)
            self.assertGreaterEqual(len(answer.citations), 1)
            self.assertEqual(answer.citations[0].source_id, TEST_SOURCE_ID)

    def test_empty_index_returns_fallback_answer(self) -> None:
        settings = load_settings()
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Use Qdrant for vector storage, BGE-M3 for embeddings, and Reranker for better results
            # Note: Qdrant must be running for tests to work
            service = AudioRAGService(
                store=QdrantChunkStore(settings.qdrant),
                embedder=BGEEmbedder(settings.bge),
                reranker=SearchReranker(settings.reranker),
                settings=settings,
            )

            answer = service.ask("any query", top_k=2)

            self.assertTrue(answer.answer.startswith(EMPTY_INDEX_ANSWER_PREFIX))
            self.assertEqual(answer.citations, [])


if __name__ == "__main__":
    unittest.main()
