import tempfile
import unittest
from pathlib import Path

from audio_rag.config import load_settings
from audio_rag.embeddings import HashingTextEmbedder
from audio_rag.service import AudioRAGService
from audio_rag.store import JsonlChunkStore


PODCAST_AUDIO_FILENAME = "podcast.mp3"
PODCAST_TRANSCRIPT_FILENAME = "podcast.transcript.txt"
QUESTION_AUDIO_FILENAME = "question.m4a"
QUESTION_TRANSCRIPT_FILENAME = "question.question.txt"
PODCAST_SOURCE_ID = "podcast-demo"
QUESTION_TEXT = "What was said about trustworthy question answering?"
PODCAST_TEXT = (
    "The speaker says retrieval augmented generation becomes more trustworthy when answers are grounded in retrieved evidence. "
    "The episode also argues that citations make question answering easier to validate."
)


class AudioRAGAudioWorkflowTest(unittest.TestCase):
    def test_ingest_podcast_and_ask_audio_via_sidecar_transcripts(self) -> None:
        settings = load_settings()
        with tempfile.TemporaryDirectory() as tmp_dir:
            temp_path = Path(tmp_dir)
            store_path = temp_path / "chunks.jsonl"
            podcast_audio_path = temp_path / PODCAST_AUDIO_FILENAME
            podcast_transcript_path = temp_path / PODCAST_TRANSCRIPT_FILENAME
            question_audio_path = temp_path / QUESTION_AUDIO_FILENAME
            question_transcript_path = temp_path / QUESTION_TRANSCRIPT_FILENAME

            podcast_audio_path.write_bytes(b"")
            podcast_transcript_path.write_text(PODCAST_TEXT, encoding="utf-8")
            question_audio_path.write_bytes(b"")
            question_transcript_path.write_text(QUESTION_TEXT, encoding="utf-8")

            service = AudioRAGService(
                store=JsonlChunkStore(store_path),
                embedder=HashingTextEmbedder(settings.embedding),
                settings=settings,
            )
            chunks = service.ingest_podcast(
                source_id=PODCAST_SOURCE_ID,
                audio_path=podcast_audio_path,
                transcript_path=podcast_transcript_path,
                metadata={"lang": "en"},
            )
            answer = service.ask_audio(
                question_audio_path=question_audio_path,
                transcript_path=question_transcript_path,
                top_k=3,
            )

            self.assertGreaterEqual(len(chunks), 1)
            self.assertEqual(answer.resolved_query_text, QUESTION_TEXT)
            self.assertGreaterEqual(len(answer.citations), 1)
            self.assertEqual(answer.citations[0].source_id, PODCAST_SOURCE_ID)
            self.assertEqual(answer.citations[0].audio_path, str(podcast_audio_path.resolve()))


if __name__ == "__main__":
    unittest.main()
