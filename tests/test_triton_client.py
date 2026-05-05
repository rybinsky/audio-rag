import json
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from audio_rag.config import load_settings
from audio_rag.triton_client import TritonHttpClient


class TritonClientPayloadTest(unittest.TestCase):
    @patch("audio_rag.triton_client.request.urlopen")
    def test_triton_ingest_podcast_maps_project_path_to_workspace(self, mock_urlopen: MagicMock) -> None:
        settings = load_settings()
        client = TritonHttpClient(settings)
        response_payload = {
            "outputs": [
                {
                    "data": [json.dumps({"source_id": "podcast", "indexed_chunks": 3})],
                }
            ]
        }
        mock_response = MagicMock()
        mock_response.read.return_value = json.dumps(response_payload).encode("utf-8")
        mock_urlopen.return_value.__enter__.return_value = mock_response

        result = client.ingest_podcast(
            source_id="podcast",
            audio_file=Path("tests/Подкаст.mp3"),
            metadata={"lang": "en"},
        )

        self.assertEqual(result["source_id"], "podcast")
        request_object = mock_urlopen.call_args[0][0]
        payload = json.loads(request_object.data.decode("utf-8"))
        mapped_audio_path = payload["inputs"][1]["data"][0]
        self.assertTrue(mapped_audio_path.startswith("/workspace/tests/"))

    @patch("audio_rag.triton_client.request.urlopen")
    def test_triton_ask_audio_allows_missing_transcript(self, mock_urlopen: MagicMock) -> None:
        settings = load_settings()
        client = TritonHttpClient(settings)
        response_payload = {
            "outputs": [
                {
                    "data": [json.dumps({"answer": "ok", "resolved_query_text": "hello", "citations": []})],
                }
            ]
        }
        mock_response = MagicMock()
        mock_response.read.return_value = json.dumps(response_payload).encode("utf-8")
        mock_urlopen.return_value.__enter__.return_value = mock_response

        result = client.ask_audio(question_audio_file=Path("tests/Вопрос.m4a"))

        self.assertEqual(result["answer"], "ok")
        self.assertEqual(result["resolved_query_text"], "hello")


if __name__ == "__main__":
    unittest.main()
