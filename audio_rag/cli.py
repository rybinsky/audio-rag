import argparse
import json
from pathlib import Path
from typing import Any, Dict

from .config import load_settings
from .factories import create_embedder, create_reranker, create_store
from .service import AudioRAGService
from .triton_client import TritonHttpClient


def build_parser(settings) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Audio RAG CLI")
    parser.add_argument("--store", type=Path, default=None, help="Path to the local JSONL store")
    parser.add_argument("--triton-url", default=settings.triton_http.base_url, help="Base HTTP URL for Triton inference")

    subparsers = parser.add_subparsers(dest="command", required=True)

    ingest_parser = subparsers.add_parser("ingest-text", help="Index a plain-text transcript")
    ingest_parser.add_argument("--source", required=True, help="Stable source identifier")
    ingest_parser.add_argument("--file", type=Path, required=True, help="Path to a UTF-8 transcript file")
    ingest_parser.add_argument("--metadata-json", default="{}", help="JSON object with extra source metadata")

    ingest_podcast_parser = subparsers.add_parser("ingest-podcast", help="Index a podcast audio file using a sidecar transcript")
    ingest_podcast_parser.add_argument("--source", required=True, help="Stable source identifier")
    ingest_podcast_parser.add_argument("--audio-file", type=Path, required=True, help="Path to the source audio file")
    ingest_podcast_parser.add_argument("--transcript-file", type=Path, default=None, help="Optional transcript path")
    ingest_podcast_parser.add_argument("--metadata-json", default="{}", help="JSON object with extra source metadata")

    triton_ingest_parser = subparsers.add_parser("triton-ingest-podcast", help="Index a podcast audio file through Triton")
    triton_ingest_parser.add_argument("--source", required=True, help="Stable source identifier")
    triton_ingest_parser.add_argument("--audio-file", type=Path, required=True, help="Path to the source audio file")
    triton_ingest_parser.add_argument("--transcript-file", type=Path, default=None, help="Optional transcript path")
    triton_ingest_parser.add_argument("--metadata-json", default="{}", help="JSON object with extra source metadata")

    ask_parser = subparsers.add_parser("ask", help="Query the local transcript index")
    ask_parser.add_argument("query", help="User query")
    ask_parser.add_argument("--top-k", type=int, default=settings.retrieval.default_top_k, help="Number of citations to return")

    triton_ask_parser = subparsers.add_parser("triton-ask", help="Send a text query through Triton")
    triton_ask_parser.add_argument("query", help="User query")
    triton_ask_parser.add_argument("--top-k", type=int, default=settings.retrieval.default_top_k, help="Number of citations to return")

    ask_audio_parser = subparsers.add_parser("ask-audio", help="Ask a question from an audio file using a sidecar transcript")
    ask_audio_parser.add_argument("--question-audio-file", type=Path, required=True, help="Path to recorded question audio")
    ask_audio_parser.add_argument("--question-transcript-file", type=Path, default=None, help="Optional transcript path")
    ask_audio_parser.add_argument("--top-k", type=int, default=settings.retrieval.default_top_k, help="Number of citations to return")

    triton_ask_audio_parser = subparsers.add_parser("triton-ask-audio", help="Ask a question from an audio file through Triton")
    triton_ask_audio_parser.add_argument("--question-audio-file", type=Path, required=True, help="Path to recorded question audio")
    triton_ask_audio_parser.add_argument("--question-transcript-file", type=Path, default=None, help="Optional transcript path")
    triton_ask_audio_parser.add_argument("--top-k", type=int, default=settings.retrieval.default_top_k, help="Number of citations to return")

    subparsers.add_parser("stats", help="Show local index stats")
    subparsers.add_parser("reset-store", help="Delete the local JSONL store")
    return parser


def main() -> None:
    settings = load_settings()
    parser = build_parser(settings)
    args = parser.parse_args()
    service = _build_service(settings, args.store, triton_url=args.triton_url)
    triton_client = TritonHttpClient(settings)

    if args.command == "ingest-text":
        metadata = _parse_metadata(args.metadata_json)
        transcript = args.file.read_text(encoding=settings.transcript.encoding)
        chunks = service.ingest_transcript(source_id=args.source, transcript=transcript, metadata=metadata)
        print(f"Indexed chunks: {len(chunks)}")
        print(f"Store: {service.store.path}")
        return

    if args.command == "ingest-podcast":
        metadata = _parse_metadata(args.metadata_json)
        chunks = service.ingest_podcast(
            source_id=args.source,
            audio_path=args.audio_file,
            transcript_path=args.transcript_file,
            metadata=metadata,
        )
        print(f"Indexed podcast chunks: {len(chunks)}")
        print(f"Audio: {args.audio_file}")
        print(f"Store: {service.store.path}")
        return

    if args.command == "triton-ingest-podcast":
        metadata = _parse_metadata(args.metadata_json)
        result = triton_client.ingest_podcast(
            source_id=args.source,
            audio_file=args.audio_file,
            transcript_file=args.transcript_file,
            metadata=metadata,
            base_url=args.triton_url,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return

    if args.command == "ask":
        answer = service.ask(args.query, top_k=args.top_k)
        _print_answer(answer)
        return

    if args.command == "triton-ask":
        result = triton_client.ask(query_text=args.query, top_k=args.top_k, base_url=args.triton_url)
        _print_triton_result(result)
        return

    if args.command == "ask-audio":
        answer = service.ask_audio(
            question_audio_path=args.question_audio_file,
            transcript_path=args.question_transcript_file,
            top_k=args.top_k,
        )
        print(f"Resolved question transcript: {answer.resolved_query_text}")
        _print_answer(answer)
        return

    if args.command == "triton-ask-audio":
        result = triton_client.ask_audio(
            question_audio_file=args.question_audio_file,
            question_transcript_file=args.question_transcript_file,
            top_k=args.top_k,
            base_url=args.triton_url,
        )
        _print_triton_result(result)
        return

    if args.command == "stats":
        print(f"Store: {service.store.path}")
        print(f"Chunks: {service.store.count_chunks()}")
        sources = service.store.list_sources()
        print(f"Sources: {', '.join(sources) if sources else '-'}")
        return

    if args.command == "reset-store":
        service.store.clear()
        print(f"Store reset: {service.store.path}")
        return

    parser.error(f"Unsupported command: {args.command}")


def _build_service(settings, explicit_store_path: Path = None, triton_url: str = "localhost:8000") -> AudioRAGService:
    """Build AudioRAGService using factories.

    Args:
        settings: Application settings
        explicit_store_path: Optional explicit path for JSONL store
        triton_url: Triton server URL for TritonBGEEmbedder

    Returns:
        AudioRAGService instance with configured components
    """
    store = create_store(settings, explicit_store_path)
    embedder = create_embedder(settings, triton_url=triton_url)
    reranker = create_reranker(settings, triton_url=triton_url)
    return AudioRAGService(
        store=store,
        embedder=embedder,
        reranker=reranker,
        settings=settings,
    )


def _print_answer(answer) -> None:
    print(answer.answer)
    if not answer.citations:
        return
    print("\nCitations:")
    for citation in answer.citations:
        audio_suffix = f" | audio={citation.audio_path}" if citation.audio_path else ""
        print(
            f"- {citation.source_id} [{citation.start_offset}:{citation.end_offset}] "
            f"score={citation.score:.3f}: {citation.snippet}{audio_suffix}"
        )


def _print_triton_result(result: Dict[str, Any]) -> None:
    resolved_query_text = result.get("resolved_query_text", "")
    if resolved_query_text:
        print(f"Resolved question transcript: {resolved_query_text}")
    print(result.get("answer", ""))
    citations = result.get("citations", [])
    if not citations:
        return
    print("\nCitations:")
    for citation in citations:
        audio_suffix = f" | audio={citation.get('audio_path', '')}" if citation.get("audio_path") else ""
        print(
            f"- {citation['source_id']} [{citation['start_offset']}:{citation['end_offset']}] "
            f"score={citation['score']:.3f}: {citation['snippet']}{audio_suffix}"
        )


def _parse_metadata(raw_metadata: str) -> Dict[str, Any]:
    payload = json.loads(raw_metadata)
    if not isinstance(payload, dict):
        raise ValueError("metadata-json must decode to an object")
    return payload
