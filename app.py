"""Gradio demo app for Hugging Face Spaces."""

import gradio as gr
from pathlib import Path
import tempfile
import json
import os
from typing import Optional

from audio_rag.config import load_settings
from audio_rag.factories import create_embedder, create_reranker, create_store
from audio_rag.service import AudioRAGService

# Global service instance
_service: Optional[AudioRAGService] = None


def get_service() -> AudioRAGService:
    """Get or create AudioRAGService instance."""
    global _service
    if _service is None:
        # Use config_hf for Hugging Face Spaces (can be overridden via env var)
        config_name = os.environ.get("AUDIO_RAG_CONFIG_NAME", "config_hf")
        settings = load_settings(config_name=config_name)
        # Use a temporary directory for the store in HF Spaces
        store_path = Path(os.environ.get("AUDIO_RAG_STORE_PATH", "/tmp/audio_rag_store.jsonl"))
        store = create_store(settings, store_path)
        embedder = create_embedder(settings)
        reranker = create_reranker(settings)
        _service = AudioRAGService(
            store=store,
            embedder=embedder,
            reranker=reranker,
            settings=settings,
        )
    return _service


def ingest_podcast(audio_file, transcript_file, source_id: str, metadata: str = "{}"):
    """Ingest a podcast audio file with transcript."""
    if not audio_file:
        return "❌ Please upload an audio file"

    if not source_id.strip():
        return "❌ Please provide a source ID"

    if not transcript_file:
        return "❌ Please upload a transcript file (.txt)"

    try:
        service = get_service()

        # Save uploaded files temporarily
        audio_path = Path(audio_file)
        transcript_path = Path(transcript_file)

        # Parse metadata
        try:
            meta_dict = json.loads(metadata) if metadata.strip() else {}
        except json.JSONDecodeError:
            return "❌ Invalid JSON in metadata field"

        # Ingest the podcast
        chunks = service.ingest_podcast(
            source_id=source_id.strip(),
            audio_path=audio_path,
            transcript_path=transcript_path,
            metadata=meta_dict,
        )

        return f"✅ Successfully ingested {len(chunks)} chunks from '{source_id}'\n\nStore location: {service.store.path}"

    except Exception as e:
        return f"❌ Error: {str(e)}"


def ask_question(query: str, top_k: int = 5):
    """Ask a question and get an answer with citations."""
    if not query.strip():
        return "❌ Please enter a question", ""

    try:
        service = get_service()
        answer = service.ask(query=query.strip(), top_k=top_k)

        # Format answer
        answer_text = f"**Answer:**\n\n{answer.answer}"

        # Format citations
        citations_text = ""
        if answer.citations:
            citations_text = "### 📚 Citations:\n\n"
            for i, citation in enumerate(answer.citations, 1):
                citations_text += f"**[{i}]** {citation.source_id} "
                citations_text += f"[{citation.start_offset}:{citation.end_offset}] "
                citations_text += f"(score: {citation.score:.3f})\n"
                citations_text += f"> {citation.snippet}\n\n"
        else:
            citations_text = "No citations found."

        return answer_text, citations_text

    except Exception as e:
        return f"❌ Error: {str(e)}", ""


def get_stats():
    """Get current index statistics."""
    try:
        service = get_service()
        sources = service.store.list_sources()
        return {
            "Total Chunks": service.store.count_chunks(),
            "Sources": ", ".join(sources) if sources else "None",
            "Store Path": str(service.store.path),
        }
    except Exception as e:
        return {"Error": str(e)}


def clear_index():
    """Clear the index."""
    try:
        service = get_service()
        service.store.clear()
        return "✅ Index cleared successfully"
    except Exception as e:
        return f"❌ Error: {str(e)}"


# Create Gradio interface
with gr.Blocks(title="🎙️ Audio RAG Demo", theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        """
        # 🎙️ Audio RAG Demo

        Upload podcasts with transcripts and ask questions about their content!

        **Note:** This demo requires:
        - Audio file (.mp3, .wav, .m4a, etc.)
        - Transcript file (.txt) with the same content as the audio
        """
    )

    with gr.Tabs():
        # Tab 1: Ingest Podcast
        with gr.Tab("📤 Upload Podcast"):
            with gr.Row():
                with gr.Column():
                    audio_input = gr.Audio(
                        label="Audio File",
                        type="filepath",
                        sources=["upload"],
                    )
                    transcript_input = gr.File(
                        label="Transcript File (.txt)",
                        file_types=[".txt"],
                        type="filepath",
                    )
                    source_id_input = gr.Textbox(
                        label="Source ID",
                        placeholder="my-podcast",
                        value="demo-podcast",
                    )
                    metadata_input = gr.Textbox(
                        label="Metadata (JSON, optional)",
                        placeholder='{"author": "John Doe", "date": "2024-01-01"}',
                        lines=2,
                    )
                    ingest_btn = gr.Button("📤 Ingest Podcast", variant="primary", size="lg")

                with gr.Column():
                    ingest_output = gr.Textbox(
                        label="Result",
                        lines=10,
                        show_copy_button=True,
                    )

            gr.Markdown(
                """
                ### 📝 Instructions:
                1. Upload an audio file (podcast, interview, etc.)
                2. Upload a transcript file (.txt) with the spoken content
                3. Provide a unique source ID (e.g., "my-podcast-episode-1")
                4. Click "Ingest Podcast" to process and index
                """
            )

            ingest_btn.click(
                fn=ingest_podcast,
                inputs=[audio_input, transcript_input, source_id_input, metadata_input],
                outputs=ingest_output,
            )

        # Tab 2: Ask Questions
        with gr.Tab("❓ Ask Questions"):
            with gr.Row():
                with gr.Column():
                    query_input = gr.Textbox(
                        label="Your Question",
                        placeholder="What was discussed about...?",
                        lines=2,
                    )
                    top_k_slider = gr.Slider(
                        minimum=1,
                        maximum=20,
                        value=5,
                        step=1,
                        label="Number of citations to retrieve",
                    )
                    ask_btn = gr.Button("🔍 Ask", variant="primary", size="lg")

                with gr.Column():
                    answer_output = gr.Textbox(
                        label="Answer",
                        lines=10,
                        show_copy_button=True,
                    )
                    citations_output = gr.Markdown(
                        label="Citations",
                    )

            gr.Examples(
                examples=[
                    ["What are the main topics discussed?"],
                    ["What was mentioned about technology?"],
                    ["Can you summarize the key points?"],
                ],
                inputs=query_input,
            )

            ask_btn.click(
                fn=ask_question,
                inputs=[query_input, top_k_slider],
                outputs=[answer_output, citations_output],
            )

        # Tab 3: Statistics
        with gr.Tab("📊 Statistics"):
            with gr.Row():
                stats_btn = gr.Button("🔄 Refresh Stats", variant="secondary")
                clear_btn = gr.Button("🗑️ Clear Index", variant="stop")

            stats_output = gr.JSON(label="Index Statistics")
            clear_output = gr.Textbox(label="Clear Result", visible=False)

            stats_btn.click(
                fn=get_stats,
                inputs=[],
                outputs=stats_output,
            )

            clear_btn.click(
                fn=clear_index,
                inputs=[],
                outputs=clear_output,
            ).then(
                fn=get_stats,
                inputs=[],
                outputs=stats_output,
            )

        # Tab 4: About
        with gr.Tab("ℹ️ About"):
            gr.Markdown(
                """
                ### 🎙️ Audio RAG - Retrieval-Augmented Generation for Audio Content

                This demo showcases an Audio RAG system that:
                - **Ingests** podcast audio with transcripts
                - **Indexes** content using BGE-M3 embeddings
                - **Retrieves** relevant segments using semantic search
                - **Reranks** results for better accuracy

                ### 🔧 Technical Stack:
                - **Embeddings**: BAAI/bge-m3 (running locally)
                - **Vector Store**: JSONL (lightweight, no database needed)
                - **Reranker**: BAAI/bge-reranker-v2-m3
                - **UI**: Gradio

                ### 📚 Use Cases:
                - Podcast search and Q&A
                - Interview analysis
                - Audio content exploration
                - Meeting transcription search

                ### 🔗 Links:
                - **Note**: This demo runs entirely on CPU with local models
                - [GitHub Repository](https://github.com/yourusername/audio-rag)
                - [Documentation](https://github.com/yourusername/audio-rag#readme)
                """
            )


if __name__ == "__main__":
    demo.launch()
