"""Gradio demo app for Hugging Face Spaces."""

import gradio as gr
from pathlib import Path
import tempfile
import json
import os
from typing import Optional

from faster_whisper import WhisperModel

from audio_rag.config import load_settings
from audio_rag.factories import create_embedder, create_reranker, create_store
from audio_rag.service import AudioRAGService
from audio_rag.llm import LocalLLM

# Global service instance
_service: Optional[AudioRAGService] = None

# Global ASR model instance
_asr_model: Optional[WhisperModel] = None

# Global LLM model instance
_llm_model: Optional[LocalLLM] = None


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

        # Initialize LLM if enabled
        llm = None
        if settings.llm.enabled:
            llm = get_llm()

        _service = AudioRAGService(
            store=store,
            embedder=embedder,
            reranker=reranker,
            settings=settings,
            llm=llm,
        )
    return _service


def get_asr_model() -> WhisperModel:
    """Get or create Whisper ASR model instance."""
    global _asr_model
    if _asr_model is None:
        # Use tiny model for CPU efficiency
        _asr_model = WhisperModel(
            "tiny",
            device="cpu",
            compute_type="int8",
        )
    return _asr_model


def get_llm() -> LocalLLM:
    """Get or create LocalLLM instance."""
    global _llm_model
    if _llm_model is None:
        # Get settings
        config_name = os.environ.get("AUDIO_RAG_CONFIG_NAME", "config_hf")
        settings = load_settings(config_name=config_name)

        # Initialize LLM
        _llm_model = LocalLLM(
            model_name=settings.llm.model_name,
            device=settings.llm.device,
            max_new_tokens=settings.llm.max_new_tokens,
        )
    return _llm_model


def transcribe_audio(audio_path: Path) -> str:
    """Transcribe audio using Whisper ASR.

    Args:
        audio_path: Path to audio file

    Returns:
        Transcribed text
    """
    model = get_asr_model()
    segments, _ = model.transcribe(
        str(audio_path),
        vad_filter=True,
    )
    transcript = " ".join(
        segment.text.strip()
        for segment in segments
        if segment.text.strip()
    ).strip()
    return transcript


def ingest_podcast(audio_file, transcript_file, source_id: str, metadata: str = "{}"):
    """Ingest a podcast audio file with optional transcript.

    If transcript is not provided, it will be automatically generated using Whisper ASR.
    """
    if not audio_file:
        return "❌ Please upload an audio file"

    if not source_id.strip():
        return "❌ Please provide a source ID"

    try:
        service = get_service()

        # Save uploaded audio file temporarily
        audio_path = Path(audio_file)

        # Handle transcript
        if transcript_file:
            # Use provided transcript
            transcript_path = Path(transcript_file)
            status = "Using provided transcript"
        else:
            # Transcribe audio using Whisper ASR
            status = "🎤 Transcribing audio with Whisper (this may take a moment)..."
            print(status)  # Log to console

            transcript_text = transcribe_audio(audio_path)

            if not transcript_text:
                return "❌ Failed to transcribe audio. Please try again or provide a transcript manually."

            # Save transcript to temporary file
            temp_transcript = tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False)
            temp_transcript.write(transcript_text)
            temp_transcript.close()
            transcript_path = Path(temp_transcript.name)

            status = f"✅ Transcribed successfully ({len(transcript_text)} characters)"
            print(status)  # Log to console

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

        return f"✅ Successfully ingested {len(chunks)} chunks from '{source_id}'\n\n{status}\n\nStore location: {service.store.path}"

    except Exception as e:
        return f"❌ Error: {str(e)}"


def ask_question(query: str, question_audio, top_k: int = 5):
    """Ask a question (text or audio) and get an answer with citations.

    Args:
        query: Text question (optional if question_audio provided)
        question_audio: Audio file with question (optional if query provided)
        top_k: Number of results to retrieve

    Returns:
        Tuple of (answer_text, citations_text)
    """
    # Determine the actual query text
    actual_query = None
    query_source = ""

    if question_audio:
        # Transcribe audio question
        audio_path = Path(question_audio)
        query_source = "🎤 Transcribing audio question..."
        print(query_source)  # Log to console

        actual_query = transcribe_audio(audio_path)

        if not actual_query:
            return "❌ Failed to transcribe audio question. Please try again or type your question.", ""

        query_source = f"🎤 Transcribed from audio: \"{actual_query[:50]}...\""
        print(query_source)  # Log to console

    elif query and query.strip():
        # Use text question
        actual_query = query.strip()
        query_source = "📝 Using text question"

    else:
        return "❌ Please provide either a text question or an audio file with your question.", ""

    try:
        service = get_service()
        answer = service.ask(query=actual_query, top_k=top_k)

        # Format answer
        answer_text = f"**Answer:**\n\n{answer.answer}\n\n{query_source}"

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
        stats = {
            "Total Chunks": service.store.count_chunks(),
            "Sources": ", ".join(sources) if sources else "None",
            "Store Path": str(service.store.path),
        }
        return json.dumps(stats, indent=2)
    except Exception as e:
        return json.dumps({"Error": str(e)}, indent=2)


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

        **Features:**
        - Upload audio files (MP3, WAV, M4A, etc.)
        - **Automatic transcription** with Whisper ASR (optional manual transcript)
        - Ask questions about your audio content
        - Get answers with precise citations
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
                2. **OPTIONAL**: Upload a transcript file (.txt) - if not provided, it will be automatically generated using Whisper ASR
                3. Provide a unique source ID (e.g., "my-podcast-episode-1")
                4. Click "Ingest Podcast" to process and index

                **Note**: Automatic transcription with Whisper may take 1-2 minutes for short audio files.
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
                        label="Your Question (text)",
                        placeholder="What was discussed about...?",
                        lines=2,
                    )
                    question_audio_input = gr.Audio(
                        label="Or upload audio question (OPTIONAL)",
                        type="filepath",
                        sources=["upload", "microphone"],
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
                inputs=[query_input, question_audio_input, top_k_slider],
                outputs=[answer_output, citations_output],
            )

        # Tab 3: Statistics
        with gr.Tab("📊 Statistics"):
            with gr.Row():
                stats_btn = gr.Button("🔄 Refresh Stats", variant="secondary")
                clear_btn = gr.Button("🗑️ Clear Index", variant="stop")

            stats_output = gr.Textbox(
                label="Index Statistics",
                lines=10,
                show_copy_button=True,
            )
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
