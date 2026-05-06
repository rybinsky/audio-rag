import json
import os
from pathlib import Path
from dataclasses import asdict

import numpy as np
import triton_python_backend_utils as pb_utils

from audio_rag.config import load_settings
from audio_rag.factories import create_embedder, create_reranker, create_store
from audio_rag.service import AudioRAGService


class TritonPythonModel:
    def initialize(self, args):
        del args
        self._settings = load_settings()

        # Use factories to create components (will use direct implementations inside Triton server)
        store = create_store(self._settings)
        embedder = create_embedder(self._settings)
        reranker = create_reranker(self._settings)

        self._service = AudioRAGService(
            store=store,
            embedder=embedder,
            reranker=reranker,
            settings=self._settings,
        )
        self._encoding = self._settings.transcript.encoding
        self._asr_model_name = self._settings.triton_http.model_asr_name

    def execute(self, requests):
        responses = []

        for request in requests:
            # Get query text (may be empty if using audio)
            query_text = ""
            try:
                query_text_tensor = pb_utils.get_input_tensor_by_name(request, "INPUT_QUERY_TEXT")
                if query_text_tensor is not None:
                    query_text_bytes = query_text_tensor.as_numpy()[0]
                    if isinstance(query_text_bytes, bytes) and query_text_bytes:
                        query_text = query_text_bytes.decode(self._encoding)
            except Exception:
                pass

            # Get question audio path (may be empty)
            question_audio_path = ""
            try:
                audio_path_tensor = pb_utils.get_input_tensor_by_name(request, "INPUT_QUESTION_AUDIO_PATH")
                if audio_path_tensor is not None:
                    audio_path_bytes = audio_path_tensor.as_numpy()[0]
                    if isinstance(audio_path_bytes, bytes) and audio_path_bytes:
                        question_audio_path = audio_path_bytes.decode(self._encoding)
            except Exception:
                pass

            # Get question transcript path (may be empty)
            question_transcript_path = ""
            try:
                transcript_path_tensor = pb_utils.get_input_tensor_by_name(request, "INPUT_QUESTION_TRANSCRIPT_PATH")
                if transcript_path_tensor is not None:
                    transcript_path_bytes = transcript_path_tensor.as_numpy()[0]
                    if isinstance(transcript_path_bytes, bytes) and transcript_path_bytes:
                        question_transcript_path = transcript_path_bytes.decode(self._encoding)
            except Exception:
                pass

            # Get top_k
            top_k = self._settings.retrieval.default_top_k
            try:
                top_k_tensor = pb_utils.get_input_tensor_by_name(request, "INPUT_TOP_K")
                if top_k_tensor is not None:
                    top_k = int(top_k_tensor.as_numpy()[0])
            except Exception:
                pass

            # Determine the query text to use
            resolved_query_text = query_text

            # If no direct text, try transcript file
            if not resolved_query_text and question_transcript_path:
                transcript_file = Path(question_transcript_path)
                if transcript_file.exists():
                    resolved_query_text = transcript_file.read_text(encoding=self._encoding).strip()

            # If still no text, perform ASR on audio
            if not resolved_query_text and question_audio_path:
                asr_request = pb_utils.InferenceRequest(
                    model_name=self._asr_model_name,
                    requested_output_names=["OUTPUT_TRANSCRIPT"],
                    inputs=[
                        pb_utils.Tensor("INPUT_AUDIO_PATH", np.array([question_audio_path.encode(self._encoding)], dtype=object))
                    ]
                )
                asr_response = asr_request.exec()
                transcript = pb_utils.get_output_tensor_by_name(asr_response, "OUTPUT_TRANSCRIPT").as_numpy()[0]
                if isinstance(transcript, bytes):
                    transcript = transcript.decode(self._encoding)
                resolved_query_text = transcript.strip()

            # Perform the query
            answer = self._service.ask(resolved_query_text, top_k=top_k, use_reranker=True)

            # Convert to dict for JSON serialization
            result = {
                "answer": answer.answer,
                "resolved_query_text": answer.resolved_query_text,
                "citations": [asdict(citation) for citation in answer.citations],
            }

            output_tensor = pb_utils.Tensor(
                "OUTPUT_ANSWER",
                np.array([json.dumps(result, ensure_ascii=False).encode(self._encoding)], dtype=object)
            )
            response = pb_utils.InferenceResponse(output_tensors=[output_tensor])
            responses.append(response)

        return responses

    def finalize(self):
        pass
