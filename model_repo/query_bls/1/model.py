import json
import os
from pathlib import Path

import numpy as np
import triton_python_backend_utils as pb_utils

from audio_rag.config import load_settings
from audio_rag.embeddings import HashingTextEmbedder
from audio_rag.service import AudioRAGService
from audio_rag.store import JsonlChunkStore


class TritonPythonModel:
    def initialize(self, args):
        del args
        self._settings = load_settings()
        store_path = Path(os.environ.get("AUDIO_RAG_STORE_PATH", self._settings.triton_server.store_path))
        self._service = AudioRAGService(
            store=JsonlChunkStore(store_path),
            embedder=HashingTextEmbedder(self._settings.embedding),
            settings=self._settings,
        )
        self._encoding = self._settings.transcript.encoding
        self._asr_model_name = self._settings.triton_http.model_asr_name
        self._llm_model_name = self._settings.triton_http.model_llm_name
        self._use_llm = os.environ.get("AUDIO_RAG_USE_LLM", "true").lower() == "true"

    def execute(self, requests):
        responses = []
        for inference_request in requests:
            query_text = self._decode_string(inference_request, "INPUT_QUERY_TEXT")
            question_audio_path = self._decode_string(inference_request, "INPUT_QUESTION_AUDIO_PATH")
            question_transcript_path = self._decode_string(inference_request, "INPUT_QUESTION_TRANSCRIPT_PATH")
            top_k = self._decode_top_k(inference_request)

            # Resolve query text from audio if provided
            if question_audio_path:
                if question_transcript_path:
                    resolved_query_text = Path(question_transcript_path).expanduser().resolve().read_text(
                        encoding=self._encoding
                    ).strip()
                else:
                    resolved_query_text = self._transcribe_audio(Path(question_audio_path).expanduser().resolve())
            else:
                resolved_query_text = query_text

            # Get answer from service (includes retrieval and citations)
            answer = self._service.ask(resolved_query_text, top_k=top_k)
            answer.resolved_query_text = resolved_query_text

            # If LLM is enabled and we have citations, generate answer with LLM
            if self._use_llm and answer.citations:
                context = self._format_context(answer.citations)
                llm_answer = self._call_llm(resolved_query_text, context)
                if llm_answer:
                    answer.answer = llm_answer

            payload = {
                "answer": answer.answer,
                "resolved_query_text": answer.resolved_query_text,
                "citations": [citation.__dict__ for citation in answer.citations],
            }
            responses.append(self._build_response(payload))
        return responses

    def _format_context(self, citations) -> str:
        """Format citations as context string for the LLM."""
        context_parts = []
        for i, citation in enumerate(citations, start=1):
            context_parts.append(f"[{i}] {citation.snippet}")
        return "\n".join(context_parts)

    def _call_llm(self, query: str, context: str) -> str:
        """Call the LLM model to generate an answer."""
        try:
            infer_input_query = pb_utils.Tensor(
                "INPUT_QUERY",
                np.array([query.encode(self._encoding)], dtype=object),
            )
            infer_input_context = pb_utils.Tensor(
                "INPUT_CONTEXT",
                np.array([context.encode(self._encoding)], dtype=object),
            )
            infer_request = pb_utils.InferenceRequest(
                model_name=self._llm_model_name,
                requested_output_names=["OUTPUT_ANSWER"],
                inputs=[infer_input_query, infer_input_context],
            )
            infer_response = infer_request.exec()
            if infer_response.has_error():
                # If LLM fails, return None to use the template answer
                return None
            answer_tensor = pb_utils.get_output_tensor_by_name(infer_response, "OUTPUT_ANSWER")
            return answer_tensor.as_numpy()[0].decode(self._encoding)
        except Exception:
            # If LLM fails, return None to use the template answer
            return None

    def _transcribe_audio(self, audio_path: Path) -> str:
        infer_input = pb_utils.Tensor(
            "INPUT_AUDIO_PATH",
            np.array([str(audio_path).encode(self._encoding)], dtype=object),
        )
        infer_request = pb_utils.InferenceRequest(
            model_name=self._asr_model_name,
            requested_output_names=["OUTPUT_TRANSCRIPT"],
            inputs=[infer_input],
        )
        infer_response = infer_request.exec()
        if infer_response.has_error():
            raise RuntimeError(infer_response.error().message())
        transcript_tensor = pb_utils.get_output_tensor_by_name(infer_response, "OUTPUT_TRANSCRIPT")
        return transcript_tensor.as_numpy()[0].decode(self._encoding)

    def _build_response(self, payload):
        output_tensor = pb_utils.Tensor(
            "OUTPUT_ANSWER",
            np.array([json.dumps(payload, ensure_ascii=False).encode(self._encoding)], dtype=object),
        )
        return pb_utils.InferenceResponse(output_tensors=[output_tensor])

    def _decode_string(self, inference_request, tensor_name):
        tensor = pb_utils.get_input_tensor_by_name(inference_request, tensor_name)
        return tensor.as_numpy()[0].decode(self._encoding)

    def _decode_top_k(self, inference_request):
        tensor = pb_utils.get_input_tensor_by_name(inference_request, "INPUT_TOP_K")
        value = int(tensor.as_numpy()[0])
        return value if value > 0 else self._settings.retrieval.default_top_k
