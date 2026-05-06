import json
import os
from pathlib import Path

import numpy as np
import triton_python_backend_utils as pb_utils

from audio_rag.config import load_settings
from audio_rag.factories import create_embedder, create_reranker, create_store
from audio_rag.service import AudioRAGService


class TritonPythonModel:
    def initialize(self, args):
        del args
        self._settings = load_settings()

        # Use factories to create components based on configuration
        store = create_store(self._settings)
        embedder = create_embedder(self._settings, triton_url="localhost:8000")
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
        for inference_request in requests:
            source_id = self._decode_input(inference_request, "INPUT_SOURCE_ID")
            audio_path = Path(self._decode_input(inference_request, "INPUT_AUDIO_PATH")).expanduser().resolve()
            transcript_path_raw = self._decode_input(inference_request, "INPUT_TRANSCRIPT_PATH")
            metadata_json = self._decode_input(inference_request, "INPUT_METADATA_JSON") or "{}"
            metadata = json.loads(metadata_json)

            if transcript_path_raw:
                transcript = Path(transcript_path_raw).expanduser().resolve().read_text(encoding=self._encoding)
                transcript_origin = transcript_path_raw
            else:
                transcript = self._transcribe_audio(audio_path)
                transcript_origin = self._asr_model_name

            enriched_metadata = dict(metadata)
            enriched_metadata[self._settings.metadata.audio_path_key] = str(audio_path)
            enriched_metadata[self._settings.metadata.transcript_origin_key] = transcript_origin
            enriched_metadata[self._settings.metadata.ingest_mode_key] = self._settings.metadata.ingest_mode_triton

            chunks = self._service.ingest_transcript(
                source_id=source_id,
                transcript=transcript,
                metadata=enriched_metadata,
            )
            payload = {
                "source_id": source_id,
                "indexed_chunks": len(chunks),
                "audio_path": str(audio_path),
                "transcript_origin": transcript_origin,
            }
            responses.append(self._build_response(payload))
        return responses

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
            "OUTPUT_RESULT",
            np.array([json.dumps(payload, ensure_ascii=False).encode(self._encoding)], dtype=object),
        )
        return pb_utils.InferenceResponse(output_tensors=[output_tensor])

    def _decode_input(self, inference_request, tensor_name):
        tensor = pb_utils.get_input_tensor_by_name(inference_request, tensor_name)
        return tensor.as_numpy()[0].decode(self._encoding)
