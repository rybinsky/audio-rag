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

        # Use factories - they will automatically choose the right implementation
        # based on TRITON_SERVER environment variable
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
            # Get input audio path
            audio_path_tensor = pb_utils.get_input_tensor_by_name(request, "INPUT_AUDIO_PATH")
            audio_path = audio_path_tensor.as_numpy()[0]
            if isinstance(audio_path, bytes):
                audio_path = audio_path.decode(self._encoding)

            # Get input source ID
            source_id_tensor = pb_utils.get_input_tensor_by_name(request, "INPUT_SOURCE_ID")
            source_id = source_id_tensor.as_numpy()[0]
            if isinstance(source_id, bytes):
                source_id = source_id.decode(self._encoding)

            # Get optional transcript path
            transcript_path = None
            try:
                transcript_path_tensor = pb_utils.get_input_tensor_by_name(request, "INPUT_TRANSCRIPT_PATH")
                if transcript_path_tensor is not None:
                    transcript_path_bytes = transcript_path_tensor.as_numpy()[0]
                    if isinstance(transcript_path_bytes, bytes) and transcript_path_bytes:
                        transcript_path = transcript_path_bytes.decode(self._encoding)
            except Exception:
                pass

            # Get optional metadata
            metadata = {}
            try:
                metadata_tensor = pb_utils.get_input_tensor_by_name(request, "INPUT_METADATA_JSON")
                if metadata_tensor is not None:
                    metadata_json = metadata_tensor.as_numpy()[0]
                    if isinstance(metadata_json, bytes):
                        metadata_json = metadata_json.decode(self._encoding)
                    if metadata_json:
                        metadata = json.loads(metadata_json)
            except Exception:
                pass

            # Perform ASR using Triton
            asr_request = pb_utils.InferenceRequest(
                model_name=self._asr_model_name,
                requested_output_names=["OUTPUT_TRANSCRIPT"],
                inputs=[
                    pb_utils.Tensor("INPUT_AUDIO_PATH", np.array([audio_path.encode(self._encoding)], dtype=object))
                ]
            )
            asr_response = asr_request.exec()

            # Get transcript from ASR response
            transcript = pb_utils.get_output_tensor_by_name(asr_response, "OUTPUT_TRANSCRIPT").as_numpy()[0]
            if isinstance(transcript, bytes):
                transcript = transcript.decode(self._encoding)

            # Ingest the transcript
            chunks = self._service.ingest_transcript(
                source_id=source_id,
                transcript=transcript,
                metadata=metadata,
            )

            # Create output
            result = {
                "status": "success",
                "chunks_count": len(chunks),
                "source_id": source_id,
            }

            output_tensor = pb_utils.Tensor("OUTPUT_RESULT", np.array([json.dumps(result, ensure_ascii=False).encode(self._encoding)], dtype=object))
            response = pb_utils.InferenceResponse(output_tensors=[output_tensor])
            responses.append(response)

        return responses

    def finalize(self):
        pass
