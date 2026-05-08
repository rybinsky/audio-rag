import os
import time
from pathlib import Path

import numpy as np
import triton_python_backend_utils as pb_utils
from faster_whisper import WhisperModel


class TritonPythonModel:
    def initialize(self, args):
        del args

        # Load settings to get cache_dir
        from audio_rag.config import load_settings
        from audio_rag.utils.logging import get_logger, log_model_loading, log_model_loaded
        settings = load_settings()

        # Setup logger
        self._logger = get_logger(__name__)



        model_size = os.environ.get("AUDIO_RAG_ASR_MODEL_SIZE", "tiny")
        compute_type = os.environ.get("AUDIO_RAG_ASR_COMPUTE_TYPE", "int8")
        device = os.environ.get("AUDIO_RAG_ASR_DEVICE", "cpu")
        vad_filter = os.environ.get("AUDIO_RAG_ASR_VAD_FILTER", "true").lower() == "true"
        self._encoding = "utf-8"
        self._input_audio_path = "INPUT_AUDIO_PATH"
        self._output_transcript = "OUTPUT_TRANSCRIPT"
        self._vad_filter = vad_filter

        # Log model loading
        log_model_loading(self._logger, f"whisper-{model_size}", None)
        start_time = time.time()

        self._model = WhisperModel(model_size, device=device, compute_type=compute_type)

        # Log successful loading
        load_time = time.time() - start_time
        log_model_loaded(self._logger, f"whisper-{model_size}", load_time)

    def execute(self, requests):
        responses = []
        for idx, inference_request in enumerate(requests):
            request_id = f"asr-{idx}-{time.time()}"

            audio_path = Path(self._decode_input(inference_request, self._input_audio_path)).expanduser().resolve()
            if not audio_path.exists():
                self._logger.error(f"[{request_id}] Audio file not found: {audio_path}")
                raise FileNotFoundError(f"audio file not found: {audio_path}")

            # Log incoming request
            self._logger.info(f"[{request_id}] ASR request - Audio: {audio_path.name}")

            start_time = time.time()
            segments, _ = self._model.transcribe(str(audio_path), vad_filter=self._vad_filter)
            transcript = " ".join(segment.text.strip() for segment in segments if segment.text.strip()).strip()

            # Log response
            transcribe_time = time.time() - start_time
            self._logger.info(f"[{request_id}] ASR response in {transcribe_time:.2f}s - Transcript: {transcript[:100]}...")

            output_tensor = pb_utils.Tensor(
                self._output_transcript,
                np.array([transcript.encode(self._encoding)], dtype=object),
            )
            responses.append(pb_utils.InferenceResponse(output_tensors=[output_tensor]))
        return responses

    def _decode_input(self, inference_request, tensor_name):
        tensor = pb_utils.get_input_tensor_by_name(inference_request, tensor_name)
        return tensor.as_numpy()[0].decode(self._encoding)
