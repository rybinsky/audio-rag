import os
from pathlib import Path

import numpy as np
import triton_python_backend_utils as pb_utils
from faster_whisper import WhisperModel


class TritonPythonModel:
    def initialize(self, args):
        del args
        model_size = os.environ.get("AUDIO_RAG_ASR_MODEL_SIZE", "tiny")
        compute_type = os.environ.get("AUDIO_RAG_ASR_COMPUTE_TYPE", "int8")
        device = os.environ.get("AUDIO_RAG_ASR_DEVICE", "cpu")
        vad_filter = os.environ.get("AUDIO_RAG_ASR_VAD_FILTER", "true").lower() == "true"
        self._encoding = "utf-8"
        self._input_audio_path = "INPUT_AUDIO_PATH"
        self._output_transcript = "OUTPUT_TRANSCRIPT"
        self._vad_filter = vad_filter
        self._model = WhisperModel(model_size, device=device, compute_type=compute_type)

    def execute(self, requests):
        responses = []
        for inference_request in requests:
            audio_path = Path(self._decode_input(inference_request, self._input_audio_path)).expanduser().resolve()
            if not audio_path.exists():
                raise FileNotFoundError(f"audio file not found: {audio_path}")
            segments, _ = self._model.transcribe(str(audio_path), vad_filter=self._vad_filter)
            transcript = " ".join(segment.text.strip() for segment in segments if segment.text.strip()).strip()
            output_tensor = pb_utils.Tensor(
                self._output_transcript,
                np.array([transcript.encode(self._encoding)], dtype=object),
            )
            responses.append(pb_utils.InferenceResponse(output_tensors=[output_tensor]))
        return responses

    def _decode_input(self, inference_request, tensor_name):
        tensor = pb_utils.get_input_tensor_by_name(inference_request, tensor_name)
        return tensor.as_numpy()[0].decode(self._encoding)
