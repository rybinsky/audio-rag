"""BGE-M3 embedder as Triton Python backend model."""

import json
import os
import triton_python_backend_utils as pb_utils
import numpy as np

from audio_rag.config import load_settings
from audio_rag.embedders import BGEEmbedder


class TritonPythonModel:
    """Triton Python backend model for BGE-M3 text embeddings."""

    def initialize(self, args):
        """Initialize the BGE-M3 model.

        Args:
            args: Dictionary with model configuration
        """
        self._settings = load_settings()
        self._embedder = BGEEmbedder(self._settings.bge)
        self._encoding = self._settings.transcript.encoding

        # Get model configuration
        self._model_config = json.loads(args["model_config"])

        # Get input/output configuration
        input_config = pb_utils.get_input_config_by_name(
            self._model_config, "INPUT_TEXT"
        )
        output_config = pb_utils.get_output_config_by_name(
            self._model_config, "OUTPUT_EMBEDDING"
        )

        # Convert Triton types to numpy types
        self._input_dtype = pb_utils.triton_string_to_numpy(
            input_config["data_type"]
        )
        self._output_dtype = pb_utils.triton_string_to_numpy(
            output_config["data_type"]
        )

    def execute(self, requests):
        """Process inference requests for text embeddings.

        Args:
            requests: List of InferenceRequest objects

        Returns:
            List of InferenceResponse objects with embeddings
        """
        responses = []

        for request in requests:
            # Get input text(s)
            input_tensor = pb_utils.get_input_tensor_by_name(request, "INPUT_TEXT")
            texts = input_tensor.as_numpy()

            # Decode texts from bytes to strings
            decoded_texts = []
            for text_bytes in texts:
                if isinstance(text_bytes, bytes):
                    decoded_texts.append(text_bytes.decode(self._encoding))
                else:
                    decoded_texts.append(str(text_bytes))

            # Generate embeddings
            embeddings = self._embedder.encode_batch(decoded_texts)

            # Convert to numpy array
            embeddings_array = np.array(embeddings, dtype=np.float32)

            # Create output tensor
            output_tensor = pb_utils.Tensor(
                "OUTPUT_EMBEDDING",
                embeddings_array
            )

            # Create response
            response = pb_utils.InferenceResponse(output_tensors=[output_tensor])
            responses.append(response)

        return responses

    def finalize(self):
        """Clean up resources."""
        pass
