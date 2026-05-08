"""Embedder as Triton Python backend model."""

import json
import os
import time
import triton_python_backend_utils as pb_utils
import numpy as np

from audio_rag.config import load_settings
from audio_rag.factories import create_embedder
from audio_rag.utils.logging import get_logger


class TritonPythonModel:
    """Triton Python backend model for text embeddings.

    Supports multiple embedder types through factories:
    - BGE-M3 embeddings (semantic understanding)
    - Hashing embeddings (deterministic, for testing)
    """

    def initialize(self, args):
        """Initialize the embedder model.

        Args:
            args: Dictionary with model configuration
        """
        self._settings = load_settings()
        self._embedder = create_embedder(self._settings, model_name="bge_embedder")
        self._encoding = self._settings.transcript.encoding

        # Setup logger
        self._logger = get_logger(__name__)

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

        for idx, request in enumerate(requests):
            request_id = f"embed-{idx}-{time.time()}"

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

            # Log incoming request
            self._logger.info(f"[{request_id}] Embedding request - {len(decoded_texts)} texts")

            # Generate embeddings with timing
            start_time = time.time()
            embeddings = self._embedder.encode_batch(decoded_texts)
            encode_time = time.time() - start_time

            # Log response
            self._logger.info(f"[{request_id}] Embedding response in {encode_time:.3f}s - Shape: {len(embeddings)}x{len(embeddings[0]) if embeddings else 0}")

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
