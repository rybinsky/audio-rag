"""Reranker as Triton Python backend model."""

import json
import triton_python_backend_utils as pb_utils
import numpy as np

from audio_rag.config import load_settings
from audio_rag.reranker import SearchReranker


class TritonPythonModel:
    """Triton Python backend model for search result reranking."""

    def initialize(self, args):
        """Initialize the reranker model.

        Args:
            args: Dictionary with model configuration
        """
        self._settings = load_settings()
        self._reranker = SearchReranker(self._settings.reranker)
        self._encoding = self._settings.transcript.encoding

        # Get model configuration
        self._model_config = json.loads(args["model_config"])

        # Get input/output configuration
        query_config = pb_utils.get_input_config_by_name(
            self._model_config, "INPUT_QUERY"
        )
        texts_config = pb_utils.get_input_config_by_name(
            self._model_config, "INPUT_TEXTS"
        )
        top_k_config = pb_utils.get_input_config_by_name(
            self._model_config, "INPUT_TOP_K"
        )

        indices_config = pb_utils.get_output_config_by_name(
            self._model_config, "OUTPUT_INDICES"
        )
        scores_config = pb_utils.get_output_config_by_name(
            self._model_config, "OUTPUT_SCORES"
        )

        # Convert Triton types to numpy types
        self._query_dtype = pb_utils.triton_string_to_numpy(
            query_config["data_type"]
        )
        self._texts_dtype = pb_utils.triton_string_to_numpy(
            texts_config["data_type"]
        )
        self._top_k_dtype = pb_utils.triton_string_to_numpy(
            top_k_config["data_type"]
        )
        self._indices_dtype = pb_utils.triton_string_to_numpy(
            indices_config["data_type"]
        )
        self._scores_dtype = pb_utils.triton_string_to_numpy(
            scores_config["data_type"]
        )

    def execute(self, requests):
        """Process inference requests for reranking.

        Args:
            requests: List of InferenceRequest objects

        Returns:
            List of InferenceResponse objects with reranked indices and scores
        """
        responses = []

        for request in requests:
            # Get input query
            query_tensor = pb_utils.get_input_tensor_by_name(request, "INPUT_QUERY")
            query = query_tensor.as_numpy()[0]
            if isinstance(query, bytes):
                query = query.decode(self._encoding)

            # Get input texts
            texts_tensor = pb_utils.get_input_tensor_by_name(request, "INPUT_TEXTS")
            texts = texts_tensor.as_numpy()

            # Decode texts from bytes to strings
            decoded_texts = []
            for text_bytes in texts:
                if isinstance(text_bytes, bytes):
                    decoded_texts.append(text_bytes.decode(self._encoding))
                else:
                    decoded_texts.append(str(text_bytes))

            # Get top_k (optional)
            top_k = None
            try:
                top_k_tensor = pb_utils.get_input_tensor_by_name(request, "INPUT_TOP_K")
                if top_k_tensor is not None:
                    top_k_value = top_k_tensor.as_numpy()[0]
                    top_k = int(top_k_value) if top_k_value > 0 else None
            except Exception:
                # If top_k is not provided, use default from settings
                pass

            # Perform reranking
            reranked_results = self._reranker.rerank_texts(
                query=query,
                texts=decoded_texts,
                top_k=top_k,
            )

            # Extract indices and scores
            indices = np.array([result[0] for result in reranked_results], dtype=np.int64)
            scores = np.array([result[1] for result in reranked_results], dtype=np.float32)

            # Create output tensors
            indices_tensor = pb_utils.Tensor("OUTPUT_INDICES", indices)
            scores_tensor = pb_utils.Tensor("OUTPUT_SCORES", scores)

            # Create response
            response = pb_utils.InferenceResponse(
                output_tensors=[indices_tensor, scores_tensor]
            )
            responses.append(response)

        return responses

    def finalize(self):
        """Clean up resources."""
        pass
