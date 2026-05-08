import json
import os
import time
from typing import Optional

import numpy as np
import torch
import triton_python_backend_utils as pb_utils
from transformers import AutoModelForCausalLM, AutoTokenizer


class TritonPythonModel:
    """Triton Python backend for Qwen2.5 LLM (default: 1.5B-Instruct)."""

    def initialize(self, args):
        """Initialize the LLM model and tokenizer."""
        del args

        # Load settings to get cache_dir
        from audio_rag.config import load_settings
        from audio_rag.utils.logging import get_logger, log_model_loading, log_model_loaded
        settings = load_settings()

        # Setup logger
        self._logger = get_logger(__name__)

        # Model configuration
        model_name = os.environ.get(
            "AUDIO_RAG_LLM_MODEL",
            "Qwen/Qwen2.5-3B-Instruct"
        )
        device = os.environ.get("AUDIO_RAG_LLM_DEVICE", "cpu")
        max_new_tokens = int(os.environ.get("AUDIO_RAG_LLM_MAX_TOKENS", "512"))

        self._device = device
        self._max_new_tokens = max_new_tokens
        self._encoding = "utf-8"

        # Log model loading
        log_model_loading(self._logger, model_name, None)
        start_time = time.time()

        # Load tokenizer
        self._logger.info(f"Loading tokenizer for '{model_name}'...")
        self._tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True
        )

        # Load model
        self._logger.info(f"Loading model '{model_name}' on device '{device}'...")
        self._model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float32,
            device_map=device,
            trust_remote_code=True
        )
        self._model.eval()

        # Log successful loading
        load_time = time.time() - start_time
        log_model_loaded(self._logger, model_name, load_time)

        # System prompt for RAG
        self._system_prompt = (
            "Ты — полезный ассистент, который отвечает на вопросы на основе "
            "предоставленного контекста. Отвечай кратко и по существу. "
            "Если информация отсутствует в контексте, так и скажи."
        )

    def execute(self, requests):
        """Execute LLM inference on batch of requests."""
        responses = []

        for idx, request in enumerate(requests):
            request_id = f"llm-{idx}-{time.time()}"

            try:
                # Decode inputs
                query = self._decode_input(request, "INPUT_QUERY")
                context = self._decode_input(request, "INPUT_CONTEXT")
                system_prompt = self._decode_input_optional(request, "INPUT_SYSTEM_PROMPT")
                max_tokens = self._decode_int_optional(request, "INPUT_MAX_TOKENS")

                # Log incoming request
                self._logger.info(f"[{request_id}] LLM request - Query: {query[:100]}...")

                # Use provided system prompt or default
                sys_prompt = system_prompt if system_prompt else self._system_prompt
                max_new = max_tokens if max_tokens and max_tokens > 0 else self._max_new_tokens

                # Build prompt and generate answer
                prompt = self._build_prompt(query, context, sys_prompt)
                start_time = time.time()
                answer = self._generate(prompt, max_new)
                gen_time = time.time() - start_time

                # Log response
                self._logger.info(f"[{request_id}] LLM response in {gen_time:.2f}s - Answer: {answer[:100]}...")

                # Build response
                output_tensor = pb_utils.Tensor(
                    "OUTPUT_ANSWER",
                    np.array([answer.encode(self._encoding)], dtype=object)
                )
                responses.append(pb_utils.InferenceResponse(output_tensors=[output_tensor]))
            except Exception as e:
                # Log error and create error response
                self._logger.error(f"[{request_id}] Error during LLM execution: {e}", exc_info=True)
                # Still need to return a response
                error_msg = f"Error: {str(e)}"
                output_tensor = pb_utils.Tensor(
                    "OUTPUT_ANSWER",
                    np.array([error_msg.encode(self._encoding)], dtype=object)
                )
                responses.append(pb_utils.InferenceResponse(output_tensors=[output_tensor]))

        return responses

    def _build_prompt(self, query: str, context: str, system_prompt: str) -> str:
        """Build the prompt for the LLM."""
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Контекст:\n{context}\n\nВопрос: {query}"}
        ]

        # Apply chat template
        text = self._tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        return text

    def _generate(self, prompt: str, max_new_tokens: int) -> str:
        """Generate answer from the LLM."""
        # Tokenize
        inputs = self._tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=4096
        )
        inputs = {k: v.to(self._device) for k, v in inputs.items()}

        # Generate
        with torch.no_grad():
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=self._tokenizer.pad_token_id,
                eos_token_id=self._tokenizer.eos_token_id
            )

        # Decode only the new tokens (exclude the prompt)
        generated_ids = outputs[0][inputs["input_ids"].shape[1]:]
        answer = self._tokenizer.decode(generated_ids, skip_special_tokens=True)

        return answer.strip()

    def _decode_input(self, request, tensor_name: str) -> str:
        """Decode a required string input tensor."""
        tensor = pb_utils.get_input_tensor_by_name(request, tensor_name)
        return tensor.as_numpy()[0].decode(self._encoding)

    def _decode_input_optional(self, request, tensor_name: str) -> Optional[str]:
        """Decode an optional string input tensor."""
        try:
            tensor = pb_utils.get_input_tensor_by_name(request, tensor_name)
            value = tensor.as_numpy()[0]
            if value is not None and len(value) > 0:
                return value.decode(self._encoding)
        except Exception:
            pass
        return None

    def _decode_int_optional(self, request, tensor_name: str) -> Optional[int]:
        """Decode an optional integer input tensor."""
        try:
            tensor = pb_utils.get_input_tensor_by_name(request, tensor_name)
            return int(tensor.as_numpy()[0])
        except Exception:
            pass
        return None

    def finalize(self):
        """Clean up resources."""
        if hasattr(self, "_model"):
            del self._model
        if hasattr(self, "_tokenizer"):
            del self._tokenizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
