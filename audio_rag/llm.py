"""Local LLM for answer generation using Qwen2.5-0.5B-Instruct."""

import time
from typing import List, Optional

from transformers import AutoModelForCausalLM, AutoTokenizer

from .utils.logging import get_logger, log_model_loading, log_model_loaded


class LocalLLM:
    """Local LLM for generating natural language answers.

    Uses Qwen2.5-0.5B-Instruct model for efficient CPU inference.
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-0.5B-Instruct",
        device: str = "cpu",
        max_new_tokens: int = 512,
    ) -> None:
        """Initialize the LLM model.

        Args:
            model_name: Hugging Face model name
            device: Device to run on ("cpu" or "cuda")
            max_new_tokens: Maximum number of tokens to generate
        """
        self._model_name = model_name
        self._device = device
        self._max_new_tokens = max_new_tokens
        self._logger = get_logger(__name__)

        # Load model and tokenizer
        log_model_loading(self._logger, model_name, None)
        start_time = time.time()

        self._tokenizer = AutoTokenizer.from_pretrained(model_name)
        self._model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype="auto",
            device_map=device,
        )

        load_time = time.time() - start_time
        log_model_loaded(self._logger, model_name, load_time)

        # System prompt for RAG
        self._system_prompt = """Ты - полезный AI ассистент, который отвечает на вопросы на основе предоставленного контекста.

Инструкции:
- Отвечай на вопросы используя ТОЛЬКО информацию из предоставленного контекста
- Если информации недостаточно, так и скажи
- Отвечай кратко и по существу
- Используй цитаты из контекста для подтверждения ответа"""

    def generate_answer(
        self,
        query: str,
        context_chunks: List[str],
        max_new_tokens: Optional[int] = None,
    ) -> str:
        """Generate answer based on query and context chunks.

        Args:
            query: User's question
            context_chunks: List of context chunks from retrieval
            max_new_tokens: Override max_new_tokens if provided

        Returns:
            Generated answer string
        """
        if not context_chunks:
            return "Не найдено релевантной информации для ответа на вопрос."

        # Build context from chunks
        context = "\n\n".join(
            f"[{i+1}] {chunk}"
            for i, chunk in enumerate(context_chunks)
        )

        # Build prompt
        messages = [
            {"role": "system", "content": self._system_prompt},
            {"role": "user", "content": f"Контекст:\n{context}\n\nВопрос: {query}"},
        ]

        # Apply chat template
        text = self._tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        # Tokenize
        inputs = self._tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=4096,
        )
        inputs = {k: v.to(self._device) for k, v in inputs.items()}

        # Generate
        self._logger.debug(f"Generating answer for query: {query[:50]}...")
        start_time = time.time()

        max_tokens = max_new_tokens if max_new_tokens else self._max_new_tokens
        outputs = self._model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=False,
            temperature=1.0,
            top_p=1.0,
            pad_token_id=self._tokenizer.eos_token_id,
        )

        # Decode
        generated_ids = outputs[0][inputs["input_ids"].shape[1]:]
        answer = self._tokenizer.decode(generated_ids, skip_special_tokens=True)

        gen_time = time.time() - start_time
        self._logger.debug(f"Generated answer in {gen_time:.2f}s: {answer[:100]}...")

        return answer.strip()
