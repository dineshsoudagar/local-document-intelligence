from __future__ import annotations

from pathlib import Path

"""Local Qwen model wrappers for embedding, reranking, and grounded generation."""

import re
from typing import Any, Sequence

import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer


class QwenDenseEmbedder:
    """Sentence-transformers wrapper for dense embedding generation."""

    def __init__(
            self,
            model_name: str,
            batch_size: int = 8,
            show_progress: bool = True,
    ) -> None:
        self._batch_size = batch_size
        self._show_progress = show_progress
        self._device = "cuda" if torch.cuda.is_available() else "cpu"

        model_kwargs = None
        if self._device == "cuda":
            model_kwargs = {"dtype": torch.float16}

        self._model = SentenceTransformer(
            model_name,
            device=self._device,
            model_kwargs=model_kwargs,
            tokenizer_kwargs={"padding_side": "left"},
        )

    @property
    def dimension(self) -> int:
        """Return the embedding dimensionality."""
        return int(self._model.get_sentence_embedding_dimension())

    def encode_documents(self, texts: Sequence[str]) -> list[list[float]]:
        """Encode document texts into normalized dense embeddings."""
        embeddings = self._model.encode(
            list(texts),
            batch_size=self._batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=self._show_progress,
        )
        return embeddings.tolist()

    def encode_query(self, query: str) -> list[float]:
        """Encode a search query into a normalized dense embedding."""
        embedding = self._model.encode(
            [query],
            batch_size=1,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
            prompt_name="query",
        )
        return embedding[0].tolist()


class QwenReranker:
    """Causal-LM-based yes/no reranker for query-document relevance scoring."""

    def __init__(
            self,
            model_name: str,
            batch_size: int = 4,
            max_length: int = 4096,
    ) -> None:
        self._batch_size = batch_size
        self._max_length = max_length
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self._tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            local_files_only=True,
            padding_side="left",
            trust_remote_code=True,
        )
        self._pad_token_id = self._resolve_pad_token_id()

        model_kwargs: dict[str, Any] = {
            "local_files_only": True,
            "trust_remote_code": True,
        }
        if self._device.type == "cuda":
            model_kwargs["torch_dtype"] = torch.float16

        self._model = AutoModelForCausalLM.from_pretrained(
            model_name,
            **model_kwargs,
        ).eval()
        self._model.to(self._device)

        self._token_false_id = self._tokenizer.convert_tokens_to_ids("no")
        self._token_true_id = self._tokenizer.convert_tokens_to_ids("yes")

        self._prefix = (
            '<|im_start|>system\nJudge whether the Document meets the requirements '
            'based on the Query and the Instruct provided. Note that the answer can '
            'only be "yes" or "no".<|im_end|>\n<|im_start|>user\n'
        )
        self._suffix = '<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n'
        self._prefix_tokens = self._encode_text(self._prefix)
        self._suffix_tokens = self._encode_text(self._suffix)

    def score(
            self,
            query: str,
            documents: Sequence[str],
            instruction: str,
    ) -> list[float]:
        """Return yes-probability relevance scores for the supplied documents."""
        pairs = [
            self._format_pair(instruction=instruction, query=query, document=text)
            for text in documents
        ]
        scores: list[float] = []

        for start in range(0, len(pairs), self._batch_size):
            batch_pairs = pairs[start: start + self._batch_size]
            batch_inputs = self._prepare_inputs(batch_pairs)
            scores.extend(self._compute_scores(batch_inputs))

        return scores

    @staticmethod
    def _format_pair(instruction: str, query: str, document: str) -> str:
        """Format a reranking sample as instruction, query, and document text."""
        return (
            f"<Instruct>: {instruction}\n"
            f"<Query>: {query}\n"
            f"<Document>: {document}"
        )

    def _encode_text(self, text: str) -> list[int]:
        """Encode a prompt fragment without adding tokenizer-specific special tokens."""
        return self._tokenizer(text, add_special_tokens=False)["input_ids"]

    def _resolve_pad_token_id(self) -> int:
        """Return a valid pad token id for batch collation."""
        if self._tokenizer.pad_token_id is not None:
            return int(self._tokenizer.pad_token_id)
        if self._tokenizer.eos_token_id is not None:
            return int(self._tokenizer.eos_token_id)
        raise ValueError("Tokenizer must define pad_token_id or eos_token_id")

    def _prepare_inputs(self, pairs: Sequence[str]) -> dict[str, torch.Tensor]:
        """Tokenize, truncate body text, and batch inputs without tokenizer.pad."""
        max_body_length = (
                self._max_length - len(self._prefix_tokens) - len(self._suffix_tokens)
        )
        encoded_pairs = self._tokenizer(
            list(pairs),
            add_special_tokens=False,
            padding=False,
            truncation=True,
            max_length=max_body_length,
            return_attention_mask=False,
        )

        sequences: list[list[int]] = []
        for token_ids in encoded_pairs["input_ids"]:
            sequences.append(self._prefix_tokens + token_ids + self._suffix_tokens)

        max_seq_len = max(len(sequence) for sequence in sequences)
        input_ids = torch.full(
            (len(sequences), max_seq_len),
            fill_value=self._pad_token_id,
            dtype=torch.long,
        )
        attention_mask = torch.zeros(
            (len(sequences), max_seq_len),
            dtype=torch.long,
        )

        for row_index, sequence in enumerate(sequences):
            sequence_tensor = torch.tensor(sequence, dtype=torch.long)
            input_ids[row_index, -len(sequence):] = sequence_tensor
            attention_mask[row_index, -len(sequence):] = 1

        return {
            "input_ids": input_ids.to(self._device),
            "attention_mask": attention_mask.to(self._device),
        }

    @torch.inference_mode()
    def _compute_scores(self, inputs: dict[str, torch.Tensor]) -> list[float]:
        """Compute yes-probability scores from the final-token logits."""
        logits = self._model(**inputs).logits[:, -1, :]
        true_logits = logits[:, self._token_true_id]
        false_logits = logits[:, self._token_false_id]
        yes_no_logits = torch.stack([false_logits, true_logits], dim=1)
        probabilities = torch.softmax(yes_no_logits, dim=1)[:, 1]
        return probabilities.float().cpu().tolist()


class LocalQwenGenerator:
    """Local wrapper for grounded text generation with Qwen instruct models."""

    def __init__(self, model_name: str | Path) -> None:
        self._model_name = model_name
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self._tokenizer = AutoTokenizer.from_pretrained(
            self._model_name,
            local_files_only=True,
            padding_side="left",
            trust_remote_code=True,
        )
        if self._tokenizer.pad_token_id is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        model_kwargs: dict[str, Any] = {
            "local_files_only": True,
            "trust_remote_code": True,
        }
        if self._device.type == "cuda":
            model_kwargs["torch_dtype"] = torch.float16
            model_kwargs["device_map"] = "auto"
        else:
            model_kwargs["torch_dtype"] = torch.float32

        self._model = AutoModelForCausalLM.from_pretrained(
            self._model_name,
            **model_kwargs,
        ).eval()

        if self._device.type != "cuda":
            self._model.to(self._device)

    @property
    def device(self) -> torch.device:
        """Return the active model device."""
        return self._model.device

    def count_tokens(self, text: str) -> int:
        """Return the token count for a text fragment."""
        return len(self._tokenizer(text, add_special_tokens=False)["input_ids"])

    def truncate_text(self, text: str, max_tokens: int) -> str:
        """Truncate a text fragment to a maximum token budget."""
        if max_tokens <= 0:
            return ""
        token_ids = self._tokenizer(text, add_special_tokens=False)["input_ids"]
        if len(token_ids) <= max_tokens:
            return text
        return self._tokenizer.decode(token_ids[:max_tokens], skip_special_tokens=True).strip()

    def build_prompt(
            self,
            *,
            query: str,
            context: str,
            system_prompt: str,
            answer_instruction: str,
    ) -> str:
        """Build a chat prompt for grounded question answering."""
        messages = [
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": (
                    f"Question:\n{query}\n\n"
                    f"Retrieved context:\n{context}\n\n"
                    f"{answer_instruction}"
                ),
            },
        ]

        if hasattr(self._tokenizer, "apply_chat_template"):
            return self._tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )

        return (
            "System:\n"
            f"{system_prompt}\n\n"
            "User:\n"
            f"Question:\n{query}\n\n"
            f"Retrieved context:\n{context}\n\n"
            f"{answer_instruction}\n\n"
            "Assistant:\n"
        )

    def generate_from_prompt(
            self,
            prompt: str,
            *,
            max_new_tokens: int,
            temperature: float,
            top_p: float,
            repetition_penalty: float,
    ) -> str:
        """Generate a completion for a prepared prompt."""
        inputs = self._tokenizer(prompt, return_tensors="pt")
        inputs = {key: value.to(self._model.device) for key, value in inputs.items()}

        do_sample = temperature > 0
        generate_kwargs: dict[str, Any] = {
            "max_new_tokens": max_new_tokens,
            "do_sample": do_sample,
            "repetition_penalty": repetition_penalty,
            "pad_token_id": self._tokenizer.pad_token_id,
            "eos_token_id": self._tokenizer.eos_token_id,
        }
        if do_sample:
            generate_kwargs["temperature"] = temperature
            generate_kwargs["top_p"] = top_p

        with torch.inference_mode():
            output_ids = self._model.generate(**inputs, **generate_kwargs)

        prompt_length = inputs["input_ids"].shape[1]
        generated_ids = output_ids[0][prompt_length:]
        answer = self._tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
        return self._strip_reasoning(answer)

    def generate_grounded_answer(
            self,
            *,
            query: str,
            context: str,
            system_prompt: str,
            answer_instruction: str,
            max_new_tokens: int,
            temperature: float,
            top_p: float,
            repetition_penalty: float,
    ) -> str:
        """Generate a grounded answer from retrieved context."""
        prompt = self.build_prompt(
            query=query,
            context=context,
            system_prompt=system_prompt,
            answer_instruction=answer_instruction,
        )
        return self.generate_from_prompt(
            prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
        )

    @staticmethod
    def _strip_reasoning(text: str) -> str:
        """Remove visible reasoning tags if the model emits them."""
        cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
        return cleaned.strip()
