from __future__ import annotations

"""Model wrappers for dense embeddings and reranking."""

from typing import Sequence

import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer


class QwenDenseEmbedder:
    """Wrap the embedding model used for dense retrieval."""

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
        """Return embedding dimensionality for collection creation."""
        return int(self._model.get_sentence_embedding_dimension())

    def encode_documents(self, texts: Sequence[str]) -> list[list[float]]:
        """Encode document texts for indexing."""
        embeddings = self._model.encode(
            list(texts),
            batch_size=self._batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=self._show_progress,
        )
        return embeddings.tolist()

    def encode_query(self, query: str) -> list[float]:
        """Encode a query for dense retrieval."""
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
    """Wrap the reranker model used after hybrid candidate retrieval."""

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
            padding_side="left",
        )

        model_kwargs = {}
        if self._device.type == "cuda":
            model_kwargs["dtype"] = torch.float16

        self._model = AutoModelForCausalLM.from_pretrained(
            model_name,
            **model_kwargs,
        ).eval()
        self._model.to(self._device)

        # Reranking is modeled as a yes/no decision on the next token.
        self._token_false_id = self._tokenizer.convert_tokens_to_ids("no")
        self._token_true_id = self._tokenizer.convert_tokens_to_ids("yes")

        self._prefix = (
            '<|im_start|>system\nJudge whether the Document meets the requirements '
            'based on the Query and the Instruct provided. Note that the answer can '
            'only be "yes" or "no".<|im_end|>\n<|im_start|>user\n'
        )
        self._suffix = '<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n'

        self._prefix_tokens = self._tokenizer.encode(
            self._prefix,
            add_special_tokens=False,
        )
        self._suffix_tokens = self._tokenizer.encode(
            self._suffix,
            add_special_tokens=False,
        )

    def score(
        self,
        query: str,
        documents: Sequence[str],
        instruction: str,
    ) -> list[float]:
        """Score candidate documents against the user query."""
        pairs = [
            self._format_pair(instruction=instruction, query=query, document=text)
            for text in documents
        ]
        scores: list[float] = []

        for start in range(0, len(pairs), self._batch_size):
            batch_pairs = pairs[start : start + self._batch_size]
            batch_inputs = self._prepare_inputs(batch_pairs)
            scores.extend(self._compute_scores(batch_inputs))

        return scores

    @staticmethod
    def _format_pair(instruction: str, query: str, document: str) -> str:
        """Format one query-document pair for the reranker prompt."""
        return (
            f"<Instruct>: {instruction}\n"
            f"<Query>: {query}\n"
            f"<Document>: {document}"
        )

    def _prepare_inputs(self, pairs: Sequence[str]) -> dict[str, torch.Tensor]:
        """Tokenize, frame, and pad reranker inputs."""
        max_body_length = (
            self._max_length - len(self._prefix_tokens) - len(self._suffix_tokens)
        )

        inputs = self._tokenizer(
            list(pairs),
            padding=False,
            truncation="longest_first",
            return_attention_mask=False,
            max_length=max_body_length,
        )

        for i, token_ids in enumerate(inputs["input_ids"]):
            inputs["input_ids"][i] = (
                self._prefix_tokens + token_ids + self._suffix_tokens
            )

        padded = self._tokenizer.pad(
            inputs,
            padding=True,
            return_tensors="pt",
        )
        return {key: value.to(self._device) for key, value in padded.items()}

    @torch.inference_mode()
    def _compute_scores(self, inputs: dict[str, torch.Tensor]) -> list[float]:
        """Return yes-probabilities for the prepared inputs."""
        logits = self._model(**inputs).logits[:, -1, :]

        true_logits = logits[:, self._token_true_id]
        false_logits = logits[:, self._token_false_id]

        yes_no_logits = torch.stack([false_logits, true_logits], dim=1)
        probabilities = torch.softmax(yes_no_logits, dim=1)[:, 1]

        return probabilities.float().cpu().tolist()
