"""Local Qwen model wrappers for embedding, reranking, and grounded generation."""

from __future__ import annotations

from pathlib import Path

import re
from dataclasses import dataclass
from typing import Any, Iterator, Literal, Sequence
from src.config.retrieval_control_config import RewriteConfig
import torch
from threading import Thread
from sentence_transformers import SentenceTransformer
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TextIteratorStreamer,
)

from src.config.generator_config import GeneratorConfig

@dataclass(slots=True)
class GeneratedText:
    """Structured generation result."""

    answer: str
    thinking_content: str | None = None
    thinking_finished: bool = False


@dataclass(slots=True)
class StreamEvent:
    """Typed generation event for streaming."""

    kind: Literal["thinking_token", "thinking_done", "answer_token"]
    text: str = ""


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
        dim = self._model.get_sentence_embedding_dimension()
        assert dim is not None, "Embedding dimension should not be None"
        return int(dim)

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
            model_kwargs["dtype"] = torch.float16

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

def _resolve_torch_dtype(name: str) -> torch.dtype | None:
    mapping = {
        "auto": None,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    return mapping[name]

class LocalQwenGenerator:
    """Local wrapper for grounded text generation with Qwen instruct models."""

    @classmethod
    def from_config(cls, config: GeneratorConfig) -> "LocalQwenGenerator":
        return cls(config.generator_model_path, config=config)

    def __init__(
        self,
        model_name: str | Path,
        config: GeneratorConfig | None = None,
    ) -> None:
        self._model_name = model_name
        self._config = config
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self._tokenizer = AutoTokenizer.from_pretrained(
            self._model_name,
            local_files_only=True,
            padding_side="left",
            trust_remote_code=True,
        )
        if self._tokenizer.pad_token_id is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        self._model = self._load_model()

    def _load_model(self):
        if self._config is None:
            model_kwargs: dict[str, Any] = {
                "local_files_only": True,
                "trust_remote_code": True,
            }
            if self._device.type == "cuda":
                model_kwargs["torch_dtype"] = torch.float16
                model_kwargs["device_map"] = "auto"
            else:
                model_kwargs["torch_dtype"] = torch.float32

            model = AutoModelForCausalLM.from_pretrained(
                self._model_name,
                **model_kwargs,
            ).eval()

            if self._device.type != "cuda":
                model.to(self._device)
            return model

        load_mode = self._config.generator_load_mode
        torch_dtype = _resolve_torch_dtype(self._config.generator_dtype)

        model_kwargs: dict[str, Any] = {
            "local_files_only": True,
            "trust_remote_code": True,
        }

        if load_mode == "standard":
            if torch_dtype is None:
                model_kwargs["torch_dtype"] = (
                    torch.float16 if self._device.type == "cuda" else torch.float32
                )
            else:
                model_kwargs["torch_dtype"] = torch_dtype

            if self._device.type == "cuda" and self._config.generator_device_map is not None:
                model_kwargs["device_map"] = self._config.generator_device_map

            model = AutoModelForCausalLM.from_pretrained(
                self._model_name,
                **model_kwargs,
            ).eval()

            if self._device.type != "cuda":
                model.to(self._device)
            return model

        if self._device.type != "cuda":
            raise RuntimeError(
                f"{load_mode} requested, but CUDA is not available. "
                "Use generator_load_mode='standard' for CPU-only execution."
            )

        if load_mode == "bnb_8bit":
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_enable_fp32_cpu_offload=self._config.bnb_int8_enable_fp32_cpu_offload,
            )
            model_kwargs["device_map"] = self._config.generator_device_map or "auto"
            return AutoModelForCausalLM.from_pretrained(
                self._model_name,
                **model_kwargs,
            ).eval()

        if load_mode == "bnb_4bit":
            compute_dtype = torch_dtype or torch.bfloat16
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type=self._config.bnb_4bit_quant_type,
                bnb_4bit_use_double_quant=self._config.bnb_4bit_use_double_quant,
                bnb_4bit_compute_dtype=compute_dtype,
            )
            model_kwargs["device_map"] = self._config.generator_device_map or "auto"
            return AutoModelForCausalLM.from_pretrained(
                self._model_name,
                **model_kwargs,
            ).eval()

        raise ValueError(f"Unsupported generator_load_mode: {load_mode}")

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

    def _apply_chat_template(
        self,
        messages: list[dict[str, str]],
        *,
        enable_thinking: bool,
    ) -> str:
        """Build a Qwen chat prompt with explicit thinking control."""
        return self._tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
        )

    def build_prompt(
            self,
            *,
            query: str,
            system_prompt: str,
            answer_instruction: str,
            context: str | None = None,
            enable_thinking: bool = False
    ) -> str:
        """Build a chat prompt for grounded or direct answering."""
        if context:
            user_content = (
                f"Question:\n{query}\n\n"
                f"Retrieved context:\n{context}\n\n"
                f"{answer_instruction}"
            )
        else:
            user_content = (
                f"Request:\n{query}\n\n"
                f"{answer_instruction}"
            )

        messages = [
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": user_content,
            },
        ]

        if hasattr(self._tokenizer, "apply_chat_template"):
            return self._tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=enable_thinking,
            )

        return (
            "System:\n"
            f"{system_prompt}\n\n"
            "User:\n"
            f"{user_content}\n\n"
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
        enable_thinking: bool = False,
        return_thinking: bool = False,
    ) -> GeneratedText:
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
        text = self._tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

        if enable_thinking:
            thinking_content, answer, thinking_finished = self._split_reasoning_text(text)
            if return_thinking:
                return GeneratedText(
                    answer=answer,
                    thinking_content=thinking_content,
                    thinking_finished=thinking_finished,
                )
            return GeneratedText(
                answer=answer,
                thinking_content=None,
                thinking_finished=thinking_finished,
            )

        return GeneratedText(
            answer=self._strip_reasoning(text),
            thinking_content=None,
            thinking_finished=False,
        )

    def stream_from_prompt(
        self,
        prompt: str,
        *,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        repetition_penalty: float,
        enable_thinking: bool = False,
        stream_thinking: bool = False,
    ) -> Iterator[StreamEvent]:
        """Yield generated text chunks for a prepared prompt."""
        inputs = self._tokenizer(prompt, return_tensors="pt")
        inputs = {key: value.to(self._model.device) for key, value in inputs.items()}

        do_sample = temperature > 0
        streamer = TextIteratorStreamer(
            self._tokenizer,
            skip_prompt=True,
            skip_special_tokens=True,
            timeout=60.0,
        )

        generate_kwargs: dict[str, Any] = {
            **inputs,
            "max_new_tokens": max_new_tokens,
            "do_sample": do_sample,
            "repetition_penalty": repetition_penalty,
            "pad_token_id": self._tokenizer.pad_token_id,
            "eos_token_id": self._tokenizer.eos_token_id,
            "streamer": streamer,
        }
        if do_sample:
            generate_kwargs["temperature"] = temperature
            generate_kwargs["top_p"] = top_p

        def _run_generation() -> None:
            with torch.inference_mode():
                self._model.generate(**generate_kwargs)

        worker = Thread(target=_run_generation)
        worker.start()

        buffer = ""
        thinking_done = not enable_thinking

        for chunk in streamer:
            if not chunk:
                continue

            if thinking_done:
                cleaned = chunk.replace("<think>", "").replace("</think>", "")
                if cleaned:
                    yield StreamEvent(kind="answer_token", text=cleaned)
                continue

            buffer += chunk
            buffer = buffer.replace("<think>", "")

            end_index = buffer.find("</think>")
            if end_index >= 0:
                thinking_part = buffer[:end_index]
                answer_part = buffer[end_index + len("</think>"):]

                if stream_thinking and thinking_part:
                    yield StreamEvent(kind="thinking_token", text=thinking_part)

                yield StreamEvent(kind="thinking_done")
                thinking_done = True

                if answer_part:
                    yield StreamEvent(kind="answer_token", text=answer_part)

                buffer = ""
                continue

            safe_length = max(0, len(buffer) - len("</think>"))
            if safe_length > 0:
                safe_text = buffer[:safe_length]
                buffer = buffer[safe_length:]
                if stream_thinking and safe_text:
                    yield StreamEvent(kind="thinking_token", text=safe_text)

        if buffer:
            if thinking_done:
                cleaned = buffer.replace("<think>", "").replace("</think>", "")
                if cleaned:
                    yield StreamEvent(kind="answer_token", text=cleaned)
            elif stream_thinking:
                yield StreamEvent(kind="thinking_token", text=buffer.replace("<think>", ""))

        worker.join()

    def generate_answer(
        self,
        *,
        query: str,
        system_prompt: str,
        answer_instruction: str,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        repetition_penalty: float,
        context: str | None = None,
        enable_thinking: bool = False,
        return_thinking: bool = False,
    ) -> GeneratedText:
        """Generate an answer with or without retrieved context."""
        prompt = self.build_prompt(
            query=query,
            context=context,
            system_prompt=system_prompt,
            answer_instruction=answer_instruction,
            enable_thinking=enable_thinking,
        )
        return self.generate_from_prompt(
            prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            enable_thinking=enable_thinking,
            return_thinking=return_thinking,
        )

    def build_chat_prompt(
            self,
            *,
            query: str,
            system_prompt: str,
            chat_instruction: str,
            enable_thinking: bool = False
    ) -> str:
        """Build a normal chat prompt without retrieved context."""
        messages = [
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": (
                    f"Request:\n{query}\n\n"
                    f"{chat_instruction}"
                ),
            },
        ]

        if hasattr(self._tokenizer, "apply_chat_template"):
            return self._tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=enable_thinking,
            )

        return (
            "System:\n"
            f"{system_prompt}\n\n"
            "User:\n"
            f"Request:\n{query}\n\n"
            f"{chat_instruction}\n\n"
            "Assistant:\n"
        )

    def generate_chat_answer(
        self,
        *,
        query: str,
        system_prompt: str,
        chat_instruction: str,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        repetition_penalty: float,
        enable_thinking: bool = False,
        return_thinking: bool = False,
    ) -> GeneratedText:
        """Generate a direct answer without retrieval context."""
        prompt = self.build_chat_prompt(
            query=query,
            system_prompt=system_prompt,
            chat_instruction=chat_instruction,
            enable_thinking=enable_thinking,
        )
        return self.generate_from_prompt(
            prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            enable_thinking=enable_thinking,
            return_thinking=return_thinking,
        )

    def generate_structured_json(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        max_new_tokens: int,
        temperature: float = 0.0,
        top_p: float = 1.0,
        repetition_penalty: float = 1.0,
        enable_thinking: bool = False,
    ) -> dict[str, Any]:
        """Generate a JSON object from the model and parse the first valid object found."""
        prompt = self.build_messages_prompt(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            enable_thinking=enable_thinking,
        )
        generated = self.generate_from_prompt(
            prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            enable_thinking=enable_thinking,
            return_thinking=False,
        )
        return self._extract_json(generated.answer)
    
    
    def generate_query_expansions(
        self,
        *,
        query: str,
        config: RewriteConfig,
    ) -> list[str]:
        """Generate normalized query expansions for retrieval fallback."""
        payload = self.generate_structured_json(
            system_prompt=(
                f"{config.system_prompt} "
                "Return exactly one JSON object and nothing else."
            ),
            user_prompt=(
                f"User query:\n{query}\n\n"
                f"{config.user_instruction}\n\n"
                f"Generate exactly {config.max_rewrites} unique rewrites.\n"
                "The first rewrite must be the original query exactly as written.\n"
                "Keep every rewrite short, standalone, and search-ready.\n"
                "Do not number items. Do not include explanations."
            ),
            max_new_tokens=config.max_new_tokens,
            temperature=config.temperature,
            top_p=config.top_p,
            repetition_penalty=config.repetition_penalty,
            enable_thinking=(config.reasoning_mode == "think"),
        )
        return self._normalize_query_expansions(
            query=query,
            rewrites=payload.get("rewrites"),
            max_rewrites=config.max_rewrites,
        )


    @staticmethod
    def _normalize_query_expansions(
        *,
        query: str,
        rewrites: Any,
        max_rewrites: int,
    ) -> list[str]:
        """Normalize expansion rewrites and keep the original query first."""
        original = " ".join(str(query or "").split()).strip()
        if not original:
            return []

        ordered: list[str] = [original]
        seen = {original.casefold()}

        if isinstance(rewrites, list):
            for item in rewrites:
                text = " ".join(str(item or "").split()).strip()
                if not text:
                    continue
                key = text.casefold()
                if key in seen:
                    continue
                seen.add(key)
                ordered.append(text)
                if len(ordered) >= max_rewrites:
                    break

        return ordered[:max_rewrites]

    def build_messages_prompt(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        enable_thinking: bool = False
    ) -> str:
        """Build a generic chat-form prompt from explicit system and user content."""

        messages = [
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": user_prompt,
            },
        ]

        if hasattr(self._tokenizer, "apply_chat_template"):
            return self._tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=enable_thinking,
            )

        return (
            "System:\n"
            f"{system_prompt}\n\n"
            "User:\n"
            f"{user_prompt}\n\n"
            "Assistant:\n"
        )
    
    @staticmethod
    def _split_reasoning_text(text: str) -> tuple[str | None, str, bool]:
        """Split Qwen output into thinking text and final answer."""
        cleaned = text.strip()

        if "</think>" in cleaned:
            thinking_raw, answer = cleaned.rsplit("</think>", 1)
            thinking = thinking_raw.replace("<think>", "").strip() or None
            return thinking, answer.strip(), True

        if "<think>" in cleaned:
            thinking = cleaned.replace("<think>", "").strip() or None
            return thinking, "", False

        return None, cleaned, False
    
    @staticmethod
    def _strip_reasoning(text: str) -> str:
        """Remove visible reasoning tags if they are present."""
        cleaned = re.sub(
            r"<think>.*?</think>",
            "",
            text,
            flags=re.DOTALL | re.IGNORECASE,
        )
        cleaned = cleaned.replace("<think>", "").replace("</think>", "")
        return cleaned.strip()

    @staticmethod
    def _extract_json(text: str) -> dict[str, Any]:
        """Extract and parse the first top-level JSON object from model output."""
        start = text.find("{")
        if start < 0:
            raise ValueError("Model output does not contain a JSON object")

        depth = 0
        end = -1
        in_string = False
        escape = False

        for index in range(start, len(text)):
            char = text[index]

            if in_string:
                if escape:
                    escape = False
                elif char == "\\":
                    escape = True
                elif char == '"':
                    in_string = False
                continue

            if char == '"':
                in_string = True
                continue

            if char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    end = index + 1
                    break

        if end < 0:
            raise ValueError("Model output contains an incomplete JSON object")

        candidate = text[start:end]
        try:
            import json
            payload = json.loads(candidate)
        except Exception as exc:
            raise ValueError("Failed to parse JSON from model output") from exc

        if not isinstance(payload, dict):
            raise ValueError("Structured output must be a JSON object")
        return payload
