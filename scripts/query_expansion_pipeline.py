from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


def _add_project_root_to_path() -> Path:
    """Resolve the repository root and add it to sys.path."""
    current_file = Path(__file__).resolve()
    candidates = [current_file.parent, *current_file.parents]

    for candidate in candidates:
        if (candidate / "src").is_dir():
            sys.path.insert(0, str(candidate))
            return candidate

    fallback_root = current_file.parent
    sys.path.insert(0, str(fallback_root))
    return fallback_root


PROJECT_ROOT = _add_project_root_to_path()

from src.config.generator_config import GeneratorConfig
from src.config.retrieval_control_config import RewriteConfig
from src.retrieval.qwen_models import LocalQwenGenerator


@dataclass(slots=True)
class QueryExpansionConfig:
    """Runtime settings for standalone query expansion."""

    query_count: int = 8
    max_new_tokens: int = 256
    temperature: float = 0.2
    top_p: float = 0.9
    repetition_penalty: float = 1.02
    max_attempts: int = 2

    def validate(self) -> None:
        """Validate query expansion settings."""
        if not 5 <= self.query_count <= 10:
            raise ValueError("query_count must be between 5 and 10")
        if self.max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be greater than 0")
        if self.temperature < 0:
            raise ValueError("temperature must be non-negative")
        if not 0 < self.top_p <= 1:
            raise ValueError("top_p must be in the range (0, 1]")
        if self.repetition_penalty <= 0:
            raise ValueError("repetition_penalty must be greater than 0")
        if self.max_attempts <= 0:
            raise ValueError("max_attempts must be greater than 0")


@dataclass(slots=True)
class QueryExpansionResult:
    """Expanded query output."""

    query: str
    expanded_queries: list[str]
    keywords: list[str]
    entities: list[str]
    raw_payload: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""
        return {
            "query": self.query,
            "expanded_queries": self.expanded_queries,
            "keywords": self.keywords,
            "entities": self.entities,
            "raw_payload": self.raw_payload,
        }


class QueryExpansionPipeline:
    """Generate retrieval-oriented query expansions with the current local generator."""

    def __init__(
        self,
        *,
        generator: LocalQwenGenerator,
        rewrite_config: RewriteConfig,
        expansion_config: QueryExpansionConfig,
    ) -> None:
        expansion_config.validate()
        self._generator = generator
        self._rewrite_config = rewrite_config
        self._expansion_config = expansion_config

    @classmethod
    def from_current_generator(
        cls,
        *,
        project_root: Path,
        load_preset: str | None,
        query_count: int,
    ) -> "QueryExpansionPipeline":
        """Build the pipeline from the project's active generator stack."""
        generator_config = GeneratorConfig(project_root=project_root)
        if load_preset:
            generator_config = generator_config.with_load_preset(load_preset)

        generator = LocalQwenGenerator.from_config(generator_config)
        rewrite_config = RewriteConfig()
        expansion_config = QueryExpansionConfig(
            query_count=query_count,
            max_new_tokens=max(256, rewrite_config.max_new_tokens),
        )
        return cls(
            generator=generator,
            rewrite_config=rewrite_config,
            expansion_config=expansion_config,
        )

    def expand(self, query: str) -> QueryExpansionResult:
        """Expand one user query into multiple retrieval probes."""
        normalized_query = self._normalize_text(query)
        if not normalized_query:
            raise ValueError("query must not be empty")

        payload: dict[str, Any] = {}
        expanded_queries: list[str] = []

        for attempt in range(1, self._expansion_config.max_attempts + 1):
            payload = self._generate_payload(normalized_query, attempt, expanded_queries)
            expanded_queries = self._normalize_rewrites(
                query=normalized_query,
                rewrites=payload.get("rewrites"),
            )
            if len(expanded_queries) >= self._expansion_config.query_count:
                break

        if len(expanded_queries) < self._expansion_config.query_count:
            expanded_queries = self._backfill_rewrites(normalized_query, expanded_queries)

        keywords = self._normalize_string_list(payload.get("keywords"))
        entities = self._normalize_string_list(payload.get("entities"))

        return QueryExpansionResult(
            query=normalized_query,
            expanded_queries=expanded_queries[: self._expansion_config.query_count],
            keywords=keywords,
            entities=entities,
            raw_payload=payload,
        )

    def _generate_payload(
        self,
        query: str,
        attempt: int,
        current_rewrites: list[str],
    ) -> dict[str, Any]:
        """Generate one structured JSON payload from the current generator."""
        return self._generator.generate_structured_json(
            system_prompt=self._build_system_prompt(),
            user_prompt=self._build_user_prompt(
                query=query,
                attempt=attempt,
                current_rewrites=current_rewrites,
            ),
            max_new_tokens=self._expansion_config.max_new_tokens,
            temperature=self._expansion_config.temperature,
            top_p=self._expansion_config.top_p,
            repetition_penalty=self._expansion_config.repetition_penalty,
            enable_thinking=False,
        )

    def _build_system_prompt(self) -> str:
        """Build the system prompt for query expansion."""
        return (
            f"{self._rewrite_config.system_prompt} "
            "You are generating retrieval expansions for an offline document intelligence system. "
            "Return exactly one JSON object and nothing else."
        )

    def _build_user_prompt(
        self,
        *,
        query: str,
        attempt: int,
        current_rewrites: list[str],
    ) -> str:
        """Build the user prompt for structured query expansion."""
        base_instruction = (
            f"User query:\n{query}\n\n"
            f"{self._rewrite_config.user_instruction}\n\n"
            f"Generate exactly {self._expansion_config.query_count} unique rewrites.\n"
            "The first rewrite must be the original query exactly as written.\n"
            "Keep every rewrite standalone and search-ready.\n"
            "Vary across paraphrase, keyword-heavy form, explicit form, section-oriented form, and alternate terminology.\n"
            "Do not number items. Do not include explanations.\n"
            'Return JSON with fields: {"rewrites": [...], "keywords": [...], "entities": [...]}.'
        )

        if attempt == 1 or not current_rewrites:
            return base_instruction

        existing = json.dumps(current_rewrites, ensure_ascii=False)
        return (
            f"{base_instruction}\n\n"
            f"The previous output did not provide enough valid unique rewrites.\n"
            f"Current accepted rewrites: {existing}\n"
            "Return a corrected full list with the required total count."
        )

    def _normalize_rewrites(self, *, query: str, rewrites: Any) -> list[str]:
        """Clean, deduplicate, and order model-generated rewrites."""
        normalized = self._normalize_string_list(rewrites)
        ordered: list[str] = [query]
        seen = {query.casefold()}

        for item in normalized:
            if item.casefold() in seen:
                continue
            seen.add(item.casefold())
            ordered.append(item)

        return ordered[: self._expansion_config.query_count]

    def _backfill_rewrites(self, query: str, current_rewrites: list[str]) -> list[str]:
        """Fill missing rewrites deterministically when the model under-produces."""
        templates = (
            "{query}",
            "{query} overview",
            "{query} summary",
            "{query} key points",
            "{query} detailed explanation",
            "{query} requirements",
            "{query} design details",
            "{query} implementation details",
            "{query} architecture",
            "{query} examples",
            "what does the document say about {query}",
            "sections about {query}",
        )

        result = list(current_rewrites) if current_rewrites else [query]
        seen = {item.casefold() for item in result}

        for template in templates:
            candidate = self._normalize_text(template.format(query=query))
            if not candidate or candidate.casefold() in seen:
                continue
            seen.add(candidate.casefold())
            result.append(candidate)
            if len(result) >= self._expansion_config.query_count:
                break

        return result

    @staticmethod
    def _normalize_string_list(value: Any) -> list[str]:
        """Normalize an arbitrary value into a clean string list."""
        if not isinstance(value, list):
            return []

        cleaned: list[str] = []
        seen: set[str] = set()
        for item in value:
            text = QueryExpansionPipeline._normalize_text(item)
            if not text:
                continue
            key = text.casefold()
            if key in seen:
                continue
            seen.add(key)
            cleaned.append(text)
        return cleaned

    @staticmethod
    def _normalize_text(value: Any) -> str:
        """Normalize one text value."""
        return " ".join(str(value or "").split()).strip()


def _build_argument_parser() -> argparse.ArgumentParser:
    """Build the command-line interface."""
    parser = argparse.ArgumentParser(
        description="Expand one query into multiple retrieval-oriented search probes.",
    )
    parser.add_argument(
        "query",
        nargs="?",
        help="Input query to expand.",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=8,
        help="Number of expanded queries to generate. Must be between 5 and 10.",
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=PROJECT_ROOT,
        help="Repository root that contains the src package and models directory.",
    )
    parser.add_argument(
        "--load-preset",
        type=str,
        default=None,
        help="Optional generator load preset such as standard, bnb_8bit, bnb_4bit, or cpu_safe.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print the full JSON payload instead of a numbered list.",
    )
    parser.add_argument(
        "--show-metadata",
        action="store_true",
        help="Print extracted keywords and entities after the rewrites.",
    )
    return parser


def main() -> int:
    """Run the standalone query expansion pipeline."""
    parser = _build_argument_parser()
    args = parser.parse_args()

    query = args.query or input("Query: ").strip()
    if not query:
        parser.error("query must not be empty")

    pipeline = QueryExpansionPipeline.from_current_generator(
        project_root=args.project_root.resolve(),
        load_preset=args.load_preset,
        query_count=args.count,
    )
    result = pipeline.expand(query)

    if args.json:
        print(json.dumps(result.to_dict(), indent=2, ensure_ascii=False))
        return 0

    print(f"Input query: {result.query}\n")
    print("Expanded queries:")
    for index, expanded_query in enumerate(result.expanded_queries, start=1):
        print(f"{index}. {expanded_query}")

    if args.show_metadata:
        print("\nKeywords:")
        if result.keywords:
            for keyword in result.keywords:
                print(f"- {keyword}")
        else:
            print("- None")

        print("\nEntities:")
        if result.entities:
            for entity in result.entities:
                print(f"- {entity}")
        else:
            print("- None")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
