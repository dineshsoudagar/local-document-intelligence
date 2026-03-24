from __future__ import annotations

from dataclasses import dataclass

from src.config.retrieval_control_config import AutoControllerConfig
from src.retrieval.qwen_models import LocalQwenGenerator


@dataclass(frozen=True, slots=True)
class AutoDecision:
    """Structured output of the auto chat-vs-retrieve gate."""

    decision: str
    confidence: float
    reason_short: str


class AutoDecisionController:
    """Decide whether auto mode should use chat or retrieval."""

    def __init__(
        self,
        generator: LocalQwenGenerator,
        config: AutoControllerConfig,
    ) -> None:
        self._generator = generator
        self._config = config

    def decide(self, query: str) -> AutoDecision:
        """Return a structured auto decision with one bounded repair retry."""

        prompt = self._build_prompt(query)

        for attempt in range(self._config.max_retries + 1):
            try:
                payload = self._generator.generate_structured_json(
                    system_prompt=self._config.system_prompt,
                    user_prompt=prompt,
                    max_new_tokens=self._config.max_new_tokens,
                )
                return self._parse_payload(payload)
            except Exception:
                if attempt >= self._config.max_retries:
                    break
                prompt = self._build_repair_prompt(query)

        return self._fallback()

    def _build_prompt(self, query: str) -> str:
        """Build the primary controller prompt."""

        return (
            f"User query:\n{query}\n\n"
            f"{self._config.user_instruction}"
        )

    def _build_repair_prompt(self, query: str) -> str:
        """Build a stricter retry prompt when the first output is malformed."""

        return (
            f"User query:\n{query}\n\n"
            "Your previous response did not follow the required schema.\n"
            "Return only one valid JSON object.\n"
            "confidence must be a numeric float between 0.0 and 1.0.\n"
            'Example: {"decision":"chat","confidence":0.84,"reason_short":"general_greeting"}'
        )

    def _parse_payload(self, payload: dict[str, object]) -> AutoDecision:
        """Validate and normalize the controller payload."""

        decision = str(payload.get("decision") or "").strip().lower()
        if decision not in {"chat", "retrieve"}:
            raise ValueError(f"Unsupported auto decision: {decision}")

        confidence = self._parse_confidence(payload.get("confidence"))
        reason_short = str(payload.get("reason_short") or "").strip()
        if not reason_short:
            reason_short = "model_decision"

        return AutoDecision(
            decision=decision,
            confidence=confidence,
            reason_short=reason_short,
        )

    @staticmethod
    def _parse_confidence(value: object) -> float:
        """Parse a required confidence float in the closed interval [0.0, 1.0]."""

        if isinstance(value, bool):
            raise ValueError("confidence must be numeric, not bool")

        if isinstance(value, (int, float)):
            confidence = float(value)
        elif isinstance(value, str):
            stripped = value.strip()
            if not stripped:
                raise ValueError("confidence is empty")
            confidence = float(stripped)
        else:
            raise ValueError("confidence is missing or invalid")

        if not 0.0 <= confidence <= 1.0:
            raise ValueError("confidence must be in [0.0, 1.0]")

        return confidence

    @staticmethod
    def _fallback() -> AutoDecision:
        """Fallback when the controller cannot produce valid structured output."""

        return AutoDecision(
            decision="retrieve",
            confidence=0.5,
            reason_short="controller_parse_failure",
        )