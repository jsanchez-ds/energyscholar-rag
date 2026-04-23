"""
Provider-agnostic LLM client.

Exposes a single `LLMClient` that wraps:
  - Groq (free tier, OpenAI-compatible endpoint)
  - Anthropic (native SDK)
  - OpenAI (native SDK)
  - OpenRouter (OpenAI-compatible endpoint)

Switching providers is a one-line change in `.env` (`LLM_PROVIDER=...`).
Typed messages + retry are handled uniformly regardless of the backend.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from tenacity import retry, stop_after_attempt, wait_exponential

from src.utils.config import get_env, load_yaml_config
from src.utils.logging import get_logger

log = get_logger(__name__)


Role = Literal["system", "user", "assistant"]


@dataclass
class Message:
    role: Role
    content: str

    def to_openai(self) -> dict[str, str]:
        return {"role": self.role, "content": self.content}


@dataclass
class Completion:
    text: str
    model: str
    provider: str
    prompt_tokens: int | None = None
    completion_tokens: int | None = None


class LLMClient:
    """Thin typed wrapper that normalises completions across providers."""

    # Endpoints that speak the OpenAI protocol
    OPENAI_COMPATIBLE_BASE_URLS = {
        "groq": "https://api.groq.com/openai/v1",
        "openai": "https://api.openai.com/v1",
        "openrouter": "https://openrouter.ai/api/v1",
    }

    def __init__(
        self,
        provider: str | None = None,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> None:
        env = get_env()
        cfg = load_yaml_config()

        self.provider = provider or env.llm_provider
        provider_cfg = cfg["generation"]["providers"][self.provider]

        self.model = model or env.llm_model_override or provider_cfg["model"]
        self.temperature = temperature if temperature is not None else provider_cfg["temperature"]
        self.max_tokens = max_tokens if max_tokens is not None else provider_cfg["max_tokens"]
        self.api_key = env.active_api_key

        self._client: Any = None          # lazy-instantiated

    def _init_client(self) -> None:
        if self._client is not None:
            return
        if self.provider == "anthropic":
            from anthropic import Anthropic

            self._client = Anthropic(api_key=self.api_key)
        else:
            from openai import OpenAI

            base_url = self.OPENAI_COMPATIBLE_BASE_URLS[self.provider]
            self._client = OpenAI(api_key=self.api_key, base_url=base_url)

    # ── Public API ───────────────────────────────────────────────────────────
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=30))
    def chat(self, messages: list[Message]) -> Completion:
        """Synchronous chat completion."""
        self._init_client()
        if self.provider == "anthropic":
            return self._chat_anthropic(messages)
        return self._chat_openai_compatible(messages)

    # ── Per-provider implementations ────────────────────────────────────────
    def _chat_openai_compatible(self, messages: list[Message]) -> Completion:
        resp = self._client.chat.completions.create(
            model=self.model,
            messages=[m.to_openai() for m in messages],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        choice = resp.choices[0]
        usage = getattr(resp, "usage", None)
        log.info(
            "llm.chat.done",
            provider=self.provider,
            model=self.model,
            finish_reason=choice.finish_reason,
            in_tokens=getattr(usage, "prompt_tokens", None),
            out_tokens=getattr(usage, "completion_tokens", None),
        )
        return Completion(
            text=choice.message.content or "",
            model=self.model,
            provider=self.provider,
            prompt_tokens=getattr(usage, "prompt_tokens", None),
            completion_tokens=getattr(usage, "completion_tokens", None),
        )

    def _chat_anthropic(self, messages: list[Message]) -> Completion:
        # Anthropic uses a separate `system` parameter, not a message with role="system"
        system_msgs = [m.content for m in messages if m.role == "system"]
        non_system = [
            {"role": m.role, "content": m.content} for m in messages if m.role != "system"
        ]
        resp = self._client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            system="\n".join(system_msgs) if system_msgs else None,
            messages=non_system,
        )
        text = "".join(block.text for block in resp.content if getattr(block, "text", None))
        log.info(
            "llm.chat.done",
            provider=self.provider,
            model=self.model,
            stop_reason=resp.stop_reason,
            in_tokens=resp.usage.input_tokens,
            out_tokens=resp.usage.output_tokens,
        )
        return Completion(
            text=text,
            model=self.model,
            provider=self.provider,
            prompt_tokens=resp.usage.input_tokens,
            completion_tokens=resp.usage.output_tokens,
        )
