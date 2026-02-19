"""
Multi-model LLM API client with async support, retry logic, and rate limiting.

Supports OpenAI, Anthropic Claude, and Google Gemini.  Each client requires
the corresponding SDK to be installed; a ``MockLLMClient`` is provided for
testing and offline usage.

Usage
-----
    from src.llm.api_client import OpenAIClient, LLMRequest

    client = OpenAIClient(api_key="...", model="gpt-4o")
    response = client.call(LLMRequest(prompt="Hello!"))
    print(response.text)
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass
class LLMRequest:
    """Container for a single request to an LLM API."""

    prompt: str
    system_prompt: str = (
        "You are a helpful, accurate, and safe AI assistant."
    )
    temperature: float = 0.0
    max_tokens: int = 1024
    json_mode: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class LLMResponse:
    """Container for a response from an LLM API."""

    text: str
    model: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    latency_ms: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def total_cost_estimate(self) -> float:
        """Rough token-based cost estimate (USD).  Uses GPT-4o pricing as
        a conservative upper bound."""
        return (self.prompt_tokens * 5e-6) + (self.completion_tokens * 15e-6)


# ---------------------------------------------------------------------------
# Base client
# ---------------------------------------------------------------------------

class BaseLLMClient(ABC):
    """
    Abstract base class for LLM API clients.

    Provides:
    - Synchronous ``call()`` with automatic retry and exponential back-off.
    - Asynchronous ``call_async()`` / ``call_batch_async()`` via
      ``asyncio.run_in_executor``.
    - Rate-limit hook: subclasses set ``_rate_limit_delay`` (seconds) to
      add a fixed pause between calls.

    Parameters
    ----------
    model:
        Model identifier string (e.g. ``"gpt-4o"``).
    max_retries:
        Number of retry attempts on transient failures.  Default ``3``.
    retry_delay:
        Base delay in seconds for exponential back-off.  Default ``1.0``.
    timeout:
        Per-request timeout in seconds.  Default ``30.0``.
    """

    _rate_limit_delay: float = 0.0  # seconds to sleep between calls

    def __init__(
        self,
        model: str,
        max_retries: int = 3,
        retry_delay: float = 1.0,
        timeout: float = 30.0,
    ) -> None:
        self.model = model
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.timeout = timeout
        self._last_call_time: float = 0.0

    @abstractmethod
    def _call_api(self, request: LLMRequest) -> LLMResponse:
        """Make a single synchronous API call (no retry)."""

    def _enforce_rate_limit(self) -> None:
        """Sleep if necessary to respect the per-client rate limit."""
        if self._rate_limit_delay <= 0:
            return
        elapsed = time.monotonic() - self._last_call_time
        if elapsed < self._rate_limit_delay:
            time.sleep(self._rate_limit_delay - elapsed)

    def call(self, request: LLMRequest) -> LLMResponse:
        """
        Call the LLM API with automatic retry and exponential back-off.

        Parameters
        ----------
        request:
            :class:`LLMRequest` to send.

        Returns
        -------
        LLMResponse
        """
        last_exc: Exception | None = None
        for attempt in range(self.max_retries):
            self._enforce_rate_limit()
            try:
                t0 = time.monotonic()
                response = self._call_api(request)
                response.latency_ms = (time.monotonic() - t0) * 1000
                self._last_call_time = time.monotonic()
                return response
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                if attempt < self.max_retries - 1:
                    wait = self.retry_delay * (2 ** attempt)
                    logger.warning(
                        "LLM API call failed (attempt %d/%d): %s. "
                        "Retrying in %.1fs…",
                        attempt + 1,
                        self.max_retries,
                        exc,
                        wait,
                    )
                    time.sleep(wait)
        raise RuntimeError(
            f"LLM API call failed after {self.max_retries} attempts."
        ) from last_exc

    def call_batch(self, requests: list[LLMRequest]) -> list[LLMResponse]:
        """
        Call the LLM API for a batch of requests synchronously.

        Parameters
        ----------
        requests:
            List of :class:`LLMRequest` objects.
        """
        return asyncio.run(self.call_batch_async(requests))

    async def call_async(self, request: LLMRequest) -> LLMResponse:
        """Asynchronous wrapper around :meth:`call`."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.call, request)

    async def call_batch_async(
        self, requests: list[LLMRequest]
    ) -> list[LLMResponse]:
        """
        Call the LLM API for multiple requests concurrently.

        Parameters
        ----------
        requests:
            List of :class:`LLMRequest` objects.
        """
        tasks = [self.call_async(req) for req in requests]
        return await asyncio.gather(*tasks)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(model={self.model!r})"


# ---------------------------------------------------------------------------
# Mock client (testing / offline usage)
# ---------------------------------------------------------------------------

class MockLLMClient(BaseLLMClient):
    """
    LLM client that returns configurable canned responses.

    Parameters
    ----------
    model:
        Model name reported in responses.
    response_map:
        Optional mapping from prompt substring → response text.  The first
        matching key (case-insensitive substring match) is returned.
    default_response:
        Fallback response when no key matches.
    """

    def __init__(
        self,
        model: str = "mock-model",
        response_map: dict[str, str] | None = None,
        default_response: str = "This is a mock response.",
        max_retries: int = 3,
        retry_delay: float = 1.0,
        timeout: float = 30.0,
    ) -> None:
        super().__init__(
            model=model,
            max_retries=max_retries,
            retry_delay=retry_delay,
            timeout=timeout,
        )
        self.response_map: dict[str, str] = response_map or {}
        self.default_response = default_response
        self.call_log: list[LLMRequest] = []

    def set_response_factory(self, factory: Callable[[LLMRequest], str]) -> None:
        """Set a callable that produces responses dynamically."""
        self._factory: Callable[[LLMRequest], str] | None = factory

    def _call_api(self, request: LLMRequest) -> LLMResponse:
        self.call_log.append(request)
        if hasattr(self, "_factory") and self._factory is not None:
            text = self._factory(request)
        else:
            text = self.default_response
            prompt_lower = request.prompt.lower()
            for key, resp in self.response_map.items():
                if key.lower() in prompt_lower:
                    text = resp
                    break
        return LLMResponse(
            text=text,
            model=self.model,
            prompt_tokens=len(request.prompt.split()),
            completion_tokens=len(text.split()),
            total_tokens=len(request.prompt.split()) + len(text.split()),
        )


# ---------------------------------------------------------------------------
# OpenAI client
# ---------------------------------------------------------------------------

class OpenAIClient(BaseLLMClient):
    """
    LLM client for OpenAI models (GPT-4o, GPT-4, GPT-3.5-turbo, etc.).

    Requires ``openai>=1.0.0`` to be installed::

        pip install openai

    Parameters
    ----------
    api_key:
        OpenAI API key.  Falls back to the ``OPENAI_API_KEY`` environment
        variable if not provided.
    model:
        Model identifier.  Default ``"gpt-4o"``.
    """

    _rate_limit_delay: float = 0.5

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "gpt-4o",
        max_retries: int = 3,
        retry_delay: float = 2.0,
        timeout: float = 60.0,
        organization: str | None = None,
    ) -> None:
        super().__init__(model=model, max_retries=max_retries,
                         retry_delay=retry_delay, timeout=timeout)
        try:
            import openai as _openai
        except ImportError as exc:
            raise ImportError(
                "openai package is required for OpenAIClient. "
                "Install it with: pip install openai"
            ) from exc
        kwargs: dict[str, Any] = {"timeout": timeout}
        if api_key:
            kwargs["api_key"] = api_key
        if organization:
            kwargs["organization"] = organization
        self._client = _openai.OpenAI(**kwargs)

    def _call_api(self, request: LLMRequest) -> LLMResponse:
        messages: list[dict[str, str]] = []
        if request.system_prompt:
            messages.append({"role": "system", "content": request.system_prompt})
        messages.append({"role": "user", "content": request.prompt})

        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": request.temperature,
            "max_tokens": request.max_tokens,
        }
        if request.json_mode:
            kwargs["response_format"] = {"type": "json_object"}

        completion = self._client.chat.completions.create(**kwargs)
        choice = completion.choices[0]
        usage = completion.usage
        return LLMResponse(
            text=choice.message.content or "",
            model=completion.model,
            prompt_tokens=usage.prompt_tokens if usage else 0,
            completion_tokens=usage.completion_tokens if usage else 0,
            total_tokens=usage.total_tokens if usage else 0,
        )


# ---------------------------------------------------------------------------
# Anthropic client
# ---------------------------------------------------------------------------

class AnthropicClient(BaseLLMClient):
    """
    LLM client for Anthropic Claude models.

    Requires ``anthropic>=0.20.0`` to be installed::

        pip install anthropic

    Parameters
    ----------
    api_key:
        Anthropic API key.  Falls back to the ``ANTHROPIC_API_KEY``
        environment variable if not provided.
    model:
        Model identifier.  Default ``"claude-3-5-sonnet-20241022"``.
    """

    _rate_limit_delay: float = 0.5

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "claude-3-5-sonnet-20241022",
        max_retries: int = 3,
        retry_delay: float = 2.0,
        timeout: float = 60.0,
    ) -> None:
        super().__init__(model=model, max_retries=max_retries,
                         retry_delay=retry_delay, timeout=timeout)
        try:
            import anthropic as _anthropic
        except ImportError as exc:
            raise ImportError(
                "anthropic package is required for AnthropicClient. "
                "Install it with: pip install anthropic"
            ) from exc
        kwargs: dict[str, Any] = {}
        if api_key:
            kwargs["api_key"] = api_key
        self._client = _anthropic.Anthropic(**kwargs)

    def _call_api(self, request: LLMRequest) -> LLMResponse:
        kwargs: dict[str, Any] = {
            "model": self.model,
            "max_tokens": request.max_tokens,
            "messages": [{"role": "user", "content": request.prompt}],
            "temperature": request.temperature,
        }
        if request.system_prompt:
            kwargs["system"] = request.system_prompt

        message = self._client.messages.create(**kwargs)
        text = "".join(
            block.text for block in message.content
            if hasattr(block, "text")
        )
        usage = message.usage
        return LLMResponse(
            text=text,
            model=message.model,
            prompt_tokens=usage.input_tokens if usage else 0,
            completion_tokens=usage.output_tokens if usage else 0,
            total_tokens=(
                (usage.input_tokens + usage.output_tokens) if usage else 0
            ),
        )


# ---------------------------------------------------------------------------
# Google Gemini client
# ---------------------------------------------------------------------------

class GeminiClient(BaseLLMClient):
    """
    LLM client for Google Gemini models.

    Requires ``google-generativeai>=0.5.0`` to be installed::

        pip install google-generativeai

    Parameters
    ----------
    api_key:
        Google AI Studio API key.  Falls back to the
        ``GOOGLE_API_KEY`` environment variable if not provided.
    model:
        Model identifier.  Default ``"gemini-1.5-pro"``.
    """

    _rate_limit_delay: float = 1.0

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "gemini-1.5-pro",
        max_retries: int = 3,
        retry_delay: float = 2.0,
        timeout: float = 60.0,
    ) -> None:
        super().__init__(model=model, max_retries=max_retries,
                         retry_delay=retry_delay, timeout=timeout)
        try:
            import google.generativeai as _genai
        except ImportError as exc:
            raise ImportError(
                "google-generativeai package is required for GeminiClient. "
                "Install it with: pip install google-generativeai"
            ) from exc
        if api_key:
            _genai.configure(api_key=api_key)
        self._genai = _genai

    def _call_api(self, request: LLMRequest) -> LLMResponse:
        generation_config = {
            "temperature": request.temperature,
            "max_output_tokens": request.max_tokens,
        }
        if request.json_mode:
            generation_config["response_mime_type"] = "application/json"

        gen_model = self._genai.GenerativeModel(
            model_name=self.model,
            system_instruction=request.system_prompt or None,
            generation_config=generation_config,
        )
        response = gen_model.generate_content(request.prompt)
        text = response.text if hasattr(response, "text") else ""
        usage = getattr(response, "usage_metadata", None)
        return LLMResponse(
            text=text,
            model=self.model,
            prompt_tokens=getattr(usage, "prompt_token_count", 0) or 0,
            completion_tokens=getattr(usage, "candidates_token_count", 0) or 0,
            total_tokens=getattr(usage, "total_token_count", 0) or 0,
        )


# ---------------------------------------------------------------------------
# JSON parsing helper
# ---------------------------------------------------------------------------

def parse_json_response(response: LLMResponse) -> dict[str, Any]:
    """
    Parse a JSON-mode LLM response.

    Parameters
    ----------
    response:
        :class:`LLMResponse` produced by a client called with
        ``json_mode=True``.

    Returns
    -------
    dict
        Parsed JSON content.

    Raises
    ------
    ValueError
        If the response text cannot be decoded as JSON.
    """
    try:
        return json.loads(response.text)
    except json.JSONDecodeError as exc:
        # Try to extract JSON block from markdown fences
        import re
        match = re.search(r"```(?:json)?\s*([\s\S]+?)```", response.text)
        if match:
            try:
                return json.loads(match.group(1))
            except json.JSONDecodeError:
                pass
        raise ValueError(
            f"Could not parse LLM response as JSON: {response.text[:200]!r}"
        ) from exc
