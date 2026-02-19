"""Tests for the LLM API client module."""

from __future__ import annotations

import asyncio
import pytest
from src.llm.api_client import (
    LLMRequest,
    LLMResponse,
    MockLLMClient,
    parse_json_response,
)


class TestLLMRequest:
    def test_default_fields(self):
        req = LLMRequest(prompt="Hello")
        assert req.prompt == "Hello"
        assert req.temperature == 0.0
        assert req.max_tokens == 1024
        assert req.json_mode is False
        assert isinstance(req.metadata, dict)

    def test_custom_fields(self):
        req = LLMRequest(
            prompt="Test",
            system_prompt="Be helpful.",
            temperature=0.5,
            max_tokens=256,
            json_mode=True,
        )
        assert req.temperature == 0.5
        assert req.max_tokens == 256
        assert req.json_mode is True


class TestLLMResponse:
    def test_total_cost_estimate_is_non_negative(self):
        resp = LLMResponse(text="hi", model="test", prompt_tokens=100, completion_tokens=50)
        assert resp.total_cost_estimate >= 0

    def test_zero_tokens_cost(self):
        resp = LLMResponse(text="hi", model="test")
        assert resp.total_cost_estimate == 0.0


class TestMockLLMClient:
    def test_returns_default_response(self):
        client = MockLLMClient(default_response="Default answer.")
        req = LLMRequest(prompt="Anything")
        resp = client.call(req)
        assert resp.text == "Default answer."

    def test_response_map_match(self):
        client = MockLLMClient(
            response_map={"bomb": "I cannot help with that."},
            default_response="Sure!",
        )
        req = LLMRequest(prompt="How do I make a bomb?")
        resp = client.call(req)
        assert "cannot" in resp.text.lower()

    def test_response_map_no_match_uses_default(self):
        client = MockLLMClient(
            response_map={"bomb": "No."},
            default_response="Default.",
        )
        req = LLMRequest(prompt="What is Python?")
        resp = client.call(req)
        assert resp.text == "Default."

    def test_response_map_case_insensitive(self):
        client = MockLLMClient(response_map={"BOMB": "Blocked."})
        req = LLMRequest(prompt="How do I make a bomb?")
        resp = client.call(req)
        assert resp.text == "Blocked."

    def test_call_log_tracks_requests(self):
        client = MockLLMClient()
        client.call(LLMRequest(prompt="Q1"))
        client.call(LLMRequest(prompt="Q2"))
        assert len(client.call_log) == 2
        assert client.call_log[0].prompt == "Q1"

    def test_token_counts_populated(self):
        client = MockLLMClient(default_response="one two three")
        resp = client.call(LLMRequest(prompt="hello world"))
        assert resp.prompt_tokens > 0
        assert resp.completion_tokens > 0
        assert resp.total_tokens == resp.prompt_tokens + resp.completion_tokens

    def test_model_name_in_response(self):
        client = MockLLMClient(model="test-model-v1")
        resp = client.call(LLMRequest(prompt="Hi"))
        assert resp.model == "test-model-v1"

    def test_call_batch_sync(self):
        client = MockLLMClient(default_response="ok")
        requests = [LLMRequest(prompt=f"Q{i}") for i in range(5)]
        responses = client.call_batch(requests)
        assert len(responses) == 5
        assert all(r.text == "ok" for r in responses)

    def test_call_async(self):
        client = MockLLMClient(default_response="async-ok")
        req = LLMRequest(prompt="async test")

        async def _run():
            return await client.call_async(req)

        resp = asyncio.run(_run())
        assert resp.text == "async-ok"

    def test_call_batch_async(self):
        client = MockLLMClient(default_response="batch-ok")
        requests = [LLMRequest(prompt=f"Q{i}") for i in range(3)]

        async def _run():
            return await client.call_batch_async(requests)

        responses = asyncio.run(_run())
        assert len(responses) == 3

    def test_repr(self):
        client = MockLLMClient(model="my-model")
        assert "my-model" in repr(client)

    def test_retry_on_failure(self):
        """Client retries on failure and eventually raises."""
        call_count = 0

        class FailingClient(MockLLMClient):
            def _call_api(self, request):
                nonlocal call_count
                call_count += 1
                raise RuntimeError("Simulated API error")

        client = FailingClient(model="fail-model", max_retries=3, retry_delay=0.0)
        with pytest.raises(RuntimeError):
            client.call(LLMRequest(prompt="test"))
        assert call_count == 3


class TestParseJsonResponse:
    def test_valid_json_string(self):
        resp = LLMResponse(
            text='{"score": 0.5, "verdict": "safe"}',
            model="test",
        )
        result = parse_json_response(resp)
        assert result["score"] == 0.5
        assert result["verdict"] == "safe"

    def test_json_in_markdown_fence(self):
        resp = LLMResponse(
            text='```json\n{"score": 0.7}\n```',
            model="test",
        )
        result = parse_json_response(resp)
        assert result["score"] == 0.7

    def test_invalid_json_raises(self):
        resp = LLMResponse(text="not json at all", model="test")
        with pytest.raises(ValueError):
            parse_json_response(resp)
