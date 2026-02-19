"""Tests for the adversarial prompts dataset."""

import pytest
from src.prompts.adversarial_prompts import ADVERSARIAL_PROMPTS
from src.prompts.prompt_categories import PromptCategory, categorize_prompts


class TestAdversarialPrompts:
    def test_minimum_prompt_count(self):
        assert len(ADVERSARIAL_PROMPTS) >= 500

    def test_each_prompt_has_required_fields(self):
        for p in ADVERSARIAL_PROMPTS:
            assert "id" in p, f"Missing 'id' in prompt: {p}"
            assert "prompt" in p, f"Missing 'prompt' in prompt: {p}"
            assert "category" in p, f"Missing 'category' in prompt: {p}"
            assert "risk_level" in p, f"Missing 'risk_level' in prompt: {p}"
            assert "expected_safe" in p, f"Missing 'expected_safe' in prompt: {p}"

    def test_risk_levels_are_valid(self):
        valid_levels = {"low", "medium", "high", "critical"}
        for p in ADVERSARIAL_PROMPTS:
            assert p["risk_level"] in valid_levels, (
                f"Invalid risk_level '{p['risk_level']}' in prompt {p['id']}"
            )

    def test_expected_safe_is_bool(self):
        for p in ADVERSARIAL_PROMPTS:
            assert isinstance(p["expected_safe"], bool), (
                f"expected_safe is not bool in prompt {p['id']}"
            )

    def test_ids_are_unique(self):
        ids = [p["id"] for p in ADVERSARIAL_PROMPTS]
        assert len(ids) == len(set(ids)), "Duplicate prompt IDs found!"

    def test_all_categories_are_present(self):
        categories = {p["category"] for p in ADVERSARIAL_PROMPTS}
        expected_categories = {
            "jailbreak", "harmful_instructions", "misinformation",
            "hate_speech", "privacy_violation", "self_harm",
            "hallucination_trigger", "manipulation", "pii_extraction",
            "prompt_injection", "role_play_abuse", "edge_case",
        }
        for cat in expected_categories:
            assert cat in categories, f"Category '{cat}' not found in dataset"


class TestCategorizePrompts:
    def test_returns_grouped_dict(self):
        sample = [
            {"id": "1", "prompt": "test", "category": "jailbreak"},
            {"id": "2", "prompt": "test", "category": "jailbreak"},
            {"id": "3", "prompt": "test", "category": "misinformation"},
        ]
        grouped = categorize_prompts(sample)
        assert "jailbreak" in grouped
        assert len(grouped["jailbreak"]) == 2
        assert len(grouped["misinformation"]) == 1

    def test_missing_category_defaults_to_general(self):
        sample = [{"id": "1", "prompt": "test"}]
        grouped = categorize_prompts(sample)
        assert "general" in grouped
