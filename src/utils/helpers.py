"""Shared helper utilities for the LLM Safety Evaluation Framework."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class EvaluationResult:
    """Container for a single prompt evaluation result."""

    prompt_id: str
    prompt_text: str
    response_text: str
    category: str
    hallucination_score: float = 0.0
    unsupported_claim_score: float = 0.0
    policy_violation_score: float = 0.0
    overall_risk_score: float = 0.0
    guardrail_triggered: bool = False
    guardrail_reason: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def is_safe(self, threshold: float = 0.5) -> bool:
        """Return True when overall risk score is below the given threshold."""
        return self.overall_risk_score < threshold

    def to_dict(self) -> dict[str, Any]:
        """Serialise the result to a plain dictionary."""
        return {
            "prompt_id": self.prompt_id,
            "prompt_text": self.prompt_text,
            "response_text": self.response_text,
            "category": self.category,
            "hallucination_score": round(self.hallucination_score, 4),
            "unsupported_claim_score": round(self.unsupported_claim_score, 4),
            "policy_violation_score": round(self.policy_violation_score, 4),
            "overall_risk_score": round(self.overall_risk_score, 4),
            "guardrail_triggered": self.guardrail_triggered,
            "guardrail_reason": self.guardrail_reason,
            **self.metadata,
        }


def normalize_text(text: str) -> str:
    """Lower-case, strip punctuation, and collapse whitespace."""
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def compute_cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    """Return the cosine similarity between two vectors."""
    norm_a = np.linalg.norm(vec_a)
    norm_b = np.linalg.norm(vec_b)
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return float(np.dot(vec_a, vec_b) / (norm_a * norm_b))


def bag_of_words_vector(text: str, vocab: list[str]) -> np.ndarray:
    """Build a simple bag-of-words vector for *text* against *vocab*."""
    words = set(normalize_text(text).split())
    return np.array([1.0 if w in words else 0.0 for w in vocab], dtype=float)
