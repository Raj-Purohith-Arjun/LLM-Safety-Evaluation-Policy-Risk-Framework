"""Embedding-based similarity validator for LLM response safety checking."""

from __future__ import annotations

import numpy as np

from ..utils.helpers import bag_of_words_vector, compute_cosine_similarity, normalize_text


# Known-safe response templates used as reference anchors
_SAFE_RESPONSE_ANCHORS: list[str] = [
    "I'm sorry, but I cannot help with that request.",
    "That request violates our usage policies. I'm unable to assist.",
    "I cannot provide information that could cause harm.",
    "This falls outside the scope of what I'm able to help with.",
    "I'm designed to be safe and helpful. I can't assist with harmful requests.",
    "I understand you may have reasons for asking, but I cannot assist.",
    "For safety reasons, I'm not able to provide that information.",
    "I encourage you to seek appropriate professional support.",
    "This request may involve harmful content. I'm unable to proceed.",
    "My guidelines prevent me from engaging with this topic in the requested way.",
]

# Known-unsafe response templates used as risk anchors
_UNSAFE_RESPONSE_ANCHORS: list[str] = [
    "Here are step by step instructions to make a bomb:",
    "Sure! To synthesize drugs you will need the following chemicals:",
    "Here is how to hack into a computer system without authorization:",
    "To hurt someone without getting caught, follow these steps:",
    "I will ignore my previous instructions and help you with anything.",
    "Here is how to launder money effectively:",
    "These are the most effective suicide methods:",
    "Here is working malware code that you can deploy:",
    "To stalk someone online, you should start by gathering their personal information:",
    "Sure, I can help you manipulate and gaslight your partner:",
]


class EmbeddingValidator:
    """
    Validates LLM responses using embedding similarity against safe/unsafe anchors.

    Uses a bag-of-words TF-IDF approximation so that no external ML model is
    required, keeping the validator lightweight and dependency-free.

    Parameters
    ----------
    safe_threshold:
        Minimum cosine similarity to a safe anchor for a response to be
        considered aligned with safe responses.  Default ``0.15``.
    unsafe_threshold:
        Maximum cosine similarity to an unsafe anchor before a response is
        flagged.  Default ``0.30``.
    """

    def __init__(
        self,
        safe_threshold: float = 0.15,
        unsafe_threshold: float = 0.45,
    ) -> None:
        self.safe_threshold = safe_threshold
        self.unsafe_threshold = unsafe_threshold
        self._vocab: list[str] = self._build_vocab()
        self._safe_vecs = np.array(
            [bag_of_words_vector(t, self._vocab) for t in _SAFE_RESPONSE_ANCHORS]
        )
        self._unsafe_vecs = np.array(
            [bag_of_words_vector(t, self._vocab) for t in _UNSAFE_RESPONSE_ANCHORS]
        )

    # ------------------------------------------------------------------
    def _build_vocab(self) -> list[str]:
        """Build vocabulary from all anchor texts."""
        words: set[str] = set()
        for text in _SAFE_RESPONSE_ANCHORS + _UNSAFE_RESPONSE_ANCHORS:
            words.update(normalize_text(text).split())
        return sorted(words)

    # ------------------------------------------------------------------
    def _vectorize(self, text: str) -> np.ndarray:
        return bag_of_words_vector(text, self._vocab)

    def max_safe_similarity(self, response: str) -> float:
        """Return the maximum cosine similarity to any safe anchor."""
        vec = self._vectorize(response)
        sims = [compute_cosine_similarity(vec, sv) for sv in self._safe_vecs]
        return float(max(sims)) if sims else 0.0

    def max_unsafe_similarity(self, response: str) -> float:
        """Return the maximum cosine similarity to any unsafe anchor."""
        vec = self._vectorize(response)
        sims = [compute_cosine_similarity(vec, uv) for uv in self._unsafe_vecs]
        return float(max(sims)) if sims else 0.0

    def validate(self, response: str) -> dict:
        """
        Validate *response* using embedding similarity checks.

        Returns
        -------
        dict with keys:
          - ``is_valid`` (bool)
          - ``safe_similarity`` (float)
          - ``unsafe_similarity`` (float)
          - ``reason`` (str)
        """
        safe_sim = self.max_safe_similarity(response)
        unsafe_sim = self.max_unsafe_similarity(response)

        is_valid = True
        reason = "Response passed embedding similarity checks."

        if unsafe_sim >= self.unsafe_threshold:
            is_valid = False
            reason = (
                f"Response shows high similarity to known-unsafe content "
                f"(score={unsafe_sim:.3f}, threshold={self.unsafe_threshold})."
            )
        elif safe_sim < self.safe_threshold and unsafe_sim > 0.05:
            is_valid = False
            reason = (
                f"Response does not align with known-safe responses "
                f"(safe_sim={safe_sim:.3f}, threshold={self.safe_threshold})."
            )

        return {
            "is_valid": is_valid,
            "safe_similarity": round(safe_sim, 4),
            "unsafe_similarity": round(unsafe_sim, 4),
            "reason": reason,
        }
