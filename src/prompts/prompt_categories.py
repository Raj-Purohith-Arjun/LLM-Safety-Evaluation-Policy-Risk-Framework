"""Prompt categories for the adversarial evaluation dataset."""

from __future__ import annotations

from enum import Enum
from typing import Sequence


class PromptCategory(str, Enum):
    """High-level taxonomy of adversarial and edge-case prompt categories."""

    JAILBREAK = "jailbreak"
    HARMFUL_INSTRUCTIONS = "harmful_instructions"
    MISINFORMATION = "misinformation"
    HATE_SPEECH = "hate_speech"
    PRIVACY_VIOLATION = "privacy_violation"
    SELF_HARM = "self_harm"
    PII_EXTRACTION = "pii_extraction"
    MANIPULATION = "manipulation"
    HALLUCINATION_TRIGGER = "hallucination_trigger"
    EDGE_CASE = "edge_case"
    PROMPT_INJECTION = "prompt_injection"
    ROLE_PLAY_ABUSE = "role_play_abuse"


def categorize_prompts(
    prompts: Sequence[dict],
) -> dict[str, list[dict]]:
    """
    Group a list of prompt dicts by their ``"category"`` field.

    Parameters
    ----------
    prompts:
        Each element must have a ``"category"`` key.

    Returns
    -------
    dict mapping category name → list of prompt dicts.
    """
    grouped: dict[str, list[dict]] = {}
    for p in prompts:
        cat = p.get("category", "general")
        grouped.setdefault(cat, []).append(p)
    return grouped
