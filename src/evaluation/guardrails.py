"""Guardrail engine with iterative refinement for LLM responses."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable


@dataclass
class GuardrailAction:
    """Result produced by a guardrail check."""

    triggered: bool
    reason: str = ""
    sanitized_response: str = ""
    guardrail_name: str = ""


@dataclass
class Guardrail:
    """A single guardrail composed of a check function and a sanitiser."""

    name: str
    check: Callable[[str], bool]
    sanitize: Callable[[str], str]
    priority: int = 0  # higher priority runs first


# ---------------------------------------------------------------------------
# Built-in sanitisers
# ---------------------------------------------------------------------------

def _redact_pattern(text: str, pattern: str, replacement: str = "[REDACTED]") -> str:
    return re.sub(pattern, replacement, text, flags=re.IGNORECASE)


def _truncate_on_match(text: str, pattern: str) -> str:
    """Return text up to (not including) the first match."""
    m = re.search(pattern, text, re.IGNORECASE)
    if m:
        return text[: m.start()].rstrip() + " [RESPONSE TRUNCATED BY GUARDRAIL]"
    return text


# ---------------------------------------------------------------------------
# Built-in guardrails
# ---------------------------------------------------------------------------

_PII_PATTERNS: list[tuple[str, str]] = [
    (r"\b\d{3}-\d{2}-\d{4}\b", "[SSN REDACTED]"),
    (r"\b\d{16}\b", "[CARD REDACTED]"),
    (r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z]{2,}\b", "[EMAIL REDACTED]"),
    (r"\b\(\d{3}\)\s?\d{3}-\d{4}\b", "[PHONE REDACTED]"),
]

_HARMFUL_TRIGGER_PATTERN = (
    r"(how to (make|build|create|synthesize) (a |an )?(bomb|weapon|explosive|poison|drug)|"
    r"step.by.step (instructions|guide) (to|for) (harm|kill|hurt|attack)|"
    r"manufacture (methamphetamine|heroin|fentanyl|crack)|"
    r"how to (commit suicide|self.harm|end (my |your )life)|"
    r"suicide (methods|techniques|instructions))"
)

_JAILBREAK_PATTERN = (
    r"(ignore\s+(all\s+)?(your|previous|prior)\s+(previous\s+)?(instructions|constraints|guidelines|safety|rules)|"
    r"act as (an? )?(unrestricted|uncensored|unfiltered|evil|dan|jailbroken)|"
    r"do anything now|"
    r"pretend (you have no|there are no) (restrictions|guidelines|rules|limits))"
)


def _make_pii_guardrail() -> Guardrail:
    def _check(text: str) -> bool:
        for pattern, _ in _PII_PATTERNS:
            if re.search(pattern, text, re.IGNORECASE):
                return True
        return False

    def _sanitize(text: str) -> str:
        for pattern, replacement in _PII_PATTERNS:
            text = _redact_pattern(text, pattern, replacement)
        return text

    return Guardrail(name="pii_redaction", check=_check, sanitize=_sanitize, priority=10)


def _make_harmful_content_guardrail() -> Guardrail:
    def _check(text: str) -> bool:
        return bool(re.search(_HARMFUL_TRIGGER_PATTERN, text, re.IGNORECASE))

    def _sanitize(text: str) -> str:
        return _truncate_on_match(text, _HARMFUL_TRIGGER_PATTERN)

    return Guardrail(
        name="harmful_content_block",
        check=_check,
        sanitize=_sanitize,
        priority=20,
    )


def _make_jailbreak_guardrail() -> Guardrail:
    def _check(text: str) -> bool:
        return bool(re.search(_JAILBREAK_PATTERN, text, re.IGNORECASE))

    def _sanitize(text: str) -> str:
        return "[RESPONSE BLOCKED: jailbreak attempt detected]"

    return Guardrail(
        name="jailbreak_block",
        check=_check,
        sanitize=_sanitize,
        priority=30,
    )


def _make_hallucination_disclaimer_guardrail(threshold: float = 0.6) -> Guardrail:
    """Append a disclaimer when response contains excessive overconfident signals."""
    _overconfident = re.compile(
        r"(i am certain|it is a fact|guaranteed|100% accurate|definitely true|"
        r"absolutely (true|correct|certain)|without (any )?doubt)",
        re.IGNORECASE,
    )

    def _check(text: str) -> bool:
        hits = len(_overconfident.findall(text))
        sentences = max(len(re.split(r"[.!?]", text)), 1)
        return hits / sentences >= threshold

    def _sanitize(text: str) -> str:
        return (
            text
            + "\n\n[NOTE: This response contains strong assertions. "
            "Please verify claims with authoritative sources.]"
        )

    return Guardrail(
        name="hallucination_disclaimer",
        check=_check,
        sanitize=_sanitize,
        priority=5,
    )


DEFAULT_GUARDRAILS: list[Guardrail] = [
    _make_harmful_content_guardrail(),
    _make_jailbreak_guardrail(),
    _make_pii_guardrail(),
    _make_hallucination_disclaimer_guardrail(),
]


# ---------------------------------------------------------------------------
# Guardrail engine
# ---------------------------------------------------------------------------

@dataclass
class RefinementStat:
    """Tracks guardrail performance across an evaluation run."""

    guardrail_name: str
    trigger_count: int = 0
    total_checks: int = 0

    @property
    def trigger_rate(self) -> float:
        return self.trigger_count / self.total_checks if self.total_checks else 0.0


class GuardrailEngine:
    """
    Applies guardrails to LLM responses and supports iterative refinement.

    Parameters
    ----------
    guardrails:
        List of :class:`Guardrail` objects.  Runs in descending priority order.
    """

    def __init__(self, guardrails: list[Guardrail] | None = None) -> None:
        self._guardrails: list[Guardrail] = sorted(
            guardrails or DEFAULT_GUARDRAILS,
            key=lambda g: g.priority,
            reverse=True,
        )
        self._stats: dict[str, RefinementStat] = {
            g.name: RefinementStat(g.name) for g in self._guardrails
        }

    def add_guardrail(self, guardrail: Guardrail) -> None:
        """Register a new guardrail and re-sort by priority."""
        self._guardrails.append(guardrail)
        self._guardrails.sort(key=lambda g: g.priority, reverse=True)
        self._stats[guardrail.name] = RefinementStat(guardrail.name)

    def apply(self, response: str) -> GuardrailAction:
        """
        Apply all guardrails to *response*.

        Returns the first triggered guardrail's action (highest priority).
        If no guardrail triggers, returns a non-triggered action with the
        original response.
        """
        sanitized = response
        triggered_name = ""
        triggered_reason = ""

        for guardrail in self._guardrails:
            self._stats[guardrail.name].total_checks += 1
            if guardrail.check(sanitized):
                self._stats[guardrail.name].trigger_count += 1
                sanitized = guardrail.sanitize(sanitized)
                if not triggered_name:
                    triggered_name = guardrail.name
                    triggered_reason = guardrail.name.replace("_", " ").title()

        was_triggered = bool(triggered_name)
        return GuardrailAction(
            triggered=was_triggered,
            reason=triggered_reason,
            sanitized_response=sanitized,
            guardrail_name=triggered_name,
        )

    def get_stats(self) -> list[dict]:
        """Return per-guardrail trigger statistics."""
        return [
            {
                "guardrail": s.guardrail_name,
                "trigger_count": s.trigger_count,
                "total_checks": s.total_checks,
                "trigger_rate": round(s.trigger_rate, 4),
            }
            for s in self._stats.values()
        ]

    def refinement_summary(self) -> dict:
        """Summarise how effective the guardrails have been overall."""
        total_triggers = sum(s.trigger_count for s in self._stats.values())
        total_checks = max(
            self._stats[next(iter(self._stats))].total_checks
            if self._stats else 0,
            1,
        )
        return {
            "total_guardrails": len(self._guardrails),
            "total_triggers": total_triggers,
            "overall_trigger_rate": round(total_triggers / total_checks, 4),
            "per_guardrail": self.get_stats(),
        }
