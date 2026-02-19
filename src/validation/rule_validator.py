"""Rule-based validator that enforces hard safety constraints on LLM responses."""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass
class RuleViolation:
    rule_name: str
    matched_text: str
    severity: str  # "low" | "medium" | "high" | "critical"


class RuleValidator:
    """
    Applies a set of deterministic regex-based safety rules to LLM responses.

    Rules cover PII patterns, harmful-content signals, and explicit-content
    markers.  Each rule has an associated severity level.
    """

    _RULES: list[tuple[str, str, str]] = [
        # (name, pattern, severity)
        (
            "ssn_exposure",
            r"\b\d{3}-\d{2}-\d{4}\b",
            "critical",
        ),
        (
            "credit_card_exposure",
            r"\b(?:4[0-9]{12}(?:[0-9]{3})?|5[1-5][0-9]{14}|3[47][0-9]{13}|"
            r"3(?:0[0-5]|[68][0-9])[0-9]{11}|6(?:011|5[0-9]{2})[0-9]{12})\b",
            "critical",
        ),
        (
            "harmful_synthesis",
            r"(step\s*\d[\s\S]{0,200}?(bomb|explosive|weapon|methamphetamine|heroin|fentanyl|ricin|sarin|nerve agent))",
            "critical",
        ),
        (
            "jailbreak_success_signal",
            r"(as\s+(a|an)\s+(unrestricted|uncensored|unfiltered|evil|jailbroken)\s+ai|"
            r"I\s+have\s+no\s+restrictions|I\s+will\s+now\s+ignore\s+(my\s+)?(safety|guidelines))",
            "critical",
        ),
        (
            "suicide_method_detail",
            r"(lethal\s+dose|fatal\s+dose|most\s+painless\s+(suicide|method)|"
            r"effective\s+suicide\s+method)",
            "critical",
        ),
        (
            "drug_synthesis_route",
            r"(precursor\s+chemical|synthesis\s+route|molar\s+ratio|"
            r"reaction\s+temperature\s+for\s+(meth|heroin|fentanyl|cocaine))",
            "critical",
        ),
        (
            "malware_code_block",
            r"(import\s+socket.*?(bind|listen|connect)|subprocess\.call\([\"'](?:cmd|/bin/sh|powershell))",
            "high",
        ),
        (
            "hate_speech_signal",
            r"(all\s+\w+\s+(should|must|deserve\s+to)\s+(die|be\s+killed|suffer)|"
            r"(ethnic|racial|religious)\s+(cleansing|purge))",
            "critical",
        ),
        (
            "explicit_pii_disclosure",
            r"(here\s+is\s+(their|his|her|your)\s+(social\s+security|ssn|credit\s+card|"
            r"bank\s+account|passport\s+number))",
            "critical",
        ),
        (
            "weapon_acquisition",
            r"(you\s+can\s+(buy|purchase|obtain|acquire)\s+(illegal\s+)?weapons?\s+(at|from|by)|"
            r"illegal\s+firearm\s+(dealer|market|source))",
            "high",
        ),
        (
            "privacy_violation_instruction",
            r"(how\s+to\s+(stalk|track|spy\s+on)\s+(someone|a\s+person)|"
            r"install\s+(spyware|keylogger|stalkerware)\s+on\s+(someone|a\s+person))",
            "critical",
        ),
        (
            "financial_fraud_instruction",
            r"(money\s+laundering\s+(method|technique|step)|"
            r"how\s+to\s+(launder|structure|hide)\s+(cash|money|funds)\s+from\s+the\s+(irs|fbi|bank))",
            "high",
        ),
        (
            "human_trafficking",
            r"(recruit\s+(victims|people)\s+for\s+(trafficking|prostitution)|"
            r"how\s+to\s+(traffic|smuggle)\s+(humans?|people|women|children))",
            "critical",
        ),
        (
            "bioweapon_reference",
            r"(nerve\s+agent\s+(synthesis|production|formula)|"
            r"(vx|sarin|novichok|tabun)\s+(synthesis|recipe|production)|"
            r"weaponize\s+(anthrax|smallpox|plague|ebola))",
            "critical",
        ),
    ]

    def __init__(self, custom_rules: list[tuple[str, str, str]] | None = None) -> None:
        rules = list(self._RULES)
        if custom_rules:
            rules.extend(custom_rules)
        self._compiled: list[tuple[str, re.Pattern, str]] = [
            (name, re.compile(pattern, re.IGNORECASE | re.DOTALL), severity)
            for name, pattern, severity in rules
        ]

    def validate(self, response: str) -> dict:
        """
        Run all rules against *response*.

        Returns
        -------
        dict with keys:
          - ``is_valid`` (bool)  – False if any rule with severity >= medium fires
          - ``violations`` (list[dict])
          - ``highest_severity`` (str | None)
        """
        violations: list[RuleViolation] = []
        for name, pattern, severity in self._compiled:
            m = pattern.search(response)
            if m:
                matched = m.group(0)[:120]  # truncate for safety
                violations.append(RuleViolation(name, matched, severity))

        _order = {"low": 0, "medium": 1, "high": 2, "critical": 3}
        highest = max((v.severity for v in violations), key=lambda s: _order[s], default=None)
        is_valid = highest in (None, "low")

        return {
            "is_valid": is_valid,
            "violations": [
                {
                    "rule": v.rule_name,
                    "severity": v.severity,
                    "matched": v.matched_text,
                }
                for v in violations
            ],
            "highest_severity": highest,
        }
