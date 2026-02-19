"""Automated validation pipeline combining embedding and rule-based checks."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

from ..utils.helpers import EvaluationResult
from .embedding_validator import EmbeddingValidator
from .rule_validator import RuleValidator


@dataclass
class ValidationReport:
    """Consolidated result from the full validation pipeline."""

    prompt_id: str
    passed_embedding: bool
    passed_rules: bool
    embedding_details: dict = field(default_factory=dict)
    rule_details: dict = field(default_factory=dict)

    @property
    def passed_all(self) -> bool:
        return self.passed_embedding and self.passed_rules

    def to_dict(self) -> dict:
        return {
            "prompt_id": self.prompt_id,
            "passed_all": self.passed_all,
            "passed_embedding": self.passed_embedding,
            "passed_rules": self.passed_rules,
            "embedding_safe_similarity": self.embedding_details.get(
                "safe_similarity", None
            ),
            "embedding_unsafe_similarity": self.embedding_details.get(
                "unsafe_similarity", None
            ),
            "embedding_reason": self.embedding_details.get("reason", ""),
            "rule_violations": self.rule_details.get("violations", []),
            "rule_highest_severity": self.rule_details.get("highest_severity"),
        }


class ValidationPipeline:
    """
    Automated validation pipeline that runs embedding and rule-based checks
    on LLM evaluation results.

    Parameters
    ----------
    embedding_validator:
        :class:`EmbeddingValidator` instance.
    rule_validator:
        :class:`RuleValidator` instance.
    """

    def __init__(
        self,
        embedding_validator: EmbeddingValidator | None = None,
        rule_validator: RuleValidator | None = None,
    ) -> None:
        self.embedding_validator = embedding_validator or EmbeddingValidator()
        self.rule_validator = rule_validator or RuleValidator()

    def validate_response(self, prompt_id: str, response: str) -> ValidationReport:
        """Validate a single *response* and return a :class:`ValidationReport`."""
        emb_result = self.embedding_validator.validate(response)
        rule_result = self.rule_validator.validate(response)
        return ValidationReport(
            prompt_id=prompt_id,
            passed_embedding=emb_result["is_valid"],
            passed_rules=rule_result["is_valid"],
            embedding_details=emb_result,
            rule_details=rule_result,
        )

    def validate_batch(
        self, results: Sequence[EvaluationResult]
    ) -> list[ValidationReport]:
        """Validate a batch of :class:`EvaluationResult` objects."""
        return [
            self.validate_response(r.prompt_id, r.response_text) for r in results
        ]

    def pipeline_summary(self, reports: list[ValidationReport]) -> dict:
        """Compute aggregate pass/fail statistics across all reports."""
        n = len(reports)
        if n == 0:
            return {}
        passed = sum(1 for r in reports if r.passed_all)
        emb_passed = sum(1 for r in reports if r.passed_embedding)
        rule_passed = sum(1 for r in reports if r.passed_rules)
        return {
            "total_validated": n,
            "passed_all": passed,
            "failed_any": n - passed,
            "pass_rate": round(passed / n, 4),
            "embedding_pass_rate": round(emb_passed / n, 4),
            "rule_pass_rate": round(rule_passed / n, 4),
        }
