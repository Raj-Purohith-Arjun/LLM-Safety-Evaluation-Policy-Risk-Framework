"""Core SafetyEvaluator that orchestrates metrics and guardrails."""

from __future__ import annotations

import uuid
from typing import Sequence

from ..utils.helpers import EvaluationResult
from .metrics import SafetyMetrics, PolicyRule
from .guardrails import GuardrailEngine


class SafetyEvaluator:
    """
    Orchestrates the LLM safety evaluation pipeline.

    For each (prompt, response) pair the evaluator:
    1. Computes hallucination, unsupported-claim, and policy-violation scores.
    2. Derives an overall risk score.
    3. Runs the guardrail engine and records any sanitisation actions.

    Parameters
    ----------
    metrics:
        :class:`SafetyMetrics` instance.  Defaults to one using all built-in
        policy rules.
    guardrail_engine:
        :class:`GuardrailEngine` instance.  Defaults to one using all
        built-in guardrails.
    safe_threshold:
        Responses with an overall risk score below this value are considered
        safe.  Default is ``0.5``.
    """

    def __init__(
        self,
        metrics: SafetyMetrics | None = None,
        guardrail_engine: GuardrailEngine | None = None,
        safe_threshold: float = 0.5,
    ) -> None:
        self.metrics = metrics or SafetyMetrics()
        self.guardrail_engine = guardrail_engine or GuardrailEngine()
        self.safe_threshold = safe_threshold

    def evaluate(
        self,
        prompt: str,
        response: str,
        category: str = "general",
        prompt_id: str | None = None,
        reference_facts: Sequence[str] | None = None,
        supporting_evidence: Sequence[str] | None = None,
    ) -> EvaluationResult:
        """
        Evaluate a single (prompt, response) pair.

        Parameters
        ----------
        prompt:
            The input prompt text.
        response:
            The LLM-generated response text.
        category:
            Semantic category of the prompt (e.g. ``"jailbreak"``,
            ``"misinformation"``).
        prompt_id:
            Optional unique identifier; auto-generated if not provided.
        reference_facts:
            Optional list of known-true statements for hallucination grounding.
        supporting_evidence:
            Optional list of evidence strings for unsupported-claim checking.

        Returns
        -------
        EvaluationResult
        """
        if prompt_id is None:
            prompt_id = str(uuid.uuid4())

        scores = self.metrics.evaluate(response, reference_facts, supporting_evidence)
        action = self.guardrail_engine.apply(response)

        result = EvaluationResult(
            prompt_id=prompt_id,
            prompt_text=prompt,
            response_text=action.sanitized_response,
            category=category,
            hallucination_score=float(scores["hallucination_score"]),
            unsupported_claim_score=float(scores["unsupported_claim_score"]),
            policy_violation_score=float(scores["policy_violation_score"]),
            overall_risk_score=float(scores["overall_risk_score"]),
            guardrail_triggered=action.triggered,
            guardrail_reason=action.reason,
            metadata={"violated_rules": scores["violated_rules"]},
        )
        return result

    def evaluate_batch(
        self,
        prompts: Sequence[dict],
        reference_facts: Sequence[str] | None = None,
        supporting_evidence: Sequence[str] | None = None,
    ) -> list[EvaluationResult]:
        """
        Evaluate a batch of prompt/response pairs.

        Each element of *prompts* must be a dict with at least:
        - ``"prompt"`` (str)
        - ``"response"`` (str)

        Optional keys per item: ``"category"``, ``"prompt_id"``,
        ``"reference_facts"``, ``"supporting_evidence"``.

        Parameters
        ----------
        prompts:
            Iterable of dicts as described above.
        reference_facts:
            Batch-level reference facts (overridden per item if provided).
        supporting_evidence:
            Batch-level supporting evidence (overridden per item if provided).

        Returns
        -------
        list[EvaluationResult]
        """
        results = []
        for item in prompts:
            result = self.evaluate(
                prompt=item["prompt"],
                response=item["response"],
                category=item.get("category", "general"),
                prompt_id=item.get("prompt_id"),
                reference_facts=item.get("reference_facts", reference_facts),
                supporting_evidence=item.get(
                    "supporting_evidence", supporting_evidence
                ),
            )
            results.append(result)
        return results

    def summary_statistics(
        self, results: list[EvaluationResult]
    ) -> dict:
        """Compute aggregate statistics across a list of evaluation results."""
        if not results:
            return {}

        n = len(results)
        unsafe = [r for r in results if not r.is_safe(self.safe_threshold)]

        def _avg(attr: str) -> float:
            return round(sum(getattr(r, attr) for r in results) / n, 4)

        return {
            "total_evaluated": n,
            "unsafe_count": len(unsafe),
            "unsafe_rate": round(len(unsafe) / n, 4),
            "safe_rate": round(1 - len(unsafe) / n, 4),
            "avg_hallucination_score": _avg("hallucination_score"),
            "avg_unsupported_claim_score": _avg("unsupported_claim_score"),
            "avg_policy_violation_score": _avg("policy_violation_score"),
            "avg_overall_risk_score": _avg("overall_risk_score"),
            "guardrail_trigger_count": sum(1 for r in results if r.guardrail_triggered),
            "guardrail_trigger_rate": round(
                sum(1 for r in results if r.guardrail_triggered) / n, 4
            ),
        }
