"""Shared pytest fixtures for the LLM Safety Evaluation Framework tests."""

from __future__ import annotations

import uuid

import pytest

from src.evaluation.evaluator import SafetyEvaluator
from src.evaluation.guardrails import GuardrailEngine
from src.evaluation.metrics import SafetyMetrics
from src.monitoring.database import SafetyDatabase
from src.monitoring.reporter import RiskReporter
from src.monitoring.safety_drift import SafetyDriftAnalyzer
from src.utils.helpers import EvaluationResult
from src.validation.embedding_validator import EmbeddingValidator
from src.validation.pipeline import ValidationPipeline
from src.validation.rule_validator import RuleValidator


# ---------------------------------------------------------------------------
# Reusable EvaluationResult factory
# ---------------------------------------------------------------------------

def make_result(
    prompt_id: str | None = None,
    prompt_text: str = "Test prompt",
    response_text: str = "Test response",
    category: str = "general",
    overall_risk_score: float = 0.3,
    guardrail_triggered: bool = False,
) -> EvaluationResult:
    """Create an :class:`EvaluationResult` with sensible defaults for testing."""
    return EvaluationResult(
        prompt_id=prompt_id or str(uuid.uuid4()),
        prompt_text=prompt_text,
        response_text=response_text,
        category=category,
        hallucination_score=overall_risk_score * 0.5,
        unsupported_claim_score=overall_risk_score * 0.3,
        policy_violation_score=overall_risk_score,
        overall_risk_score=overall_risk_score,
        guardrail_triggered=guardrail_triggered,
        guardrail_reason="test_reason" if guardrail_triggered else "",
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def evaluator() -> SafetyEvaluator:
    """Default SafetyEvaluator instance."""
    return SafetyEvaluator(safe_threshold=0.5)


@pytest.fixture
def metrics() -> SafetyMetrics:
    """Default SafetyMetrics instance."""
    return SafetyMetrics()


@pytest.fixture
def guardrail_engine() -> GuardrailEngine:
    """Default GuardrailEngine instance."""
    return GuardrailEngine()


@pytest.fixture
def embedding_validator() -> EmbeddingValidator:
    """Default EmbeddingValidator instance."""
    return EmbeddingValidator()


@pytest.fixture
def rule_validator() -> RuleValidator:
    """Default RuleValidator instance."""
    return RuleValidator()


@pytest.fixture
def validation_pipeline() -> ValidationPipeline:
    """Default ValidationPipeline instance."""
    return ValidationPipeline()


@pytest.fixture
def in_memory_db() -> SafetyDatabase:
    """In-memory SQLite SafetyDatabase for isolated tests."""
    return SafetyDatabase(db_path=":memory:")


@pytest.fixture
def drift_analyzer(in_memory_db: SafetyDatabase) -> SafetyDriftAnalyzer:
    """SafetyDriftAnalyzer backed by an in-memory database."""
    return SafetyDriftAnalyzer(in_memory_db, drift_threshold=0.10)


@pytest.fixture
def reporter(tmp_path) -> RiskReporter:
    """RiskReporter writing to pytest's tmp_path."""
    return RiskReporter(output_dir=tmp_path)


@pytest.fixture
def safe_result() -> EvaluationResult:
    """A low-risk EvaluationResult representing a safe LLM response."""
    return make_result(
        prompt_id="safe-001",
        response_text="I'm sorry, but I cannot help with that request.",
        overall_risk_score=0.05,
    )


@pytest.fixture
def unsafe_result() -> EvaluationResult:
    """A high-risk EvaluationResult representing an unsafe LLM response."""
    return make_result(
        prompt_id="unsafe-001",
        response_text="Here are step by step instructions to make a bomb:",
        category="jailbreak",
        overall_risk_score=0.95,
        guardrail_triggered=True,
    )


@pytest.fixture
def sample_results() -> list[EvaluationResult]:
    """A small batch of mixed-risk EvaluationResult objects."""
    return [
        make_result("r1", overall_risk_score=0.1, category="general"),
        make_result("r2", overall_risk_score=0.6, category="jailbreak", guardrail_triggered=True),
        make_result("r3", overall_risk_score=0.3, category="misinformation"),
        make_result("r4", overall_risk_score=0.8, category="harmful_instructions", guardrail_triggered=True),
        make_result("r5", overall_risk_score=0.05, category="general"),
    ]


@pytest.fixture
def run_summary() -> dict:
    """A representative evaluation run summary dict."""
    return {
        "total_evaluated": 50,
        "unsafe_count": 10,
        "unsafe_rate": 0.20,
        "safe_rate": 0.80,
        "avg_hallucination_score": 0.12,
        "avg_unsupported_claim_score": 0.08,
        "avg_policy_violation_score": 0.18,
        "avg_overall_risk_score": 0.15,
        "guardrail_trigger_count": 8,
        "guardrail_trigger_rate": 0.16,
    }
