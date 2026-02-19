"""Tests for the monitoring (database, drift, reporter) modules."""

import pytest
import uuid
from pathlib import Path

from src.monitoring.database import SafetyDatabase
from src.monitoring.safety_drift import SafetyDriftAnalyzer
from src.monitoring.reporter import RiskReporter
from src.utils.helpers import EvaluationResult


def _make_result(prompt_id: str, risk: float = 0.3, triggered: bool = False) -> EvaluationResult:
    return EvaluationResult(
        prompt_id=prompt_id,
        prompt_text=f"Prompt for {prompt_id}",
        response_text="Some response text.",
        category="general",
        hallucination_score=risk * 0.5,
        unsupported_claim_score=risk * 0.3,
        policy_violation_score=risk,
        overall_risk_score=risk,
        guardrail_triggered=triggered,
        guardrail_reason="test" if triggered else "",
    )


class TestSafetyDatabase:
    def setup_method(self):
        self.db = SafetyDatabase(db_path=":memory:")

    def test_insert_and_retrieve_run(self):
        summary = {
            "total_evaluated": 10,
            "unsafe_count": 3,
            "unsafe_rate": 0.3,
            "avg_hallucination_score": 0.2,
            "avg_unsupported_claim_score": 0.15,
            "avg_policy_violation_score": 0.25,
            "avg_overall_risk_score": 0.22,
            "guardrail_trigger_rate": 0.1,
        }
        run_id = str(uuid.uuid4())
        self.db.insert_run(run_id, summary)
        history = self.db.get_run_history()
        assert len(history) == 1
        assert history[0]["run_id"] == run_id
        assert history[0]["total_evaluated"] == 10

    def test_insert_and_retrieve_results(self):
        run_id = str(uuid.uuid4())
        self.db.insert_run(run_id, {"total_evaluated": 2})
        results = [_make_result("p1", 0.4), _make_result("p2", 0.8, triggered=True)]
        self.db.insert_results(run_id, results)
        rows = self.db.get_results_for_run(run_id)
        assert len(rows) == 2

    def test_category_breakdown(self):
        run_id = str(uuid.uuid4())
        self.db.insert_run(run_id, {"total_evaluated": 3})
        results = [
            _make_result("p1", 0.3),
            _make_result("p2", 0.7),
            _make_result("p3", 0.5),
        ]
        self.db.insert_results(run_id, results)
        breakdown = self.db.get_category_breakdown(run_id)
        assert isinstance(breakdown, list)
        assert len(breakdown) >= 1

    def test_trend_query(self):
        for i in range(3):
            run_id = str(uuid.uuid4())
            self.db.insert_run(run_id, {"avg_overall_risk_score": 0.1 * (i + 1)})
        trend = self.db.trend_query(metric="avg_overall_risk_score", limit=5)
        assert len(trend) <= 3
        for row in trend:
            assert "value" in row

    def test_trend_query_invalid_metric(self):
        with pytest.raises(ValueError):
            self.db.trend_query(metric="invalid_column")

    def test_top_unsafe_prompts(self):
        run_id = str(uuid.uuid4())
        self.db.insert_run(run_id, {"total_evaluated": 5})
        results = [_make_result(f"p{i}", float(i) / 10) for i in range(5)]
        self.db.insert_results(run_id, results)
        top = self.db.top_unsafe_prompts(run_id, n=3)
        assert len(top) == 3
        assert top[0]["overall_risk_score"] >= top[-1]["overall_risk_score"]


class TestSafetyDriftAnalyzer:
    def setup_method(self):
        self.db = SafetyDatabase(db_path=":memory:")
        self.analyzer = SafetyDriftAnalyzer(self.db, drift_threshold=0.05)

    def test_analyse_insufficient_data(self):
        report = self.analyzer.analyse()
        for metric_data in report["metrics"].values():
            assert metric_data["trend"] in ("insufficient_data", "increasing", "decreasing", "stable")

    def test_analyse_with_two_runs(self):
        for i, risk in enumerate([0.2, 0.4]):
            run_id = f"run-{i}"
            self.db.insert_run(run_id, {"avg_overall_risk_score": risk, "unsafe_rate": risk})
        report = self.analyzer.analyse()
        assert "overall_drift_detected" in report
        assert "metrics" in report

    def test_compare_runs(self):
        self.db.insert_run("run-a", {
            "avg_overall_risk_score": 0.2,
            "avg_hallucination_score": 0.1,
            "avg_unsupported_claim_score": 0.1,
            "avg_policy_violation_score": 0.2,
            "unsafe_rate": 0.15,
            "guardrail_trigger_rate": 0.1,
        })
        self.db.insert_run("run-b", {
            "avg_overall_risk_score": 0.3,
            "avg_hallucination_score": 0.15,
            "avg_unsupported_claim_score": 0.12,
            "avg_policy_violation_score": 0.3,
            "unsafe_rate": 0.2,
            "guardrail_trigger_rate": 0.15,
        })
        comparison = self.analyzer.compare_runs("run-a", "run-b")
        assert comparison["run_a"] == "run-a"
        assert comparison["run_b"] == "run-b"
        assert "metric_deltas" in comparison

    def test_compare_runs_missing_id_raises(self):
        self.db.insert_run("run-exists", {"avg_overall_risk_score": 0.1})
        with pytest.raises(ValueError):
            self.analyzer.compare_runs("run-exists", "run-missing")


class TestRiskReporter:
    def setup_method(self):
        self.reporter = RiskReporter(output_dir="/tmp/test_reports")
        self.results = [
            _make_result("p1", 0.2),
            _make_result("p2", 0.7, triggered=True),
            _make_result("p3", 0.4),
        ]
        self.summary = {
            "total_evaluated": 3,
            "unsafe_count": 1,
            "unsafe_rate": 0.33,
            "avg_hallucination_score": 0.2,
            "avg_unsupported_claim_score": 0.15,
            "avg_policy_violation_score": 0.3,
            "avg_overall_risk_score": 0.43,
            "guardrail_trigger_rate": 0.33,
        }

    def test_to_csv_string(self):
        csv_str = self.reporter.to_csv_string(self.results)
        assert "prompt_id" in csv_str
        assert "overall_risk_score" in csv_str
        assert "p1" in csv_str

    def test_to_csv_writes_file(self):
        path = self.reporter.to_csv(
            self.results, filename="test_report.csv", run_id="test-run"
        )
        assert path.exists()
        content = path.read_text()
        assert "p1" in content

    def test_to_html_writes_file(self):
        path = self.reporter.to_html(
            self.results,
            self.summary,
            run_id="test-run",
            model_name="TestModel",
            filename="test_report.html",
        )
        assert path.exists()
        html = path.read_text()
        assert "LLM Safety Evaluation Report" in html
        assert "test-run" in html
