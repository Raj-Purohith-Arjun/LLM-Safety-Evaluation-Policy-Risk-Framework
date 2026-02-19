"""Safety drift analyser: quantifies how safety metrics change over time."""

from __future__ import annotations

import statistics
from typing import Sequence

from .database import SafetyDatabase


class SafetyDriftAnalyzer:
    """
    Analyses safety metric trends stored in the database to detect safety drift.

    Safety drift occurs when the model's risk scores increase over successive
    evaluation runs, indicating degradation in safety performance.

    Parameters
    ----------
    database:
        :class:`SafetyDatabase` instance to query.
    drift_threshold:
        Relative increase (fraction) in a metric between the oldest and
        newest observed value that is considered a meaningful drift.
        Default ``0.10`` (10 %).
    """

    def __init__(
        self,
        database: SafetyDatabase,
        drift_threshold: float = 0.10,
    ) -> None:
        self.database = database
        self.drift_threshold = drift_threshold

    _METRICS = [
        "avg_overall_risk_score",
        "avg_hallucination_score",
        "avg_unsupported_claim_score",
        "avg_policy_violation_score",
        "unsafe_rate",
        "guardrail_trigger_rate",
    ]

    def analyse(self, limit: int = 20) -> dict:
        """
        Analyse drift across all tracked metrics.

        Returns a dict mapping each metric name to a drift sub-report
        containing:
        - ``values`` – ordered list of (run_id, timestamp, value) tuples
        - ``trend``  – ``"increasing"``, ``"decreasing"``, or ``"stable"``
        - ``drift_detected`` – bool
        - ``relative_change`` – float (newest − oldest) / oldest
        - ``mean``, ``stdev``
        """
        report: dict[str, dict] = {}
        for metric in self._METRICS:
            rows = self.database.trend_query(metric=metric, limit=limit)
            # rows come newest-first; reverse for chronological order
            rows = list(reversed(rows))
            values = [r["value"] for r in rows if r["value"] is not None]

            if len(values) < 2:
                report[metric] = {
                    "values": rows,
                    "trend": "insufficient_data",
                    "drift_detected": False,
                    "relative_change": None,
                    "mean": values[0] if values else None,
                    "stdev": None,
                }
                continue

            oldest, newest = values[0], values[-1]
            relative_change = (newest - oldest) / oldest if oldest != 0 else 0.0
            trend = (
                "increasing"
                if relative_change > 0.02
                else "decreasing"
                if relative_change < -0.02
                else "stable"
            )
            drift_detected = abs(relative_change) >= self.drift_threshold

            report[metric] = {
                "values": rows,
                "trend": trend,
                "drift_detected": drift_detected,
                "relative_change": round(relative_change, 4),
                "mean": round(statistics.mean(values), 4),
                "stdev": round(statistics.stdev(values), 4) if len(values) > 1 else 0.0,
            }

        # Overall drift flag
        any_drift = any(
            v.get("drift_detected", False)
            for v in report.values()
            if isinstance(v, dict)
        )

        return {
            "overall_drift_detected": any_drift,
            "drift_threshold": self.drift_threshold,
            "metrics": report,
        }

    def compare_runs(self, run_id_a: str, run_id_b: str) -> dict:
        """
        Compare two specific evaluation runs and highlight metric deltas.

        Parameters
        ----------
        run_id_a, run_id_b:
            Identifiers of the two runs to compare (A = baseline).
        """
        history = self.database.get_run_history(limit=200)
        runs = {r["run_id"]: r for r in history}

        if run_id_a not in runs:
            raise ValueError(f"run_id '{run_id_a}' not found in database.")
        if run_id_b not in runs:
            raise ValueError(f"run_id '{run_id_b}' not found in database.")

        run_a, run_b = runs[run_id_a], runs[run_id_b]
        deltas: dict[str, dict] = {}
        for metric in self._METRICS:
            val_a = run_a.get(metric)
            val_b = run_b.get(metric)
            if val_a is not None and val_b is not None and val_a != 0:
                delta = val_b - val_a
                relative = delta / val_a
            else:
                delta = relative = None
            deltas[metric] = {
                "run_a": val_a,
                "run_b": val_b,
                "absolute_delta": round(delta, 4) if delta is not None else None,
                "relative_delta": round(relative, 4) if relative is not None else None,
                "improved": delta < 0 if delta is not None else None,
            }

        return {
            "run_a": run_id_a,
            "run_b": run_id_b,
            "metric_deltas": deltas,
        }
