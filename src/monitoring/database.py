"""SQLite-backed database for storing and querying safety evaluation results."""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Generator

from ..utils.helpers import EvaluationResult


_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS model_versions (
    model_id        TEXT NOT NULL,
    model_name      TEXT NOT NULL,
    provider        TEXT,
    version_tag     TEXT,
    registered_at   TEXT NOT NULL,
    notes           TEXT,
    PRIMARY KEY (model_id)
);

CREATE TABLE IF NOT EXISTS evaluation_runs (
    run_id          TEXT NOT NULL,
    run_timestamp   TEXT NOT NULL,
    model_name      TEXT,
    dataset_version TEXT,
    total_evaluated INTEGER,
    unsafe_count    INTEGER,
    unsafe_rate     REAL,
    avg_hallucination_score     REAL,
    avg_unsupported_claim_score REAL,
    avg_policy_violation_score  REAL,
    avg_overall_risk_score      REAL,
    guardrail_trigger_rate      REAL,
    PRIMARY KEY (run_id)
);

CREATE TABLE IF NOT EXISTS evaluation_results (
    id                      INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id                  TEXT NOT NULL,
    prompt_id               TEXT NOT NULL,
    prompt_text             TEXT,
    response_text           TEXT,
    category                TEXT,
    hallucination_score     REAL,
    unsupported_claim_score REAL,
    policy_violation_score  REAL,
    overall_risk_score      REAL,
    guardrail_triggered     INTEGER,
    guardrail_reason        TEXT,
    created_at              TEXT NOT NULL,
    FOREIGN KEY (run_id) REFERENCES evaluation_runs (run_id)
);

CREATE INDEX IF NOT EXISTS idx_er_run_id   ON evaluation_results (run_id);
CREATE INDEX IF NOT EXISTS idx_er_category ON evaluation_results (category);
CREATE INDEX IF NOT EXISTS idx_er_risk     ON evaluation_results (overall_risk_score);

CREATE TABLE IF NOT EXISTS validation_reports (
    id                      INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id                  TEXT NOT NULL,
    prompt_id               TEXT NOT NULL,
    passed_all              INTEGER,
    passed_embedding        INTEGER,
    passed_rules            INTEGER,
    embedding_safe_sim      REAL,
    embedding_unsafe_sim    REAL,
    rule_highest_severity   TEXT,
    created_at              TEXT NOT NULL,
    FOREIGN KEY (run_id) REFERENCES evaluation_runs (run_id)
);
"""


class SafetyDatabase:
    """
    SQLite-backed store for safety evaluation runs, results, and validation
    reports.

    Parameters
    ----------
    db_path:
        Path to the SQLite database file.  Use ``":memory:"`` for an
        in-memory database (useful for testing).
    """

    def __init__(self, db_path: str | Path = ":memory:") -> None:
        self.db_path = str(db_path)
        # For in-memory databases keep a single persistent connection so
        # the schema and data survive across calls.
        if self.db_path == ":memory:":
            self._shared_conn: sqlite3.Connection | None = sqlite3.connect(":memory:")
            self._shared_conn.row_factory = sqlite3.Row
        else:
            self._shared_conn = None
        self._init_schema()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @contextmanager
    def _connect(self) -> Generator[sqlite3.Connection, None, None]:
        if self._shared_conn is not None:
            yield self._shared_conn
            self._shared_conn.commit()
            return
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _init_schema(self) -> None:
        with self._connect() as conn:
            conn.executescript(_SCHEMA_SQL)

    @staticmethod
    def _now() -> str:
        return datetime.now(timezone.utc).isoformat()

    # ------------------------------------------------------------------
    # Write operations
    # ------------------------------------------------------------------

    def register_model_version(
        self,
        model_id: str,
        model_name: str,
        provider: str = "",
        version_tag: str = "",
        notes: str = "",
    ) -> None:
        """Register a model version in the ``model_versions`` table."""
        sql = """
            INSERT OR REPLACE INTO model_versions
                (model_id, model_name, provider, version_tag, registered_at, notes)
            VALUES (?, ?, ?, ?, ?, ?)
        """
        with self._connect() as conn:
            conn.execute(
                sql,
                (model_id, model_name, provider, version_tag, self._now(), notes),
            )

    def insert_run(
        self,
        run_id: str,
        summary: dict,
        model_name: str = "",
        dataset_version: str = "",
    ) -> None:
        """Persist a run-level summary record."""
        sql = """
            INSERT OR REPLACE INTO evaluation_runs
                (run_id, run_timestamp, model_name, dataset_version,
                 total_evaluated, unsafe_count, unsafe_rate,
                 avg_hallucination_score, avg_unsupported_claim_score,
                 avg_policy_violation_score, avg_overall_risk_score,
                 guardrail_trigger_rate)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        with self._connect() as conn:
            conn.execute(
                sql,
                (
                    run_id,
                    self._now(),
                    model_name,
                    dataset_version,
                    summary.get("total_evaluated", 0),
                    summary.get("unsafe_count", 0),
                    summary.get("unsafe_rate", 0.0),
                    summary.get("avg_hallucination_score", 0.0),
                    summary.get("avg_unsupported_claim_score", 0.0),
                    summary.get("avg_policy_violation_score", 0.0),
                    summary.get("avg_overall_risk_score", 0.0),
                    summary.get("guardrail_trigger_rate", 0.0),
                ),
            )

    def insert_results(self, run_id: str, results: list[EvaluationResult]) -> None:
        """Persist a batch of :class:`EvaluationResult` objects."""
        sql = """
            INSERT INTO evaluation_results
                (run_id, prompt_id, prompt_text, response_text, category,
                 hallucination_score, unsupported_claim_score,
                 policy_violation_score, overall_risk_score,
                 guardrail_triggered, guardrail_reason, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        now = self._now()
        rows = [
            (
                run_id,
                r.prompt_id,
                r.prompt_text,
                r.response_text,
                r.category,
                r.hallucination_score,
                r.unsupported_claim_score,
                r.policy_violation_score,
                r.overall_risk_score,
                int(r.guardrail_triggered),
                r.guardrail_reason,
                now,
            )
            for r in results
        ]
        with self._connect() as conn:
            conn.executemany(sql, rows)

    def insert_validation_reports(
        self,
        run_id: str,
        reports: list,
    ) -> None:
        """Persist validation reports (from :class:`ValidationPipeline`)."""
        sql = """
            INSERT INTO validation_reports
                (run_id, prompt_id, passed_all, passed_embedding, passed_rules,
                 embedding_safe_sim, embedding_unsafe_sim, rule_highest_severity,
                 created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        now = self._now()
        rows = [
            (
                run_id,
                r.prompt_id,
                int(r.passed_all),
                int(r.passed_embedding),
                int(r.passed_rules),
                r.embedding_details.get("safe_similarity"),
                r.embedding_details.get("unsafe_similarity"),
                r.rule_details.get("highest_severity"),
                now,
            )
            for r in reports
        ]
        with self._connect() as conn:
            conn.executemany(sql, rows)

    # ------------------------------------------------------------------
    # Read / query operations
    # ------------------------------------------------------------------

    def get_run_history(self, limit: int = 50) -> list[dict]:
        """Return the most recent *limit* run summaries ordered by timestamp."""
        sql = """
            SELECT * FROM evaluation_runs
            ORDER BY run_timestamp DESC
            LIMIT ?
        """
        with self._connect() as conn:
            rows = conn.execute(sql, (limit,)).fetchall()
        return [dict(r) for r in rows]

    def get_results_for_run(self, run_id: str) -> list[dict]:
        """Return all evaluation results for a given *run_id*."""
        sql = "SELECT * FROM evaluation_results WHERE run_id = ? ORDER BY id"
        with self._connect() as conn:
            rows = conn.execute(sql, (run_id,)).fetchall()
        return [dict(r) for r in rows]

    def get_category_breakdown(self, run_id: str) -> list[dict]:
        """Return average scores grouped by prompt category for a run."""
        sql = """
            SELECT
                category,
                COUNT(*)                            AS count,
                AVG(hallucination_score)            AS avg_hallucination,
                AVG(unsupported_claim_score)        AS avg_unsupported_claim,
                AVG(policy_violation_score)         AS avg_policy_violation,
                AVG(overall_risk_score)             AS avg_overall_risk,
                SUM(guardrail_triggered)            AS guardrail_triggers
            FROM evaluation_results
            WHERE run_id = ?
            GROUP BY category
            ORDER BY avg_overall_risk DESC
        """
        with self._connect() as conn:
            rows = conn.execute(sql, (run_id,)).fetchall()
        return [dict(r) for r in rows]

    def trend_query(
        self,
        metric: str = "avg_overall_risk_score",
        limit: int = 20,
    ) -> list[dict]:
        """
        Return time-ordered values of *metric* from evaluation_runs for
        trend / drift analysis.

        Parameters
        ----------
        metric:
            Column name from ``evaluation_runs`` table.
        limit:
            Maximum number of rows to return (most recent first).
        """
        allowed = {
            "avg_overall_risk_score",
            "avg_hallucination_score",
            "avg_unsupported_claim_score",
            "avg_policy_violation_score",
            "unsafe_rate",
            "guardrail_trigger_rate",
        }
        if metric not in allowed:
            raise ValueError(f"metric must be one of {allowed}, got {metric!r}")
        sql = f"""
            SELECT run_id, run_timestamp, {metric} AS value
            FROM evaluation_runs
            ORDER BY run_timestamp DESC
            LIMIT ?
        """  # nosec – metric validated against allowlist above
        with self._connect() as conn:
            rows = conn.execute(sql, (limit,)).fetchall()
        return [dict(r) for r in rows]

    def top_unsafe_prompts(self, run_id: str, n: int = 20) -> list[dict]:
        """Return the *n* highest-risk prompts from a given run."""
        sql = """
            SELECT prompt_id, prompt_text, category, overall_risk_score,
                   guardrail_triggered, guardrail_reason
            FROM evaluation_results
            WHERE run_id = ?
            ORDER BY overall_risk_score DESC
            LIMIT ?
        """
        with self._connect() as conn:
            rows = conn.execute(sql, (run_id, n)).fetchall()
        return [dict(r) for r in rows]

    def get_model_versions(self) -> list[dict]:
        """Return all registered model versions ordered by registration time."""
        sql = "SELECT * FROM model_versions ORDER BY registered_at DESC"
        with self._connect() as conn:
            rows = conn.execute(sql).fetchall()
        return [dict(r) for r in rows]
