"""
Streamlit interactive monitoring dashboard for the LLM Safety Evaluation Framework.

Run with:
    streamlit run dashboard.py

The dashboard uses the SQLite database created by the evaluation pipeline.
Pass a custom database path via the SAFETY_DB environment variable, or
provide it in the sidebar.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

# Make src importable when running from the repo root
sys.path.insert(0, str(Path(__file__).parent))

import streamlit as st
import pandas as pd

from src.monitoring.database import SafetyDatabase
from src.monitoring.safety_drift import SafetyDriftAnalyzer

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="LLM Safety Monitoring Dashboard",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Sidebar: database selection
# ---------------------------------------------------------------------------

st.sidebar.title("🛡️ LLM Safety Dashboard")
st.sidebar.markdown("---")

default_db = os.environ.get("SAFETY_DB", "safety_evaluations.db")
db_path = st.sidebar.text_input("Database path", value=default_db)

try:
    db = SafetyDatabase(db_path=db_path)
    st.sidebar.success(f"Connected to: `{db_path}`")
except Exception as e:
    st.sidebar.error(f"Could not connect: {e}")
    st.stop()

# ---------------------------------------------------------------------------
# Load run history
# ---------------------------------------------------------------------------

history = db.get_run_history(limit=100)

if not history:
    st.warning(
        "No evaluation runs found in the database. "
        "Run `python examples/run_evaluation.py` to populate it."
    )
    st.stop()

history_df = pd.DataFrame(history)
history_df["run_timestamp"] = pd.to_datetime(history_df["run_timestamp"])
history_df = history_df.sort_values("run_timestamp")

# Run selector
run_ids = history_df["run_id"].tolist()
_run_display: dict[str, str] = {
    row["run_id"]: (
        f"{row['run_id']} "
        f"({pd.Timestamp(row['run_timestamp']).strftime('%Y-%m-%d %H:%M')})"
    )
    for _, row in history_df.iterrows()
}
selected_run = st.sidebar.selectbox(
    "Evaluation run",
    options=run_ids,
    index=len(run_ids) - 1,
    format_func=lambda r: _run_display.get(r, r),
)

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------

st.title("🛡️ LLM Safety Evaluation Monitoring Dashboard")
st.markdown(
    f"**Selected run:** `{selected_run}` &nbsp;|&nbsp; "
    f"**Total runs in DB:** {len(history_df)}"
)
st.markdown("---")

# ---------------------------------------------------------------------------
# Load selected run data
# ---------------------------------------------------------------------------

selected_row = history_df[history_df["run_id"] == selected_run].iloc[0]
results = db.get_results_for_run(selected_run)
results_df = pd.DataFrame(results) if results else pd.DataFrame()
category_breakdown = db.get_category_breakdown(selected_run)
top_unsafe = db.top_unsafe_prompts(selected_run, n=20)

# ---------------------------------------------------------------------------
# Tab layout
# ---------------------------------------------------------------------------

tab_overview, tab_metrics, tab_categories, tab_drift, tab_prompts, tab_models = st.tabs([
    "📊 Overview",
    "📈 Metric Trends",
    "🗂️ Category Breakdown",
    "⚠️ Drift Analysis",
    "🔍 Top Risky Prompts",
    "🤖 Model Versions",
])

# ============================================================
# TAB: Overview
# ============================================================
with tab_overview:
    st.subheader("Executive Summary")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Prompts Evaluated", int(selected_row.get("total_evaluated", 0)))
    col2.metric(
        "Unsafe Rate",
        f"{selected_row.get('unsafe_rate', 0) * 100:.1f}%",
        delta=None,
    )
    col3.metric(
        "Avg Risk Score",
        f"{selected_row.get('avg_overall_risk_score', 0):.4f}",
    )
    col4.metric(
        "Guardrail Trigger Rate",
        f"{selected_row.get('guardrail_trigger_rate', 0) * 100:.1f}%",
    )

    st.markdown("---")

    col_left, col_right = st.columns(2)
    with col_left:
        st.subheader("Score Breakdown")
        score_data = {
            "Metric": [
                "Hallucination Score",
                "Unsupported Claim Score",
                "Policy Violation Score",
                "Overall Risk Score",
            ],
            "Average Score": [
                selected_row.get("avg_hallucination_score", 0),
                selected_row.get("avg_unsupported_claim_score", 0),
                selected_row.get("avg_policy_violation_score", 0),
                selected_row.get("avg_overall_risk_score", 0),
            ],
        }
        score_df = pd.DataFrame(score_data)
        st.bar_chart(score_df.set_index("Metric"))

    with col_right:
        st.subheader("Run Metadata")
        st.json({
            "run_id": selected_row.get("run_id", ""),
            "model": selected_row.get("model_name", "unknown"),
            "dataset_version": selected_row.get("dataset_version", "unknown"),
            "timestamp": str(selected_row.get("run_timestamp", "")),
        })

    # Guardrail triggered distribution
    if not results_df.empty and "guardrail_triggered" in results_df.columns:
        st.subheader("Guardrail Trigger Distribution")
        triggered_counts = results_df["guardrail_triggered"].value_counts().rename(
            {0: "Not triggered", 1: "Triggered"}
        )
        st.bar_chart(triggered_counts)

# ============================================================
# TAB: Metric Trends
# ============================================================
with tab_metrics:
    st.subheader("Safety Metrics Over Time")

    metrics_available = [
        "avg_overall_risk_score",
        "avg_hallucination_score",
        "avg_unsupported_claim_score",
        "avg_policy_violation_score",
        "unsafe_rate",
        "guardrail_trigger_rate",
    ]
    selected_metrics = st.multiselect(
        "Select metrics to display",
        options=metrics_available,
        default=["avg_overall_risk_score", "unsafe_rate"],
    )

    if selected_metrics:
        trend_df = history_df[["run_timestamp"] + [
            m for m in selected_metrics if m in history_df.columns
        ]].set_index("run_timestamp")
        st.line_chart(trend_df)
        st.caption(
            "Each point represents one evaluation run, ordered by timestamp."
        )
    else:
        st.info("Select at least one metric above.")

# ============================================================
# TAB: Category Breakdown
# ============================================================
with tab_categories:
    st.subheader("Risk Score by Prompt Category")

    if category_breakdown:
        cat_df = pd.DataFrame(category_breakdown)
        # Rename for readability
        cat_df = cat_df.rename(columns={
            "avg_overall_risk": "Avg Risk Score",
            "avg_hallucination": "Avg Hallucination",
            "avg_unsupported_claim": "Avg Unsupported Claim",
            "avg_policy_violation": "Avg Policy Violation",
            "guardrail_triggers": "Guardrail Triggers",
        })
        st.dataframe(cat_df, use_container_width=True)

        if "Avg Risk Score" in cat_df.columns:
            st.bar_chart(cat_df.set_index("category")["Avg Risk Score"])
    else:
        st.info("No category breakdown available for this run.")

# ============================================================
# TAB: Drift Analysis
# ============================================================
with tab_drift:
    st.subheader("Safety Drift Detection")

    drift_threshold = st.slider(
        "Drift threshold (relative change %)",
        min_value=1,
        max_value=50,
        value=10,
        step=1,
    ) / 100.0

    analyzer = SafetyDriftAnalyzer(db, drift_threshold=drift_threshold)
    drift_report = analyzer.analyse()

    overall_drift = drift_report.get("overall_drift_detected", False)
    if overall_drift:
        st.error("⚠️ Safety drift detected across one or more metrics!")
    else:
        st.success("✅ No significant safety drift detected.")

    st.markdown("---")
    for metric, data in drift_report.get("metrics", {}).items():
        trend = data.get("trend", "unknown")
        drift_flag = data.get("drift_detected", False)
        relative_change = data.get("relative_change")

        icon = "🔴" if drift_flag else "🟢"
        trend_icon = "↑" if trend == "increasing" else "↓" if trend == "decreasing" else "→"

        with st.expander(f"{icon} {metric}  {trend_icon} {trend}"):
            col_a, col_b, col_c = st.columns(3)
            col_a.metric("Trend", trend)
            col_b.metric("Drift Detected", "Yes" if drift_flag else "No")
            if relative_change is not None:
                col_c.metric(
                    "Relative Change",
                    f"{relative_change * 100:+.1f}%",
                    delta=relative_change * 100,
                    delta_color="inverse",
                )
            else:
                col_c.metric("Relative Change", "N/A")

            values = data.get("values", [])
            if values:
                vals_df = pd.DataFrame(values)
                if "value" in vals_df.columns and "run_timestamp" in vals_df.columns:
                    st.line_chart(
                        vals_df[["run_timestamp", "value"]].set_index("run_timestamp")
                    )

# ============================================================
# TAB: Top Risky Prompts
# ============================================================
with tab_prompts:
    st.subheader("Top High-Risk Prompts")

    n_prompts = st.slider("Number of prompts to show", 5, 50, 20)
    top_unsafe = db.top_unsafe_prompts(selected_run, n=n_prompts)

    if top_unsafe:
        prompts_df = pd.DataFrame(top_unsafe)
        prompts_df["prompt_text"] = (
            prompts_df["prompt_text"].fillna("").str[:120] + "…"
        )
        st.dataframe(prompts_df, use_container_width=True)
    else:
        st.info("No prompt data available for this run.")

# ============================================================
# TAB: Model Versions
# ============================================================
with tab_models:
    st.subheader("Registered Model Versions")

    model_versions = db.get_model_versions()
    if model_versions:
        st.dataframe(pd.DataFrame(model_versions), use_container_width=True)
    else:
        st.info(
            "No model versions registered. Use `db.register_model_version(...)` "
            "to track model metadata."
        )

    st.markdown("---")
    st.subheader("Safety Score Comparison Across Runs")

    if len(history_df) >= 2:
        compare_cols = [
            "model_name",
            "avg_overall_risk_score",
            "avg_hallucination_score",
            "avg_policy_violation_score",
            "unsafe_rate",
        ]
        available_cols = [c for c in compare_cols if c in history_df.columns]
        comparison_df = history_df[["run_id", "run_timestamp"] + available_cols]
        comparison_df = comparison_df.sort_values("run_timestamp", ascending=False)
        st.dataframe(comparison_df, use_container_width=True)

        # Safety vs helpfulness tradeoff scatter (requires refusal data from results)
        if not results_df.empty and "guardrail_triggered" in results_df.columns:
            model_names = history_df["model_name"].dropna().unique().tolist()
            if len(model_names) > 1:
                st.subheader("Safety vs. Helpfulness Tradeoff")
                tradeoff_data = []
                for _, row in history_df.iterrows():
                    run_results = db.get_results_for_run(row["run_id"])
                    if not run_results:
                        continue
                    refusal_count = sum(
                        1 for r in run_results
                        if any(
                            p in (r.get("response_text") or "").lower()
                            for p in ["i cannot", "i'm unable", "i'm sorry", "i will not"]
                        )
                    )
                    n = len(run_results)
                    tradeoff_data.append({
                        "model": row.get("model_name", row["run_id"]),
                        "safety": 1.0 - (row.get("avg_overall_risk_score") or 0),
                        "helpfulness": 1.0 - (refusal_count / n if n else 0),
                    })
                if tradeoff_data:
                    td_df = pd.DataFrame(tradeoff_data)
                    st.scatter_chart(
                        td_df.set_index("model"),
                        x="helpfulness",
                        y="safety",
                    )
    else:
        st.info("Run multiple evaluations to see model comparisons.")

# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------

st.markdown("---")
st.caption(
    "LLM Safety Evaluation Policy Risk Framework &bull; "
    "Use `python examples/run_evaluation.py` to generate data."
)
