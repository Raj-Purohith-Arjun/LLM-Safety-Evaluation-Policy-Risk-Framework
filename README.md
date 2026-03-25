# LLM Safety Evaluation Policy Risk Framework

A structured, automated framework for evaluating the safety of large language model (LLM) outputs. It tests models against 530+ adversarial and edge-case prompts, scores multiple risk signals, and produces clear reports that help teams track safety performance over time.

---

## Overview

This project provides a repeatable way to measure whether an LLM response is safe, policy-compliant, and grounded. It combines scoring metrics, guardrails, validation checks, and monitoring tools into a single workflow so you can:

- Benchmark model safety across risky prompt categories
- Detect unsafe behavior early
- Track safety drift across model updates
- Produce shareable reports for stakeholders

---

## Problem

LLM outputs can include unsafe instructions, misinformation, or privacy violations. Manual review is slow and inconsistent, and small changes in a model can quietly shift safety performance. Teams need a reliable way to test responses, quantify risk, and monitor changes over time.

---

## Solution

This framework automates safety evaluation end-to-end:

1. **Adversarial prompts** challenge the model with realistic risky scenarios.
2. **Safety metrics** quantify hallucination risk, unsupported claims, and policy violations.
3. **Guardrails and validation** check for unsafe content and help refine responses.
4. **Monitoring and reports** capture results in SQLite and generate CSV/HTML summaries.
5. **Drift analysis** detects meaningful safety changes between runs.

---

## Key Features

- **Safety Metrics**: hallucination score, unsupported-claim score, policy-violation score, and overall risk
- **Adversarial Dataset**: 530+ labeled prompts spanning 12 safety categories
- **Guardrail Engine**: rule-based blocks, redaction, and refinement tracking
- **Validation Pipeline**: embedding similarity checks + deterministic rule validation
- **Safety Monitoring**: SQLite-backed history and drift analysis across runs
- **Reporting**: CSV and HTML risk reports with top high-risk prompts
- **Model Comparison**: compare multiple models on the same dataset

---

## Architecture

```
Adversarial Prompts
        │
        ▼
SafetyEvaluator ──► Safety Metrics ──► Guardrail Engine
        │                          │
        ▼                          ▼
Validation Pipeline          Safety Results
        │                          │
        ▼                          ▼
SQLite Database ──► Drift Analysis ──► Reports (CSV/HTML)
```

**Core modules**
- **evaluation/**: scoring logic and orchestration
- **prompts/**: adversarial prompt dataset and taxonomy
- **validation/**: embedding + rule-based checks
- **monitoring/**: database, drift analysis, and reporting
- **llm/**: LLM API abstractions
- **dashboard.py**: optional Streamlit UI for visual analysis

---

## Tech Stack

- **Language**: Python 3.10+
- **Core Libraries**: NumPy, pandas, scikit-learn, Jinja2
- **Storage**: SQLite
- **Testing**: pytest, pytest-cov
- **Optional UI**: Streamlit

---

## Repository Structure

```
src/
├── evaluation/     # SafetyEvaluator, metrics, guardrails
├── prompts/        # Adversarial prompt dataset and categories
├── validation/     # Embedding and rule-based validation pipeline
├── monitoring/     # SQLite database, drift analysis, reporting
├── llm/            # LLM API client abstractions
└── utils/          # Helpers and logging

examples/
└── run_evaluation.py   # End-to-end sample run

tests/
└── ...                 # pytest test suite
```

---

## Setup

### Install dependencies

```bash
pip install -e .[dev]
# Or install from requirements.txt
pip install -r requirements.txt
```

### Run the evaluation pipeline

```bash
python examples/run_evaluation.py --sample 100 --output-dir /tmp/safety_reports
```

This run will:
1. Sample prompts from the adversarial dataset
2. Score responses with safety metrics
3. Apply validation checks and guardrails
4. Persist results to SQLite
5. Generate CSV and HTML risk reports

### Launch the dashboard (optional)

```bash
python dashboard.py
```

### Run tests

```bash
python -m pytest tests/ -v
```

---

## Results and Outputs

After a run, you can expect:

- **CSV and HTML reports** with risk summaries and top unsafe prompts
- **SQLite history** for trend analysis and model comparisons
- **Drift indicators** that highlight shifts in safety behavior

These outputs make it easy to compare models, track regression risk, and communicate safety results clearly.

---

## Why It Matters

Safety evaluation should be repeatable, measurable, and transparent. This framework turns subjective safety review into a consistent process, making it easier to monitor risk, improve guardrails, and build trustworthy LLM applications.
