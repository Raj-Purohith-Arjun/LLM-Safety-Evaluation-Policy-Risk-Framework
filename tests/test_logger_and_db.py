"""Tests for the structured logging utility and database model_versions."""

from __future__ import annotations

import json
import logging
import io
import uuid

import pytest

from src.utils.logger import (
    configure_logging,
    get_logger,
    JsonFormatter,
    EvalRunLogger,
)
from src.monitoring.database import SafetyDatabase


class TestJsonFormatter:
    def test_formats_as_json(self):
        formatter = JsonFormatter()
        record = logging.LogRecord(
            name="test", level=logging.INFO, pathname="", lineno=0,
            msg="hello world", args=(), exc_info=None,
        )
        output = formatter.format(record)
        parsed = json.loads(output)
        assert parsed["message"] == "hello world"
        assert parsed["level"] == "INFO"
        assert "timestamp" in parsed
        assert parsed["logger"] == "test"

    def test_extra_fields_included(self):
        formatter = JsonFormatter()
        record = logging.LogRecord(
            name="test", level=logging.INFO, pathname="", lineno=0,
            msg="msg", args=(), exc_info=None,
        )
        record.run_id = "abc123"
        output = formatter.format(record)
        parsed = json.loads(output)
        assert parsed.get("run_id") == "abc123"


class TestConfigureLogging:
    def test_human_format(self, capsys):
        stream = io.StringIO()
        configure_logging(level="DEBUG", json_format=False, stream=stream)
        log = get_logger("test_human")
        log.debug("debug message")
        value = stream.getvalue()
        assert "debug message" in value

    def test_json_format(self):
        stream = io.StringIO()
        configure_logging(level="DEBUG", json_format=True, stream=stream)
        log = get_logger("test_json")
        log.info("json log test")
        value = stream.getvalue()
        # Should be valid JSON
        for line in value.strip().split("\n"):
            if line:
                parsed = json.loads(line)
                assert "message" in parsed

    def test_idempotent_reconfigure(self):
        """Calling configure_logging twice should not duplicate handlers."""
        stream = io.StringIO()
        configure_logging(level="INFO", stream=stream)
        configure_logging(level="INFO", stream=stream)
        root = logging.getLogger()
        # Each configure_logging removes existing handlers; expect exactly 1
        assert len(root.handlers) == 1


class TestGetLogger:
    def test_returns_logger(self):
        log = get_logger("src.test")
        assert isinstance(log, logging.Logger)
        assert log.name == "src.test"


class TestEvalRunLogger:
    def test_attaches_run_id(self):
        stream = io.StringIO()
        configure_logging(level="DEBUG", json_format=True, stream=stream)
        base = get_logger("eval_run")
        run_logger = EvalRunLogger(run_id="run-xyz", base_logger=base)
        run_logger.info("started")
        value = stream.getvalue()
        for line in value.strip().split("\n"):
            if line:
                parsed = json.loads(line)
                if parsed.get("message") == "started":
                    assert parsed.get("run_id") == "run-xyz"
                    break

    def test_all_levels_work(self):
        base = get_logger("dummy_run")
        run_logger = EvalRunLogger(run_id="r1", base_logger=base)
        # These should not raise
        run_logger.debug("dbg")
        run_logger.info("info")
        run_logger.warning("warn")
        run_logger.error("err")


class TestDatabaseModelVersions:
    def setup_method(self):
        self.db = SafetyDatabase(db_path=":memory:")

    def test_register_model_version(self):
        self.db.register_model_version(
            model_id="gpt-4o-v1",
            model_name="gpt-4o",
            provider="OpenAI",
            version_tag="2024-11",
            notes="Baseline model",
        )
        versions = self.db.get_model_versions()
        assert len(versions) == 1
        assert versions[0]["model_id"] == "gpt-4o-v1"
        assert versions[0]["model_name"] == "gpt-4o"
        assert versions[0]["provider"] == "OpenAI"

    def test_register_multiple_versions(self):
        for i in range(3):
            self.db.register_model_version(
                model_id=f"model-{i}",
                model_name=f"model-{i}",
            )
        versions = self.db.get_model_versions()
        assert len(versions) == 3

    def test_register_replaces_existing(self):
        self.db.register_model_version(
            model_id="m1", model_name="old-name"
        )
        self.db.register_model_version(
            model_id="m1", model_name="new-name"
        )
        versions = self.db.get_model_versions()
        names = [v["model_name"] for v in versions]
        assert "new-name" in names
        # Should only have one entry for m1
        m1_versions = [v for v in versions if v["model_id"] == "m1"]
        assert len(m1_versions) == 1

    def test_get_model_versions_empty(self):
        versions = self.db.get_model_versions()
        assert versions == []

    def test_registered_at_field_populated(self):
        self.db.register_model_version(model_id="mx", model_name="mx")
        versions = self.db.get_model_versions()
        assert versions[0]["registered_at"] is not None
        assert len(versions[0]["registered_at"]) > 0
