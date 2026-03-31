"""Unit tests for src/observability/mlflow_tracker.py"""

from unittest.mock import MagicMock, patch

import src.observability.mlflow_tracker as tracker_module
from src.observability.mlflow_tracker import MLflowTracker, get_mlflow_tracker

# ─── Helpers ──────────────────────────────────────────────────────────────────


def _disabled_tracker() -> MLflowTracker:
    """Return a tracker with enabled=False (no mlflow calls)."""
    t = MLflowTracker.__new__(MLflowTracker)
    t.enabled = False
    return t


# ─── MLflowTracker ────────────────────────────────────────────────────────────


class TestMLflowTrackerDisabled:
    def test_log_ingestion_run_noop(self):
        t = _disabled_tracker()
        # Should return immediately without error
        t.log_ingestion_run({"run_id": "abc", "documents_processed": 0, "chunks_ingested": 0, "chunks_skipped": 0})

    def test_log_query_result_noop(self):
        t = _disabled_tracker()
        t.log_query_result("some query", {"answer": "yes"})


class TestMLflowTrackerEnabled:
    def test_log_ingestion_run_calls_mlflow(self):
        t = _disabled_tracker()
        t.enabled = True

        mock_mlflow = MagicMock()
        # Simulate context manager for start_run
        mock_run_ctx = MagicMock()
        mock_mlflow.start_run.return_value.__enter__ = MagicMock(return_value=mock_run_ctx)
        mock_mlflow.start_run.return_value.__exit__ = MagicMock(return_value=False)

        summary = {
            "run_id": "deadbeef1234",
            "triggered_by": "test",
            "documents_processed": 5,
            "chunks_ingested": 20,
            "chunks_skipped": 1,
            "error_count": 0,
        }

        with patch.object(tracker_module, "mlflow", mock_mlflow):
            t.log_ingestion_run(summary)

        mock_mlflow.start_run.assert_called_once()
        mock_mlflow.log_params.assert_called_once()
        mock_mlflow.log_metrics.assert_called_once()
        mock_mlflow.log_dict.assert_called_once()

    def test_log_query_result_calls_mlflow(self):
        t = _disabled_tracker()
        t.enabled = True

        mock_mlflow = MagicMock()
        mock_mlflow.start_run.return_value.__enter__ = MagicMock(return_value=MagicMock())
        mock_mlflow.start_run.return_value.__exit__ = MagicMock(return_value=False)

        with patch.object(tracker_module, "mlflow", mock_mlflow):
            t.log_query_result("What is Chancenkarte?", {"answer": "..."})

        mock_mlflow.start_run.assert_called_once()
        mock_mlflow.log_text.assert_called_once()
        mock_mlflow.log_dict.assert_called_once()

    def test_log_ingestion_run_swallows_mlflow_error(self):
        t = _disabled_tracker()
        t.enabled = True

        mock_mlflow = MagicMock()
        mock_mlflow.start_run.side_effect = RuntimeError("mlflow down")

        with patch.object(tracker_module, "mlflow", mock_mlflow):
            # Should not raise
            t.log_ingestion_run({"run_id": "x", "documents_processed": 0, "chunks_ingested": 0, "chunks_skipped": 0})

    def test_log_query_result_swallows_mlflow_error(self):
        t = _disabled_tracker()
        t.enabled = True

        mock_mlflow = MagicMock()
        mock_mlflow.start_run.side_effect = RuntimeError("mlflow down")

        with patch.object(tracker_module, "mlflow", mock_mlflow):
            t.log_query_result("q", {})


class TestMLflowTrackerInit:
    def test_disabled_when_enable_mlflow_false(self):
        with patch("src.observability.mlflow_tracker.settings") as mock_settings:
            mock_settings.enable_mlflow = False
            mock_settings.mlflow_tracking_uri = "http://localhost:5000"
            mock_settings.mlflow_experiment_name = "test"
            t = MLflowTracker()
        assert t.enabled is False

    def test_disabled_when_mlflow_not_available(self):
        with (
            patch.object(tracker_module, "MLFLOW_AVAILABLE", False),
            patch("src.observability.mlflow_tracker.settings") as mock_settings,
        ):
            mock_settings.enable_mlflow = True
            t = MLflowTracker()
        assert t.enabled is False


# ─── Singleton ────────────────────────────────────────────────────────────────


class TestGetMLflowTracker:
    def test_returns_same_instance(self):
        # Reset singleton
        tracker_module._mlflow_tracker = None
        with patch("src.observability.mlflow_tracker.settings") as s:
            s.enable_mlflow = False
            a = get_mlflow_tracker()
            b = get_mlflow_tracker()
        assert a is b
        # Cleanup
        tracker_module._mlflow_tracker = None
