"""MLflow integration for tracking ingestion runs and experiments."""

from typing import Optional, Dict, Any
import json

try:
    import mlflow
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False

from src.config import settings
from src.logger import logger


class MLflowTracker:
    """MLflow tracking wrapper."""

    def __init__(self):
        self.enabled = settings.enable_mlflow and MLFLOW_AVAILABLE
        
        if self.enabled:
            try:
                mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
                mlflow.set_experiment(settings.mlflow_experiment_name)
                logger.info(f"MLflow initialized: {settings.mlflow_tracking_uri}")
            except Exception as e:
                logger.warning(f"MLflow initialization failed: {e}")
                self.enabled = False

    def log_ingestion_run(self, summary: Dict[str, Any]):
        """Log ingestion run to MLflow."""
        if not self.enabled:
            return
        
        try:
            with mlflow.start_run(run_name=f"ingest_{summary['run_id'][:8]}"):
                mlflow.log_params({
                    "run_id": summary["run_id"],
                    "triggered_by": summary.get("triggered_by", "unknown"),
                })
                
                mlflow.log_metrics({
                    "documents_processed": summary["documents_processed"],
                    "chunks_ingested": summary["chunks_ingested"],
                    "chunks_skipped": summary["chunks_skipped"],
                    "error_count": summary.get("error_count", 0),
                })
                
                mlflow.log_dict(
                    {"summary": summary},
                    artifact_file="ingestion_summary.json",
                )
                
                logger.debug("Ingestion run logged to MLflow")
        except Exception as e:
            logger.warning(f"MLflow logging failed: {e}")

    def log_query_result(self, query: str, result: Dict[str, Any]):
        """Log query result for evaluation."""
        if not self.enabled:
            return
        
        try:
            with mlflow.start_run(run_name="query"):
                mlflow.log_text(query, artifact_file="query.txt")
                mlflow.log_dict(result, artifact_file="result.json")
        except Exception as e:
            logger.warning(f"Query logging failed: {e}")


# Singleton instance
_mlflow_tracker: Optional[MLflowTracker] = None


def get_mlflow_tracker() -> MLflowTracker:
    """Get or create MLflow tracker singleton."""
    global _mlflow_tracker
    if _mlflow_tracker is None:
        _mlflow_tracker = MLflowTracker()
    return _mlflow_tracker
