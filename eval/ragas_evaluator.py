"""
RAG evaluation using Ragas framework.
Computes Context Precision, Answer Faithfulness, and other metrics.
"""

import asyncio
import json
from typing import Dict, Any, List, Optional
from datetime import datetime
import pandas as pd

from ragas import evaluate
from ragas.metrics import (
    context_precision,
    answer_relevancy,
    faithfulness,
    context_recall,
)
from datasets import Dataset

from src.config import settings
from src.logger import logger
from src.rag.answer_generator import get_answer_generator
from src.observability.mlflow_tracker import get_mlflow_tracker


class RagasEvaluator:
    """
    Evaluate RAG pipeline using Ragas framework.
    
    Metrics:
    - Context Precision: Are retrieved contexts relevant to query?
    - Context Recall: Are all necessary information retrieved?
    - Faithfulness: Is answer grounded in retrieved context?
    - Answer Relevancy: Does answer address the query?
    """

    def __init__(self):
        self.generator = get_answer_generator()
        self.mlflow = get_mlflow_tracker()

    async def evaluate_from_dataset(
        self,
        dataset_path: str,
        output_dir: str = "eval/results",
    ) -> Dict[str, Any]:
        """
        Evaluate RAG pipeline on a test dataset.
        
        Args:
            dataset_path: Path to eval_dataset.json
            output_dir: Output directory for results
            
        Returns:
            Evaluation results with metrics and analysis
        """
        try:
            # Load dataset
            logger.info(f"Loading evaluation dataset from {dataset_path}")
            with open(dataset_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            questions = data["questions"]
            ground_truths = data["ground_truths"]
            
            if len(questions) != len(ground_truths):
                raise ValueError("Questions and ground truths length mismatch")
            
            logger.info(f"Loaded {len(questions)} test cases")
            
            # Generate predictions and contexts
            predictions = []
            contexts = []
            
            for i, question in enumerate(questions):
                logger.debug(f"Evaluating question {i+1}/{len(questions)}")
                
                try:
                    result = await self.generator.generate_answer(question)
                    predictions.append(result["answer"])
                    
                    # Extract source texts as context
                    source_texts = [
                        f"[{src.get('title', 'Unknown')}]\n{src.get('url', '')}"
                        for src in result["sources"]
                    ]
                    contexts.append(source_texts)
                    
                except Exception as e:
                    logger.error(f"Failed to generate answer for question {i+1}: {e}")
                    predictions.append("")
                    contexts.append([])
            
            # Prepare dataset for Ragas
            eval_dataset = Dataset.from_dict({
                "question": questions,
                "ground_truth": ground_truths,
                "answer": predictions,
                "contexts": contexts,
            })
            
            logger.info("Running Ragas evaluation...")
            
            # Run evaluation
            results = evaluate(
                eval_dataset,
                metrics=[
                    context_precision,
                    context_recall,
                    faithfulness,
                    answer_relevancy,
                ],
            )
            
            # Convert to dataframe for analysis
            results_df = results.to_pandas()
            
            # Calculate aggregate metrics
            aggregate_metrics = {
                "context_precision": float(results_df["context_precision"].mean()),
                "context_recall": float(results_df["context_recall"].mean()),
                "faithfulness": float(results_df["faithfulness"].mean()),
                "answer_relevancy": float(results_df["answer_relevancy"].mean()),
                "count": len(results_df),
            }
            
            logger.info(f"Evaluation completed", extra=aggregate_metrics)
            
            # Save results
            import os
            os.makedirs(output_dir, exist_ok=True)
            
            output_file = f"{output_dir}/ragas_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            
            report = {
                "timestamp": datetime.utcnow().isoformat(),
                "aggregate_metrics": aggregate_metrics,
                "sample_results": results_df.head(5).to_dict("records"),
                "full_results_csv": f"{output_dir}/ragas_detailed.csv",
            }
            
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
            
            # Save detailed CSV
            csv_file = f"{output_dir}/ragas_detailed.csv"
            results_df.to_csv(csv_file, index=False)
            
            logger.info(f"Results saved to {output_file}")
            
            # Log to MLflow
            if self.mlflow:
                self._log_to_mlflow(aggregate_metrics, results_df)
            
            return report
            
        except Exception as e:
            logger.error(f"Evaluation failed: {e}", exc_info=True)
            raise

    def _log_to_mlflow(self, metrics: Dict[str, float], results_df: pd.DataFrame):
        """Log evaluation results to MLflow."""
        try:
            import mlflow
            
            with mlflow.start_run(run_name="ragas_evaluation"):
                # Log metrics
                for metric_name, value in metrics.items():
                    if metric_name != "count":
                        mlflow.log_metric(metric_name, value)
                
                # Log params
                mlflow.log_params({
                    "test_count": metrics["count"],
                    "timestamp": datetime.utcnow().isoformat(),
                })
                
                # Log artifact: CSV
                results_df.to_csv("/tmp/ragas_results.csv", index=False)
                mlflow.log_artifact("/tmp/ragas_results.csv")
                
                logger.info("Results logged to MLflow")
                
        except Exception as e:
            logger.warning(f"MLflow logging failed: {e}")

    async def evaluate_single_query(
        self,
        question: str,
        expected_answer: str,
    ) -> Dict[str, Any]:
        """
        Evaluate a single query-answer pair.
        
        Useful for quick validation during development.
        """
        try:
            result = await self.generator.generate_answer(question)
            
            # Manual evaluation
            evaluation = {
                "question": question,
                "generated_answer": result["answer"],
                "expected_answer": expected_answer,
                "sources": result["sources"],
                "sources_count": len(result["sources"]),
            }
            
            return evaluation
            
        except Exception as e:
            logger.error(f"Single query evaluation failed: {e}")
            return {"error": str(e)}


async def main():
    """Run evaluation script."""
    import sys
    
    evaluator = RagasEvaluator()
    
    dataset_path = "eval/eval_dataset.json"
    
    if len(sys.argv) > 1:
        dataset_path = sys.argv[1]
    
    report = await evaluator.evaluate_from_dataset(dataset_path)
    
    print("\n" + "=" * 60)
    print("EVALUATION REPORT")
    print("=" * 60)
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    asyncio.run(main())
