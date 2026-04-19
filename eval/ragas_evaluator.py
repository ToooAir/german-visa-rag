"""
RAG evaluation using Ragas framework.
Computes Faithfulness and Answer Relevancy metrics.

Metrics chosen:
- Faithfulness: Is the answer grounded in the retrieved context? (hallucination detection)
- Answer Relevancy: Does the answer address the question? (response quality)

Context Precision and Context Recall are excluded from this run to stay within
the LLM API call budget (~100 calls for 10 questions vs ~260 for all 4 metrics).
"""

import asyncio
import json
import warnings
from datetime import datetime
from typing import Any, Dict

import pandas as pd
from datasets import Dataset
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings, ChatOpenAI, OpenAIEmbeddings
from ragas import evaluate
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import answer_relevancy, faithfulness

from src.config import settings
from src.logger import logger
from src.observability.mlflow_tracker import get_mlflow_tracker
from src.rag.answer_generator import AnswerGenerator
from src.rag.hybrid_retriever import HybridRetriever
from src.vector_db.qdrant_client_wrapper import get_qdrant_client


class RagasEvaluator:
    """
    Evaluate RAG pipeline using Ragas framework.

    Metrics:
    - Faithfulness: Is answer grounded in retrieved context? (hallucination proxy)
    - Answer Relevancy: Does answer address the query? (response quality)
    """

    def __init__(self):
        import os

        qdrant = get_qdrant_client()
        retriever = HybridRetriever(qdrant_client=qdrant)
        self.generator = AnswerGenerator(retriever=retriever)
        self.mlflow = get_mlflow_tracker()

        # Ragas judge LLM + embeddings — mirrors the active provider so judge and
        # answer generator use the same model family (avoids judge/system mismatch).
        if settings.use_azure_openai:
            os.environ["AZURE_OPENAI_API_KEY"] = settings.azure_openai_api_key or ""
            self.ragas_llm = LangchainLLMWrapper(
                AzureChatOpenAI(
                    azure_deployment=settings.azure_llm_deployment,
                    azure_endpoint=settings.azure_openai_endpoint,
                    api_key=settings.azure_openai_api_key,
                    api_version=settings.azure_openai_api_version,
                )
            )
            self.ragas_embeddings = LangchainEmbeddingsWrapper(
                AzureOpenAIEmbeddings(
                    azure_deployment=settings.azure_embedding_deployment,
                    azure_endpoint=settings.azure_openai_endpoint,
                    api_key=settings.azure_openai_api_key,
                    api_version=settings.azure_openai_api_version,
                )
            )
        else:
            os.environ["OPENAI_API_KEY"] = settings.openai_api_key
            self.ragas_llm = LangchainLLMWrapper(
                ChatOpenAI(
                    model=settings.openai_model,
                    api_key=settings.openai_api_key,
                    base_url=settings.openai_api_base,
                )
            )
            self.ragas_embeddings = LangchainEmbeddingsWrapper(
                OpenAIEmbeddings(
                    model=settings.embedding_model,
                    api_key=settings.openai_api_key,
                    base_url=settings.openai_api_base,
                )
            )

    async def evaluate_from_dataset(
        self,
        dataset_path: str,
        output_dir: str = "eval/results",
        run_name: str = "ragas_evaluation",
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
            logger.info(f"Loading evaluation dataset from {dataset_path}")
            with open(dataset_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            questions = data["questions"]
            ground_truths = data["ground_truths"]

            if len(questions) != len(ground_truths):
                raise ValueError("Questions and ground truths length mismatch")

            logger.info(f"Loaded {len(questions)} test cases")

            predictions = []
            contexts = []

            for i, question in enumerate(questions):
                logger.info(f"Generating answer {i+1}/{len(questions)}: {question[:60]}...")

                try:
                    result = await self.generator.generate_answer(question)
                    predictions.append(result["answer"])
                    # Use actual retrieved text (not just title/URL) for faithful evaluation
                    context_texts = [t for t in result.get("contexts", []) if t]
                    contexts.append(context_texts if context_texts else [""])

                except Exception as e:
                    logger.error(f"Failed to generate answer for question {i+1}: {e}")
                    predictions.append("")
                    contexts.append([""])

            # ragas 0.4.x uses new column names; column_map bridges old Dataset format
            eval_dataset = Dataset.from_dict(
                {
                    "question": questions,
                    "answer": predictions,
                    "contexts": contexts,
                    "ground_truth": ground_truths,
                }
            )

            logger.info("Running Ragas evaluation (faithfulness + answer_relevancy)...")

            with warnings.catch_warnings():
                warnings.simplefilter("ignore", DeprecationWarning)
                results = evaluate(
                    eval_dataset,
                    metrics=[faithfulness, answer_relevancy],
                    llm=self.ragas_llm,
                    embeddings=self.ragas_embeddings,
                    column_map={
                        "user_input": "question",
                        "response": "answer",
                        "retrieved_contexts": "contexts",
                        "reference": "ground_truth",
                    },
                    raise_exceptions=False,
                    batch_size=1,  # Serialize to respect GitHub Models rate limits
                )

            results_df = results.to_pandas()

            aggregate_metrics = {
                "faithfulness": float(results_df["faithfulness"].mean()),
                "answer_relevancy": float(results_df["answer_relevancy"].mean()),
                "count": len(results_df),
            }

            logger.info("Evaluation completed", extra=aggregate_metrics)

            import os

            os.makedirs(output_dir, exist_ok=True)

            output_file = f"{output_dir}/ragas_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

            report = {
                "timestamp": datetime.utcnow().isoformat(),
                "metrics_evaluated": ["faithfulness", "answer_relevancy"],
                "aggregate_metrics": aggregate_metrics,
                "sample_results": results_df.to_dict("records"),
                "full_results_csv": f"{output_dir}/ragas_detailed.csv",
            }

            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(report, f, indent=2, ensure_ascii=False)

            results_df.to_csv(f"{output_dir}/ragas_detailed.csv", index=False)

            logger.info(f"Results saved to {output_file}")

            if self.mlflow:
                self._log_to_mlflow(aggregate_metrics, results_df, run_name=run_name)

            return report

        except Exception as e:
            logger.error(f"Evaluation failed: {e}", exc_info=True)
            raise

    def _log_to_mlflow(self, metrics: Dict[str, float], results_df: pd.DataFrame, run_name: str = "ragas_evaluation"):
        """Log evaluation results to MLflow."""
        try:
            import mlflow

            with mlflow.start_run(run_name=run_name):
                for metric_name, value in metrics.items():
                    if metric_name != "count":
                        mlflow.log_metric(metric_name, value)

                mlflow.log_params(
                    {
                        "test_count": metrics["count"],
                        "metrics": "faithfulness,answer_relevancy",
                        "timestamp": datetime.utcnow().isoformat(),
                    }
                )

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
        """Evaluate a single query-answer pair for quick validation."""
        try:
            result = await self.generator.generate_answer(question)

            return {
                "question": question,
                "generated_answer": result["answer"],
                "expected_answer": expected_answer,
                "sources": result["sources"],
                "sources_count": len(result["sources"]),
                "contexts_count": len(result.get("contexts", [])),
            }

        except Exception as e:
            logger.error(f"Single query evaluation failed: {e}")
            return {"error": str(e)}


async def main():
    """Run evaluation script.

    Usage:
        python -m eval.ragas_evaluator [dataset_path] [run_name]

    Example:
        python -m eval.ragas_evaluator eval/eval_dataset.json ragas_run5
    """
    import sys

    evaluator = RagasEvaluator()

    dataset_path = sys.argv[1] if len(sys.argv) > 1 else "eval/eval_dataset.json"
    run_name = sys.argv[2] if len(sys.argv) > 2 else "ragas_evaluation"

    report = await evaluator.evaluate_from_dataset(dataset_path, run_name=run_name)

    print("\n" + "=" * 60)
    print("EVALUATION REPORT")
    print("=" * 60)
    print(json.dumps(report["aggregate_metrics"], indent=2, ensure_ascii=False))
    print("\nPer-sample results:")
    for r in report["sample_results"]:
        q = r.get("question", "")[:60]
        f = r.get("faithfulness", "N/A")
        ar = r.get("answer_relevancy", "N/A")
        print(f"  [{f:.2f}/{ar:.2f}] {q}")


if __name__ == "__main__":
    asyncio.run(main())
