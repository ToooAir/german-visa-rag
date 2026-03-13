#!/bin/bash
# Ragas evaluation runner script

set -e

echo "================================"
echo "German Visa RAG - Evaluation"
echo "================================"
echo ""

# Check if dataset exists
if [ ! -f "eval/eval_dataset.json" ]; then
    echo "❌ Error: eval/eval_dataset.json not found"
    echo "Please create evaluation dataset first."
    exit 1
fi

# Check environment
if [ ! -f ".env" ]; then
    echo "❌ Error: .env file not found"
    echo "Please copy .env.example to .env and configure"
    exit 1
fi

# Load environment
source .env

echo "📊 Starting evaluation..."
echo "Dataset: eval/eval_dataset.json"
echo ""

# Run evaluation
python -m eval.ragas_evaluator eval/eval_dataset.json

echo ""
echo "✅ Evaluation complete!"
echo "Results saved to: eval/results/"
echo ""
echo "View results with:"
echo "  cat eval/results/ragas_report_*.json"
echo "  cat eval/results/ragas_detailed.csv"
