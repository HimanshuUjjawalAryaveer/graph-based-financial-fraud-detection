
import json
import pandas as pd
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
RESULT_DIR = ROOT_DIR / "results"

# Load saved validation metrics
with open(RESULT_DIR / "baseline_validation_metrics.json") as f:
    baseline = json.load(f)

with open(RESULT_DIR / "graph_validation_metrics.json") as f:
    graph = json.load(f)

# Build comparison table
comparison = pd.DataFrame({
    "Metric": [
        "Accuracy",
        "Precision",
        "Recall",
        "F1-score",
        "PR-AUC"
    ],
    "Baseline XGBoost": [
        baseline["accuracy"],
        baseline["precision"],
        baseline["recall"],
        baseline["f1_score"],
        baseline["pr_auc"]
    ],
    "Graph-enhanced XGBoost": [
        graph["accuracy"],
        graph["precision"],
        graph["recall"],
        graph["f1_score"],
        graph["pr_auc"]
    ]
})

# Calculate difference
comparison["Difference (Graph - Baseline)"] = (
    comparison["Graph-enhanced XGBoost"]
    - comparison["Baseline XGBoost"]
)

print("\n========== VALIDATION MODEL COMPARISON ==========")
print(comparison.to_string(index=False, float_format="%.4f".__mod__))

# Save comparison
comparison.to_csv(
    RESULT_DIR / "validation_model_comparison.csv",
    index=False
)

print("\nComparison saved successfully!")
print("File: results/validation_model_comparison.csv")
