
import json
import pandas as pd

from pathlib import Path
from xgboost import XGBClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    average_precision_score,
    confusion_matrix,
    classification_report
)


# ==========================================
# 1. PROJECT DIRECTORIES
# ==========================================

ROOT_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = ROOT_DIR / "data" / "processed"
MODEL_DIR = ROOT_DIR / "models"
RESULT_DIR = ROOT_DIR / "results"

RESULT_DIR.mkdir(exist_ok=True)


# ==========================================
# 2. LOAD TEST DATASETS
# ==========================================

print("\nLoading test datasets...")

test = pd.read_csv(DATA_DIR / "test.csv")

test_graph = pd.read_csv(
    DATA_DIR / "test_graph.csv"
)

print("Baseline test shape:", test.shape)
print("Graph test shape:", test_graph.shape)


# ==========================================
# 3. VERIFY DATA ALIGNMENT
# ==========================================

assert test["txId"].equals(test_graph["txId"]), (
    "Transaction IDs do not match!"
)

assert test["target"].equals(test_graph["target"]), (
    "Target labels do not match!"
)

print("\nTest datasets are correctly aligned.")


# ==========================================
# 4. SELECT MODEL FEATURES
# ==========================================

exclude_columns = [
    "txId",
    "time_step",
    "target"
]

# Baseline: 165 original features
baseline_features = [
    col for col in test.columns
    if col not in exclude_columns
]

# Graph-enhanced: 165 original + 3 graph features
graph_features = [
    col for col in test_graph.columns
    if col not in exclude_columns
]

X_test = test[baseline_features]
y_test = test["target"]

X_test_graph = test_graph[graph_features]
y_test_graph = test_graph["target"]

print("\nBaseline input features:", X_test.shape[1])
print("Graph input features:", X_test_graph.shape[1])

assert X_test.shape[1] == 165
assert X_test_graph.shape[1] == 168


# ==========================================
# 5. LOAD TRAINED MODELS
# ==========================================

print("\nLoading trained models...")

baseline_model = XGBClassifier()

baseline_model.load_model(
    MODEL_DIR / "xgboost_baseline.json"
)

graph_model = XGBClassifier()

graph_model.load_model(
    MODEL_DIR / "xgboost_graph.json"
)

print("Both models loaded successfully!")


# ==========================================
# 6. VERIFY MODEL FEATURE COMPATIBILITY
# ==========================================

baseline_model_features = (
    baseline_model.get_booster().feature_names
)

graph_model_features = (
    graph_model.get_booster().feature_names
)

assert X_test.columns.tolist() == baseline_model_features, (
    "Baseline feature names or order do not match!"
)

assert X_test_graph.columns.tolist() == graph_model_features, (
    "Graph feature names or order do not match!"
)

print("\nBoth models have matching test features!")


# ==========================================
# 7. GENERATE PREDICTIONS
# ==========================================

print("\nGenerating predictions...")

baseline_pred = baseline_model.predict(X_test)

baseline_prob = baseline_model.predict_proba(
    X_test
)[:, 1]

graph_pred = graph_model.predict(X_test_graph)

graph_prob = graph_model.predict_proba(
    X_test_graph
)[:, 1]

print("Predictions generated successfully!")


# ==========================================
# 8. EVALUATION FUNCTION
# ==========================================

def evaluate_model(name, y_true, y_pred, y_prob):

    metrics = {
        "accuracy": float(
            accuracy_score(y_true, y_pred)
        ),

        "precision": float(
            precision_score(
                y_true, y_pred, zero_division=0
            )
        ),

        "recall": float(
            recall_score(
                y_true, y_pred, zero_division=0
            )
        ),

        "f1_score": float(
            f1_score(
                y_true, y_pred, zero_division=0
            )
        ),

        "pr_auc": float(
            average_precision_score(
                y_true, y_prob
            )
        )
    }

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1]
    )

    print(f"\n========== {name} ==========")

    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")

    print("\nConfusion Matrix:")
    print(cm)

    print("\nClassification Report:")
    print(
        classification_report(
            y_true,
            y_pred,
            zero_division=0
        )
    )

    return metrics, cm.tolist()


# ==========================================
# 9. EVALUATE BOTH MODELS
# ==========================================

baseline_metrics, baseline_cm = evaluate_model(
    "BASELINE XGBOOST",
    y_test,
    baseline_pred,
    baseline_prob
)

graph_metrics, graph_cm = evaluate_model(
    "GRAPH-ENHANCED XGBOOST",
    y_test_graph,
    graph_pred,
    graph_prob
)


# ==========================================
# 10. SAVE TEST RESULTS
# ==========================================

results = {
    "baseline_xgboost": {
        "metrics": baseline_metrics,
        "confusion_matrix": baseline_cm
    },

    "graph_enhanced_xgboost": {
        "metrics": graph_metrics,
        "confusion_matrix": graph_cm
    }
}

output_path = RESULT_DIR / "test_evaluation_results.json"

with open(output_path, "w") as file:
    json.dump(results, file, indent=4)


# ==========================================
# 11. FINAL SUMMARY
# ==========================================

print("\n========== FINAL SUMMARY ==========")

print("\nBaseline PR-AUC:",
      round(baseline_metrics["pr_auc"], 4))

print("Graph PR-AUC:",
      round(graph_metrics["pr_auc"], 4))

print("\nTest evaluation completed successfully!")

print("Results saved to:")
print(output_path)
