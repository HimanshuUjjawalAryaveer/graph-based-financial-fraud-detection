
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path
from xgboost import XGBClassifier

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    average_precision_score
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
# 2. LOAD TEST DATA
# ==========================================

print("\nLoading test dataset...")

test = pd.read_csv(DATA_DIR / "test.csv")

exclude_columns = [
    "txId",
    "time_step",
    "target"
]

feature_columns = [
    col for col in test.columns
    if col not in exclude_columns
]

X_test = test[feature_columns]
y_test = test["target"]

time_steps = test["time_step"]

print("Test samples:", len(test))
print("Test time steps:", sorted(time_steps.unique()))


# ==========================================
# 3. LOAD BASELINE MODEL
# ==========================================

print("\nLoading baseline model...")

model = XGBClassifier()

model.load_model(
    MODEL_DIR / "xgboost_baseline.json"
)

print("Model loaded successfully!")


# ==========================================
# 4. GENERATE PREDICTIONS
# ==========================================

print("\nGenerating predictions...")

predictions = model.predict(X_test)

probabilities = model.predict_proba(X_test)[:, 1]

test_results = test.copy()

test_results["prediction"] = predictions
test_results["probability"] = probabilities


# ==========================================
# 5. TEMPORAL EVALUATION
# ==========================================

results = []

print("\n========== TEMPORAL PERFORMANCE ==========")

for step in sorted(test_results["time_step"].unique()):

    step_data = test_results[
        test_results["time_step"] == step
    ]

    y_true = step_data["target"]
    y_pred = step_data["prediction"]
    y_prob = step_data["probability"]

    # Check whether both classes exist
    if y_true.nunique() < 2:
        print(f"Skipping time step {step}: only one class present")
        continue

    metrics = {
        "time_step": int(step),
        "transactions": len(step_data),
        "illicit_transactions": int((y_true == 1).sum()),
        "licit_transactions": int((y_true == 0).sum()),
        "illicit_percentage": float((y_true == 1).mean() * 100),
        "precision": precision_score(
            y_true, y_pred, zero_division=0
        ),
        "recall": recall_score(
            y_true, y_pred, zero_division=0
        ),
        "f1_score": f1_score(
            y_true, y_pred, zero_division=0
        ),
        "pr_auc": average_precision_score(
            y_true, y_prob
        )
    }

    results.append(metrics)

    print(f"\nTime step: {step}")
    print("Transactions:", metrics["transactions"])
    print("Illicit transactions:", metrics["illicit_transactions"])
    print("Illicit percentage:", round(metrics["illicit_percentage"], 2))
    print("Precision:", round(metrics["precision"], 4))
    print("Recall:", round(metrics["recall"], 4))
    print("F1-score:", round(metrics["f1_score"], 4))
    print("PR-AUC:", round(metrics["pr_auc"], 4))


# ==========================================
# 6. SAVE TEMPORAL RESULTS
# ==========================================

temporal_df = pd.DataFrame(results)

temporal_df.to_csv(
    RESULT_DIR / "temporal_analysis.csv",
    index=False
)

print("\nTemporal results saved successfully!")


# ==========================================
# 7. PLOT PERFORMANCE TRENDS
# ==========================================

metrics_to_plot = [
    "precision",
    "recall",
    "f1_score",
    "pr_auc"
]

for metric in metrics_to_plot:

    plt.figure(figsize=(9, 5))

    plt.plot(
        temporal_df["time_step"],
        temporal_df[metric],
        marker="o"
    )

    plt.xlabel("Time Step")
    plt.ylabel(metric.replace("_", " ").title())

    plt.title(
        f"Baseline XGBoost: {metric.replace('_', ' ').title()} Across Test Time Steps"
    )

    plt.xticks(temporal_df["time_step"])
    plt.grid(True, alpha=0.3)

    plt.tight_layout()

    plt.savefig(
        RESULT_DIR / f"temporal_{metric}.png",
        dpi=300
    )

    plt.close()


print("\nAll temporal plots saved successfully!")
print("\nTemporal analysis completed!")
