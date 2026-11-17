
import pandas as pd
import numpy as np
import json

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

# Project directories
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data" / "processed"
RESULTS_DIR = ROOT_DIR / "results"
MODELS_DIR = ROOT_DIR / "models"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# Load datasets
train = pd.read_csv(DATA_DIR / "train.csv")
validation = pd.read_csv(DATA_DIR / "validation.csv")

# Select only transaction-level numerical features
feature_columns = [
    col for col in train.columns
    if col.startswith("feature_")
]

X_train = train[feature_columns]
y_train = train["target"]

X_val = validation[feature_columns]
y_val = validation["target"]

print("========== BASELINE MODEL ==========")

print("Training samples:", len(X_train))
print("Validation samples:", len(X_val))
print("Number of features:", len(feature_columns))

# Handle class imbalance
negative_count = (y_train == 0).sum()
positive_count = (y_train == 1).sum()

scale_pos_weight = negative_count / positive_count

print("\nLicit transactions:", negative_count)
print("Illicit transactions:", positive_count)
print("Scale positive weight:", round(scale_pos_weight, 3))

# Create XGBoost model
model = XGBClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=scale_pos_weight,
    eval_metric="logloss",
    random_state=42,
    n_jobs=-1
)

# Train model
print("\nTraining model...")

model.fit(X_train, y_train)

print("Training completed!")

# Predict on validation data
y_pred = model.predict(X_val)
y_prob = model.predict_proba(X_val)[:, 1]

# Calculate evaluation metrics
accuracy = accuracy_score(y_val, y_pred)
precision = precision_score(y_val, y_pred, zero_division=0)
recall = recall_score(y_val, y_pred, zero_division=0)
f1 = f1_score(y_val, y_pred, zero_division=0)
pr_auc = average_precision_score(y_val, y_prob)

print("\n========== VALIDATION RESULTS ==========")

print("Accuracy:", round(accuracy, 4))
print("Precision:", round(precision, 4))
print("Recall:", round(recall, 4))
print("F1 Score:", round(f1, 4))
print("PR-AUC:", round(pr_auc, 4))

print("\nConfusion Matrix:")
print(confusion_matrix(y_val, y_pred))

print("\nClassification Report:")
print(classification_report(
    y_val,
    y_pred,
    target_names=["Licit", "Illicit"],
    zero_division=0
))

# Save trained model
model_path = MODELS_DIR / "xgboost_baseline.json"
model.save_model(model_path)

# Save validation metrics
metrics = {
    "accuracy": float(accuracy),
    "precision": float(precision),
    "recall": float(recall),
    "f1_score": float(f1),
    "pr_auc": float(pr_auc)
}

metrics_path = RESULTS_DIR / "baseline_validation_metrics.json"

with open(metrics_path, "w") as file:
    json.dump(metrics, file, indent=4)

print("\nModel saved to:", model_path)
print("Metrics saved to:", metrics_path)
print("\nBaseline training completed successfully!")
