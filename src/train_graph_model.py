
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

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data" / "processed"
MODEL_DIR = ROOT_DIR / "models"
RESULT_DIR = ROOT_DIR / "results"

MODEL_DIR.mkdir(exist_ok=True)
RESULT_DIR.mkdir(exist_ok=True)

# Load datasets
train = pd.read_csv(DATA_DIR / "train_graph.csv")
validation = pd.read_csv(DATA_DIR / "validation_graph.csv")

# Select input features
exclude_columns = ["txId", "time_step", "target"]

feature_columns = [
    col for col in train.columns
    if col not in exclude_columns
]

X_train = train[feature_columns]
y_train = train["target"]

X_val = validation[feature_columns]
y_val = validation["target"]

print("Training samples:", len(X_train))
print("Validation samples:", len(X_val))
print("Number of input features:", len(feature_columns))

# Handle class imbalance
negative_count = (y_train == 0).sum()
positive_count = (y_train == 1).sum()

scale_pos_weight = negative_count / positive_count

print("Scale positive weight:", scale_pos_weight)

# Train XGBoost model
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

print("\nTraining graph-enhanced XGBoost model...")

model.fit(X_train, y_train)

print("Training completed!")

# Predictions
y_pred = model.predict(X_val)
y_prob = model.predict_proba(X_val)[:, 1]

# Evaluation
metrics = {
    "accuracy": float(accuracy_score(y_val, y_pred)),
    "precision": float(precision_score(y_val, y_pred)),
    "recall": float(recall_score(y_val, y_pred)),
    "f1_score": float(f1_score(y_val, y_pred)),
    "pr_auc": float(average_precision_score(y_val, y_prob))
}

print("\n========== GRAPH MODEL RESULTS ==========")

for metric, value in metrics.items():
    print(f"{metric}: {value:.4f}")

print("\nConfusion Matrix:")
print(confusion_matrix(y_val, y_pred))

print("\nClassification Report:")
print(classification_report(y_val, y_pred))

# Save model
model.save_model(MODEL_DIR / "xgboost_graph.json")

# Save metrics
with open(
    RESULT_DIR / "graph_validation_metrics.json",
    "w"
) as file:
    json.dump(metrics, file, indent=4)

print("\nModel saved successfully!")
print("Metrics saved successfully!")
