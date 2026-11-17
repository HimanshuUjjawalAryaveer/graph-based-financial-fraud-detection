
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path
from xgboost import XGBClassifier

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data" / "processed"
RESULTS_DIR = ROOT_DIR / "results"
MODELS_DIR = ROOT_DIR / "models"

# Load training data
train = pd.read_csv(DATA_DIR / "train.csv")

feature_columns = [
    col for col in train.columns
    if col.startswith("feature_")
]

X_train = train[feature_columns]
y_train = train["target"]

# Load trained baseline model
model = XGBClassifier()
model.load_model(MODELS_DIR / "xgboost_baseline.json")

# Extract feature importance
importance = pd.DataFrame({
    "Feature": feature_columns,
    "Importance": model.feature_importances_
})

importance = importance.sort_values(
    by="Importance",
    ascending=False
)

print("\n========== FEATURE IMPORTANCE ==========")
print("\nTop 20 important features:")
print(importance.head(20).to_string(index=False))

# Plot top 20
top_features = importance.head(20).sort_values(
    by="Importance"
)

plt.figure(figsize=(10, 7))

plt.barh(
    top_features["Feature"],
    top_features["Importance"]
)

plt.xlabel("Importance")
plt.ylabel("Feature")
plt.title("Top 20 XGBoost Feature Importances")

plt.tight_layout()

plt.savefig(
    RESULTS_DIR / "feature_importance.png",
    dpi=300
)

plt.show()

print("\nFeature importance analysis completed!")
