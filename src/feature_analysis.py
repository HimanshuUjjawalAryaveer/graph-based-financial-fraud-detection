
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data" / "processed"
RESULTS_DIR = ROOT_DIR / "results"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Load training data only
data = pd.read_csv(DATA_DIR / "train.csv")

# Select numerical transaction features
feature_columns = [
    col for col in data.columns
    if col.startswith("feature_")
]

X = data[feature_columns]

print("\n========== FEATURE ANALYSIS ==========")

print("Number of features:", len(feature_columns))

# Statistical summary
print("\nFeature statistics:")
print(X.describe().round(3).iloc[:, :10])

# Check zero percentages
zero_percentage = (X == 0).mean() * 100

print("\nTop 10 features with highest zero percentage:")
print(zero_percentage.sort_values(ascending=False).head(10))

# Plot feature scales
feature_ranges = X.max() - X.min()

plt.figure(figsize=(12, 5))

plt.bar(
    range(len(feature_ranges)),
    feature_ranges.values
)

plt.xlabel("Feature Index")
plt.ylabel("Maximum - Minimum")
plt.title("Range of Numerical Features")
plt.tight_layout()

plt.savefig(
    RESULTS_DIR / "feature_ranges.png",
    dpi=300
)

plt.show()

print("\nFeature analysis completed!")
