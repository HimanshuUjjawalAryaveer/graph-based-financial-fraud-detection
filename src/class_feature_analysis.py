import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data" / "processed"
RESULTS_DIR = ROOT_DIR / "results"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Load training data
data = pd.read_csv(DATA_DIR / "train.csv")

# Select numerical features
feature_columns = [
    col for col in data.columns
    if col.startswith("feature_")
]

# Calculate mean feature values for each class
class_means = data.groupby("target")[feature_columns].mean()

print("\n========== CLASS FEATURE ANALYSIS ==========")

print("\nMean values for first 10 features:")
print(class_means.iloc[:, :10].round(4))

# Calculate absolute difference between class means
mean_difference = (
    class_means.loc[1] - class_means.loc[0]
).abs()

top_features = mean_difference.sort_values(
    ascending=False
).head(15)

print("\nTop 15 features with largest mean differences:")
print(top_features.round(4))

# Plot top 15 features
plt.figure(figsize=(12, 6))

top_features.sort_values().plot(kind="barh")

plt.xlabel("Absolute Difference in Mean Values")
plt.ylabel("Feature")
plt.title("Top 15 Features Differing Between Licit and Illicit Transactions")

plt.tight_layout()

plt.savefig(
    RESULTS_DIR / "class_feature_differences.png",
    dpi=300
)

plt.show()

print("\nClass feature analysis completed!")