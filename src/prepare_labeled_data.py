
import pandas as pd
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent

INPUT_PATH = ROOT_DIR / "data" / "processed" / "elliptic_merged.csv"
OUTPUT_PATH = ROOT_DIR / "data" / "processed" / "elliptic_labeled.csv"

# Load merged dataset
data = pd.read_csv(INPUT_PATH)

print("Original dataset shape:", data.shape)

# Keep only transactions with known labels
labeled_data = data[data["class"].isin(["1", "2"])].copy()

# Convert labels into binary format
# 1 = Illicit
# 0 = Licit
labeled_data["target"] = labeled_data["class"].map({
    "1": 1,
    "2": 0
}).astype(int)

# Remove original class column
labeled_data.drop(columns=["class"], inplace=True)

print("\n========== LABELED DATASET ==========")
print("Shape:", labeled_data.shape)

print("\nTarget distribution:")
print(labeled_data["target"].value_counts())

print("\nTarget percentages:")
print(labeled_data["target"].value_counts(normalize=True) * 100)

# Save labeled dataset
labeled_data.to_csv(OUTPUT_PATH, index=False)

print("\nSaved successfully:", OUTPUT_PATH)
