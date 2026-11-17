
import pandas as pd
from pathlib import Path

# Define project directories
ROOT_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT_DIR / "data" / "raw"
PROCESSED_DIR = ROOT_DIR / "data" / "processed"

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# Load features
features = pd.read_csv(
    RAW_DIR / "elliptic_txs_features.csv",
    header=None
)

# Assign column names
features.columns = (
    ["txId", "time_step"]
    + [f"feature_{i}" for i in range(1, 166)]
)

# Load classes
classes = pd.read_csv(
    RAW_DIR / "elliptic_txs_classes.csv"
)

# Merge using transaction ID
data = features.merge(
    classes,
    on="txId",
    how="inner",
    validate="one_to_one"
)

# Convert labels into readable values
data["class"] = data["class"].astype(str).str.strip()

print("\n========== MERGED DATASET ==========")
print("Shape:", data.shape)
print("\nFirst 5 rows:")
print(data.head())

print("\nClass distribution:")
print(data["class"].value_counts())

print("\nTime step range:")
print(data["time_step"].min(), "to", data["time_step"].max())

# Save merged dataset
output_path = PROCESSED_DIR / "elliptic_merged.csv"
data.to_csv(output_path, index=False)

print("\nSaved successfully to:")
print(output_path)
