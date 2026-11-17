
import pandas as pd
from pathlib import Path

# Project dataset location
DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"

features_path = DATA_DIR / "elliptic_txs_features.csv"
edges_path = DATA_DIR / "elliptic_txs_edgelist.csv"
classes_path = DATA_DIR / "elliptic_txs_classes.csv"

# Load the three datasets
features = pd.read_csv(features_path, header=None)
edges = pd.read_csv(edges_path)
classes = pd.read_csv(classes_path)

# Give meaningful names to feature columns
features.columns = (
    ["txId", "time_step"]
    + [f"feature_{i}" for i in range(1, features.shape[1] - 1)]
)

print("\n========== FEATURES DATASET ==========")
print("Shape:", features.shape)
print("\nFirst 5 rows:")
print(features.head())

print("\n========== EDGES DATASET ==========")
print("Shape:", edges.shape)
print("\nFirst 5 rows:")
print(edges.head())

print("\n========== CLASSES DATASET ==========")
print("Shape:", classes.shape)
print("\nFirst 5 rows:")
print(classes.head())

print("\n========== CLASS DISTRIBUTION ==========")
print(classes["class"].value_counts(dropna=False))

print("\n========== TIME STEP DISTRIBUTION ==========")
print(features["time_step"].value_counts().sort_index())

print("\n========== MISSING VALUES ==========")
print("Features missing values:", features.isnull().sum().sum())
print("Edges missing values:", edges.isnull().sum().sum())
print("Classes missing values:", classes.isnull().sum().sum())

print("\n========== DATASET INSPECTION COMPLETE ==========")




print("\n========== TRANSACTION ID VALIDATION ==========")

# Extract transaction IDs
feature_ids = set(features["txId"])
class_ids = set(classes["txId"])

# Check transaction ID overlap
print("Feature IDs:", len(feature_ids))
print("Class IDs:", len(class_ids))
print("IDs in features but not classes:",
      len(feature_ids - class_ids))
print("IDs in classes but not features:",
      len(class_ids - feature_ids))

# Check graph edge connections
source_ids = set(edges.iloc[:, 0])
destination_ids = set(edges.iloc[:, 1])

all_edge_ids = source_ids | destination_ids

print("Unique source transactions:", len(source_ids))
print("Unique destination transactions:", len(destination_ids))
print("Unique transactions in graph:", len(all_edge_ids))

print("Edge IDs missing from features:",
      len(all_edge_ids - feature_ids))

print("\n========== VALIDATION COMPLETE ==========")





print("\n========== FEATURE INFORMATION ==========")

print("Total columns:", features.shape[1])

print("\nFirst 10 column names:")
print(features.columns[:10].tolist())

print("\nData types:")
print(features.dtypes.value_counts())

print("\nTime step information:")
print("Minimum:", features["time_step"].min())
print("Maximum:", features["time_step"].max())
print("Unique time steps:", features["time_step"].nunique())

print("\nFirst 5 transaction IDs:")
print(features["txId"].head().tolist())

