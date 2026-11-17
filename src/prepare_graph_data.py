
import pandas as pd
from pathlib import Path

# Project directories
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data" / "processed"

# Load original datasets
train = pd.read_csv(DATA_DIR / "train.csv")
validation = pd.read_csv(DATA_DIR / "validation.csv")
test = pd.read_csv(DATA_DIR / "test.csv")

# Load graph features
graph = pd.read_csv(DATA_DIR / "elliptic_graph_features.csv")

# Keep only the required graph features
graph_columns = [
    "txId",
    "in_degree",
    "out_degree",
    "total_degree"
]

graph = graph[graph_columns].copy()

# Check for duplicate transaction IDs
assert graph["txId"].is_unique, "Duplicate txId found in graph features!"

# Function to add graph features
def add_graph_features(data, split_name):

    original_rows = len(data)

    # Merge using transaction ID
    enhanced = data.merge(
        graph,
        on="txId",
        how="left",
        validate="one_to_one"
    )

    # Validate merge
    assert len(enhanced) == original_rows, (
        f"Row count changed in {split_name}!"
    )

    assert not enhanced[
        ["in_degree", "out_degree", "total_degree"]
    ].isnull().any().any(), (
        f"Missing graph features in {split_name}!"
    )

    print(f"\n{split_name} dataset:")
    print("Original rows:", original_rows)
    print("Enhanced rows:", len(enhanced))
    print("Total features:", len(enhanced.columns))

    return enhanced


# Prepare all three datasets
train_graph = add_graph_features(train, "Train")
validation_graph = add_graph_features(validation, "Validation")
test_graph = add_graph_features(test, "Test")

# Save enhanced datasets
train_graph.to_csv(
    DATA_DIR / "train_graph.csv",
    index=False
)

validation_graph.to_csv(
    DATA_DIR / "validation_graph.csv",
    index=False
)

test_graph.to_csv(
    DATA_DIR / "test_graph.csv",
    index=False
)

print("\nAll graph-enhanced datasets saved successfully!")
