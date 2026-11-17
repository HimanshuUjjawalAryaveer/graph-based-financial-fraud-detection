
import pandas as pd
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data" / "processed"
RAW_DIR = ROOT_DIR / "data" / "raw"

# Load graph-enhanced data
data = pd.read_csv(
    DATA_DIR / "elliptic_graph_features.csv"
)

# Load original edges
edges = pd.read_csv(
    RAW_DIR / "elliptic_txs_edgelist.csv"
)

graph_columns = [
    "in_degree",
    "out_degree",
    "total_degree",
    "unique_neighbours"
]

print("\n========== GRAPH FEATURE VALIDATION ==========")

# Check missing values
print("\nMissing graph feature values:")
print(data[graph_columns].isnull().sum())

# Check negative values
print("\nNegative graph feature values:")
print((data[graph_columns] < 0).sum())

# Check duplicate edges
print("\nTotal edges:", len(edges))
print("Unique directed edges:", len(edges.drop_duplicates()))

# Check reciprocal edges
edge_pairs = set(
    zip(edges["txId1"], edges["txId2"])
)

reciprocal_count = sum(
    (destination, source) in edge_pairs
    for source, destination in edge_pairs
)

print("\nEdges with a reciprocal counterpart:", reciprocal_count)

# Check degree consistency
print("\nChecking total degree consistency:")
print(
    (
        data["total_degree"] ==
        data["in_degree"] + data["out_degree"]
    ).all()
)

# Check unique neighbour consistency
print("\nChecking unique neighbour consistency:")
print(
    (
        data["unique_neighbours"] ==
        data["total_degree"]
    ).all()
)

# Compare graph features by class (labelled transactions only)
labelled = data[data["class"].isin(["1", "2"])].copy()

print("\nGraph feature means by class:")
print(
    labelled.groupby("class")[graph_columns]
    .mean()
    .round(3)
)

print("\nGraph feature validation completed!")
