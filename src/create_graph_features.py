
import pandas as pd
import networkx as nx

from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT_DIR / "data" / "raw"
PROCESSED_DIR = ROOT_DIR / "data" / "processed"

# Load transaction information
transactions = pd.read_csv(
    PROCESSED_DIR / "elliptic_merged.csv"
)

# Load graph edges
edges = pd.read_csv(
    RAW_DIR / "elliptic_txs_edgelist.csv"
)

print("\n========== CREATING GRAPH FEATURES ==========")

# Create directed graph
G = nx.from_pandas_edgelist(
    edges,
    source="txId1",
    target="txId2",
    create_using=nx.DiGraph()
)

print("Number of graph nodes:", G.number_of_nodes())
print("Number of graph edges:", G.number_of_edges())

# Calculate graph features
in_degree = dict(G.in_degree())
out_degree = dict(G.out_degree())

# Include isolated transactions too
transactions["in_degree"] = (
    transactions["txId"].map(in_degree).fillna(0)
)

transactions["out_degree"] = (
    transactions["txId"].map(out_degree).fillna(0)
)

transactions["total_degree"] = (
    transactions["in_degree"] +
    transactions["out_degree"]
)

transactions["unique_neighbours"] = (
    transactions["txId"].map(
        dict(G.degree())
    ).fillna(0)
)

# Save graph-enhanced dataset
output_path = (
    PROCESSED_DIR / "elliptic_graph_features.csv"
)

transactions.to_csv(output_path, index=False)

print("\n========== GRAPH FEATURE SUMMARY ==========")

print(
    transactions[
        [
            "in_degree",
            "out_degree",
            "total_degree",
            "unique_neighbours"
        ]
    ].describe()
)

print("\nTransactions with zero total degree:")
print((transactions["total_degree"] == 0).sum())

print("\nSaved graph features to:")
print(output_path)

print("\nGraph feature creation completed!")
