
import pandas as pd
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT_DIR / "data" / "raw"
PROCESSED_DIR = ROOT_DIR / "data" / "processed"

# Load edges
edges = pd.read_csv(
    RAW_DIR / "elliptic_txs_edgelist.csv"
)

# Load transaction information
transactions = pd.read_csv(
    PROCESSED_DIR / "elliptic_merged.csv"
)

print("\n========== GRAPH INSPECTION ==========")

print("\nEdge dataset shape:")
print(edges.shape)

print("\nEdge columns:")
print(edges.columns.tolist())

print("\nFirst 10 edges:")
print(edges.head(10))

print("\nUnique source transactions:")
print(edges.iloc[:, 0].nunique())

print("\nUnique destination transactions:")
print(edges.iloc[:, 1].nunique())

print("\nTotal unique transactions in edges:")
print(
    pd.unique(edges.to_numpy().ravel()).shape[0]
)

# Add transaction time steps to edges
time_map = transactions.set_index("txId")["time_step"]

edges["source_time"] = edges.iloc[:, 0].map(time_map)
edges["destination_time"] = edges.iloc[:, 1].map(time_map)

print("\n========== EDGE TIME ANALYSIS ==========")

print("\nMissing source timestamps:")
print(edges["source_time"].isna().sum())

print("\nMissing destination timestamps:")
print(edges["destination_time"].isna().sum())

print("\nSource time vs destination time:")
print(
    (edges["source_time"] - edges["destination_time"])
    .describe()
)

print("\nSource time greater than destination time:")
print(
    (edges["source_time"] > edges["destination_time"]).sum()
)

print("\nDestination time greater than source time:")
print(
    (edges["destination_time"] > edges["source_time"]).sum()
)

print("\nSame time step:")
print(
    (edges["source_time"] == edges["destination_time"]).sum()
)

print("\nGraph inspection completed!")
