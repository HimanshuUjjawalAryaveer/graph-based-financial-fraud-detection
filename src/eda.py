
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data" / "processed"
RESULTS_DIR = ROOT_DIR / "results"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Load datasets
train = pd.read_csv(DATA_DIR / "train.csv")
validation = pd.read_csv(DATA_DIR / "validation.csv")
test = pd.read_csv(DATA_DIR / "test.csv")

datasets = {
    "Training": train,
    "Validation": validation,
    "Testing": test
}

# Display class distribution
print("\n========== CLASS DISTRIBUTION ==========")

for name, df in datasets.items():
    print(f"\n{name} Dataset")
    print("Total transactions:", len(df))

    counts = df["target"].value_counts().sort_index()

    print("Licit (0):", counts.get(0, 0))
    print("Illicit (1):", counts.get(1, 0))

    print("\nPercentage distribution:")
    print((counts / len(df) * 100).round(2))

# Prepare data for visualization
labels = ["Training", "Validation", "Testing"]
licit_counts = [
    (df["target"] == 0).sum()
    for df in datasets.values()
]
illicit_counts = [
    (df["target"] == 1).sum()
    for df in datasets.values()
]

# Plot class distribution
x = range(len(labels))
width = 0.35

plt.figure(figsize=(9, 5))

plt.bar(
    [i - width/2 for i in x],
    licit_counts,
    width,
    label="Licit"
)

plt.bar(
    [i + width/2 for i in x],
    illicit_counts,
    width,
    label="Illicit"
)

plt.xticks(list(x), labels)
plt.ylabel("Number of transactions")
plt.title("Licit vs Illicit Transactions Across Datasets")
plt.legend()
plt.tight_layout()

plt.savefig(
    RESULTS_DIR / "class_distribution.png",
    dpi=300
)

plt.show()

print("\nChart saved successfully!")



# ==========================================
# STEP 4.2 - TIME STEP ANALYSIS
# ==========================================

print("\n========== TIME STEP ANALYSIS ==========")

# Combine all labelled datasets
all_data = pd.concat(
    [train, validation, test],
    ignore_index=True
)

# Count licit and illicit transactions per time step
time_analysis = all_data.groupby(
    ["time_step", "target"]
).size().unstack(fill_value=0)

# Ensure both classes exist as columns
time_analysis = time_analysis.reindex(
    columns=[0, 1],
    fill_value=0
)

time_analysis.columns = ["Licit", "Illicit"]

# Calculate total transactions
time_analysis["Total"] = (
    time_analysis["Licit"] + time_analysis["Illicit"]
)

# Calculate illicit percentage
time_analysis["Illicit_Percentage"] = (
    time_analysis["Illicit"] /
    time_analysis["Total"] * 100
)

print("\nFirst 10 time steps:")
print(time_analysis.head(10))

print("\nLast 10 time steps:")
print(time_analysis.tail(10))

# Plot transaction counts over time
plt.figure(figsize=(12, 5))

plt.plot(
    time_analysis.index,
    time_analysis["Licit"],
    label="Licit",
    marker="."
)

plt.plot(
    time_analysis.index,
    time_analysis["Illicit"],
    label="Illicit",
    marker="."
)

plt.xlabel("Time Step")
plt.ylabel("Number of Transactions")
plt.title("Transaction Distribution Across Time Steps")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()

plt.savefig(
    RESULTS_DIR / "transactions_over_time.png",
    dpi=300
)

plt.show()

# Plot illicit transaction percentage
plt.figure(figsize=(12, 5))

plt.plot(
    time_analysis.index,
    time_analysis["Illicit_Percentage"],
    marker="."
)

plt.xlabel("Time Step")
plt.ylabel("Illicit Transactions (%)")
plt.title("Illicit Transaction Percentage Over Time")
plt.grid(alpha=0.3)
plt.tight_layout()

plt.savefig(
    RESULTS_DIR / "illicit_percentage_over_time.png",
    dpi=300
)

plt.show()

print("\nTime step analysis completed!")
