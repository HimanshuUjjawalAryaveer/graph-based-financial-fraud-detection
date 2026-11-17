
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path

# ==========================================
# PROJECT PATHS
# ==========================================

ROOT_DIR = Path(__file__).resolve().parent.parent
RESULT_DIR = ROOT_DIR / "results"
RESULT_DIR.mkdir(exist_ok=True)

# ==========================================
# LOAD RESULTS
# ==========================================

print("\nLoading experimental results...")

with open(RESULT_DIR / "test_evaluation_results.json", "r") as file:
    test_results = json.load(file)

validation_df = pd.read_csv(
    RESULT_DIR / "validation_model_comparison.csv"
)

temporal_df = pd.read_csv(
    RESULT_DIR / "temporal_analysis.csv"
)

print("All result files loaded successfully!")

# ==========================================
# HELPER: MODEL COMPARISON PLOT
# ==========================================

def plot_model_comparison(baseline, graph, title, filename):

    metrics = [
        "accuracy",
        "precision",
        "recall",
        "f1_score",
        "pr_auc"
    ]

    labels = [
        "Accuracy",
        "Precision",
        "Recall",
        "F1-score",
        "PR-AUC"
    ]

    x = np.arange(len(metrics))
    width = 0.35

    fig, ax = plt.subplots(figsize=(11, 6))

    bars1 = ax.bar(
        x - width / 2,
        [baseline[m] for m in metrics],
        width,
        label="Baseline XGBoost"
    )

    bars2 = ax.bar(
        x + width / 2,
        [graph[m] for m in metrics],
        width,
        label="Graph-enhanced XGBoost"
    )

    ax.set_title(title, fontsize=15, fontweight="bold")
    ax.set_ylabel("Score")
    ax.set_xlabel("Evaluation Metrics")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.12)

    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    ax.bar_label(bars1, fmt="%.3f", padding=3, fontsize=9)
    ax.bar_label(bars2, fmt="%.3f", padding=3, fontsize=9)

    plt.tight_layout()
    plt.savefig(RESULT_DIR / filename, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved: {filename}")


# ==========================================
# 1. VALIDATION MODEL COMPARISON
# ==========================================

print("\nCreating validation comparison plot...")

validation_baseline = {
    "accuracy": 0.983778,
    "precision": 0.918960,
    "recall": 0.890370,
    "f1_score": 0.904439,
    "pr_auc": 0.933509
}

validation_graph = {
    "accuracy": 0.982884,
    "precision": 0.910470,
    "recall": 0.888889,
    "f1_score": 0.899550,
    "pr_auc": 0.933447
}

plot_model_comparison(
    validation_baseline,
    validation_graph,
    "Validation Performance: Baseline vs Graph-enhanced XGBoost",
    "validation_model_comparison.png"
)


# ==========================================
# 2. TEST MODEL COMPARISON
# ==========================================

print("\nCreating test comparison plot...")

baseline_test = test_results["baseline_xgboost"]["metrics"]
graph_test = test_results["graph_enhanced_xgboost"]["metrics"]

plot_model_comparison(
    baseline_test,
    graph_test,
    "Test Performance: Baseline vs Graph-enhanced XGBoost",
    "test_model_comparison.png"
)


# ==========================================
# 3. CONFUSION MATRIX COMPARISON
# ==========================================

print("\nCreating confusion matrix comparison...")

baseline_cm = np.array(
    test_results["baseline_xgboost"]["confusion_matrix"]
)

graph_cm = np.array(
    test_results["graph_enhanced_xgboost"]["confusion_matrix"]
)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

matrices = [
    (baseline_cm, "Baseline XGBoost"),
    (graph_cm, "Graph-enhanced XGBoost")
]

for ax, (cm, title) in zip(axes, matrices):

    image = ax.imshow(cm, interpolation="nearest")

    ax.set_title(title, fontweight="bold")
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("Actual Label")

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])

    ax.set_xticklabels(["Licit", "Illicit"])
    ax.set_yticklabels(["Licit", "Illicit"])

    for i in range(2):
        for j in range(2):
            ax.text(
                j,
                i,
                str(cm[i, j]),
                ha="center",
                va="center",
                fontsize=13
            )

fig.colorbar(image, ax=axes.ravel().tolist(), shrink=0.85)

fig.suptitle(
    "Test Set Confusion Matrix Comparison",
    fontsize=15,
    fontweight="bold"
)

plt.tight_layout()
plt.savefig(
    RESULT_DIR / "test_confusion_matrix_comparison.png",
    dpi=300,
    bbox_inches="tight"
)
plt.close()

print("Saved: test_confusion_matrix_comparison.png")


# ==========================================
# 4. TEMPORAL PERFORMANCE PLOT
# ==========================================

print("\nCreating temporal performance plots...")

metrics = [
    ("precision", "Precision"),
    ("recall", "Recall"),
    ("f1_score", "F1-score"),
    ("pr_auc", "PR-AUC")
]

fig, axes = plt.subplots(2, 2, figsize=(13, 9))

for ax, (column, label) in zip(axes.flatten(), metrics):

    ax.plot(
        temporal_df["time_step"],
        temporal_df[column],
        marker="o",
        linewidth=2
    )

    ax.set_title(label, fontweight="bold")
    ax.set_xlabel("Time Step")
    ax.set_ylabel("Score")
    ax.set_ylim(-0.05, 1.05)
    ax.set_xticks(temporal_df["time_step"])
    ax.grid(True, alpha=0.3)

fig.suptitle(
    "Baseline XGBoost: Temporal Performance Across Test Time Steps",
    fontsize=15,
    fontweight="bold"
)

plt.tight_layout()

plt.savefig(
    RESULT_DIR / "temporal_performance_summary.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("Saved: temporal_performance_summary.png")


# ==========================================
# 5. TEMPORAL CLASS DISTRIBUTION
# ==========================================

print("\nCreating temporal class distribution plot...")

fig, ax = plt.subplots(figsize=(10, 5))

ax.plot(
    temporal_df["time_step"],
    temporal_df["illicit_percentage"],
    marker="o",
    linewidth=2,
    label="Illicit transaction percentage"
)

ax.set_title(
    "Illicit Transaction Distribution Across Test Time Steps",
    fontweight="bold"
)

ax.set_xlabel("Time Step")
ax.set_ylabel("Illicit Transactions (%)")
ax.set_xticks(temporal_df["time_step"])
ax.grid(True, alpha=0.3)
ax.legend()

plt.tight_layout()

plt.savefig(
    RESULT_DIR / "temporal_class_distribution.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("Saved: temporal_class_distribution.png")


# ==========================================
# COMPLETION
# ==========================================

print("\n====================================")
print("ALL FINAL VISUALIZATIONS GENERATED!")
print("====================================")

print("\nGenerated files:")

for file in sorted(RESULT_DIR.glob("*.png")):
    print("-", file.name)

print("\nVisualisation process completed successfully!")
