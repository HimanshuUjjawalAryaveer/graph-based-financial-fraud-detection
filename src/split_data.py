
import pandas as pd
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent

INPUT_PATH = ROOT_DIR / "data" / "processed" / "elliptic_labeled.csv"
OUTPUT_DIR = ROOT_DIR / "data" / "processed"

# Load labelled data
data = pd.read_csv(INPUT_PATH)

# Sort chronologically
data = data.sort_values("time_step").reset_index(drop=True)

# Define temporal boundaries
train_end = 34
val_end = 41

# Create chronological splits
train_data = data[data["time_step"] <= train_end].copy()

val_data = data[
    (data["time_step"] > train_end) &
    (data["time_step"] <= val_end)
].copy()

test_data = data[data["time_step"] > val_end].copy()

# Print split information
def show_info(name, dataset):
    print(f"\n{name}")
    print("Shape:", dataset.shape)
    print("Time steps:",
          dataset["time_step"].min(),
          "to",
          dataset["time_step"].max())
    print("Class distribution:")
    print(dataset["target"].value_counts().sort_index())

show_info("TRAINING DATA", train_data)
show_info("VALIDATION DATA", val_data)
show_info("TEST DATA", test_data)

# Save datasets
train_data.to_csv(OUTPUT_DIR / "train.csv", index=False)
val_data.to_csv(OUTPUT_DIR / "validation.csv", index=False)
test_data.to_csv(OUTPUT_DIR / "test.csv", index=False)

print("\nAll datasets saved successfully!")
