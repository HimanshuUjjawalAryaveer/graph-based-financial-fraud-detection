import json
import pandas as pd
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
RESULT_DIR = ROOT_DIR / "results"

# Inspect test evaluation results
print("\n========== TEST EVALUATION RESULTS ==========")

json_path = RESULT_DIR / "test_evaluation_results.json"

with open(json_path, "r") as file:
    test_results = json.load(file)

print(json.dumps(test_results, indent=4))

# Inspect validation comparison
print("\n========== VALIDATION MODEL COMPARISON ==========")

validation_path = RESULT_DIR / "validation_model_comparison.csv"
validation_df = pd.read_csv(validation_path)

print(validation_df.to_string(index=False))

# Inspect temporal analysis
print("\n========== TEMPORAL ANALYSIS ==========")

temporal_path = RESULT_DIR / "temporal_analysis.csv"
temporal_df = pd.read_csv(temporal_path)

print(temporal_df.to_string(index=False))

print("\nAll result files inspected successfully!")