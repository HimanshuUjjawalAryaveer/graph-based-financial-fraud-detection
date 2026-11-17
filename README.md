# Graph-Based Financial Fraud Detection

**Course:** ES 678 — IIT Gandhinagar  
**Author:** Himanshu  
**Project title:** *Evaluating Graph-Derived Features for Illicit Cryptocurrency Transaction Detection*

## 1. Project overview

This project investigates whether simple graph-derived features improve the detection of illicit Bitcoin transactions compared with transaction-level numerical features alone.

The experiment uses the Elliptic Bitcoin transaction dataset and compares two XGBoost classifiers:

- **Baseline:** 165 original transaction-level numerical features.
- **Graph-enhanced:** the same 165 features plus `in_degree`, `out_degree`, and `total_degree` (168 model input features).

This is a graph-feature augmentation experiment. It is **not** a graph neural network (GNN) implementation.

### Research question

Can incorporating graph-based features into traditional machine learning models improve illicit cryptocurrency transaction detection compared with transaction-level features alone, and how does the change affect precision, recall, and PR-AUC on chronologically later transactions?

## 2. Dataset

The project uses three raw CSV files:

| File | Content |
|---|---|
| `elliptic_txs_features.csv` | Transaction IDs, time steps, and 165 numerical features |
| `elliptic_txs_edgelist.csv` | Directed transaction-to-transaction edges (`txId1`, `txId2`) |
| `elliptic_txs_classes.csv` | Transaction labels |

Dataset summary from the inspected files:

| Quantity | Count |
|---|---:|
| Transaction nodes | 203,769 |
| Directed edges | 234,355 |
| Numerical transaction features | 165 |
| Time steps | 49 |
| Unknown labels | 157,205 |
| Licit transactions (original class 2) | 42,019 |
| Illicit transactions (original class 1) | 4,545 |
| Labelled transactions | 46,564 |

The raw class distribution is strongly imbalanced. Unknown-labelled transactions were excluded from supervised training and evaluation, but the full supplied graph was used when deriving node degree features.

### Target mapping

- Original class `1` → target `1` (illicit; positive class).
- Original class `2` → target `0` (licit; negative class).
- Original label `unknown` → excluded from supervised model tables.

The inspected feature, edge, and class files contained no missing values. All feature IDs matched class IDs, and the graph inspection found no missing endpoint timestamps. All supplied edges connected transactions from the same time step.

## 3. Preprocessing and graph construction

1. Inspect the raw feature, edge, and class files.
2. Merge features and labels using transaction IDs.
3. Remove unknown-labelled transactions from the supervised modelling table.
4. Map licit and illicit labels to binary targets.
5. Create a chronological train/validation/test split.
6. Build a directed graph from the edge list.
7. Compute node degree features and join them to the transaction table.
8. Prepare baseline and graph-enhanced datasets.

For node `v`:

- `in_degree(v)`: number of incoming edges.
- `out_degree(v)`: number of outgoing edges.
- `total_degree(v) = in_degree(v) + out_degree(v)`.

The graph inspection found 234,355 directed edges, no duplicate directed edges, and no reciprocal edge pairs. The computed `unique_neighbours` statistic was identical to `total_degree` in this dataset and was not retained as a separate model feature. Validation found no missing or negative graph-feature values and confirmed degree consistency. Across all nodes, mean in-degree was 1.1501, mean out-degree was 1.1501, and mean total degree was 2.3002. The mean total degree was 2.012 for illicit nodes and 3.095 for licit nodes; this is descriptive and does not, on its own, prove that degree features are predictive.

**Important graph-availability assumption:** graph features were calculated from the supplied graph before selecting labelled rows. This is a transductive, within-time-step setup: the relevant graph for a time step is assumed to be available when transactions from that time step are scored. It is not strict streaming inference in which a transaction must be scored before the remainder of its time-step graph is observed. No future labels were used to calculate degree features.

## 4. Chronological evaluation design

The labelled records were split by time step, not randomly:

| Partition | Time steps | Rows | Licit | Illicit | Illicit share |
|---|---:|---:|---:|---:|---:|
| Train | 1–34 | 29,894 | 26,432 | 3,462 | 11.58% |
| Validation | 35–41 | 7,829 | 7,154 | 675 | 8.62% |
| Test | 42–49 | 8,841 | 8,433 | 408 | about 4.6% |

The test split contains only 408 illicit examples. Accuracy therefore should not be used alone to judge detection quality.

## 5. Models and training configuration

Both classifiers used the same XGBoost configuration to keep the feature-set comparison controlled:

| Parameter | Value |
|---|---:|
| `n_estimators` | 300 |
| `max_depth` | 6 |
| `learning_rate` | 0.05 |
| `subsample` | 0.8 |
| `colsample_bytree` | 0.8 |
| `scale_pos_weight` | 7.6349 (negative/positive training count) |
| `eval_metric` | `logloss` |
| `random_state` | 42 |
| prediction threshold | 0.5 |

No threshold optimisation or hyperparameter search was performed. The test set was not used for fitting or threshold selection.

## 6. Results

### Validation results

| Metric | Baseline XGBoost | Graph-enhanced XGBoost |
|---|---:|---:|
| Accuracy | 0.9838 | 0.9829 |
| Precision | 0.9190 | 0.9105 |
| Recall | 0.8904 | 0.8889 |
| F1-score | 0.9044 | 0.8996 |
| PR-AUC (Average Precision) | 0.9335 | 0.9334 |

The baseline is marginally higher on all reported validation metrics.

### Held-out test results

| Metric | Baseline XGBoost | Graph-enhanced XGBoost |
|---|---:|---:|
| Accuracy | 0.9641443 | 0.9640312 |
| Precision | 0.6511628 | 0.6480263 |
| Recall | 0.4803922 | 0.4828431 |
| F1-score | 0.5528914 | 0.5533708 |
| PR-AUC (Average Precision) | 0.5581592 | 0.5543083 |

Test confusion matrices (actual rows; predicted columns; class order licit, illicit):

**Baseline:** `[[8328, 105], [212, 196]]`  
**Graph-enhanced:** `[[8326, 107], [211, 197]]`

The graph model detects one additional illicit transaction at the 0.5 threshold, but produces two additional false positives. The baseline has slightly higher precision and average precision; the graph model has marginally higher recall and F1. These small descriptive differences do not establish a meaningful or statistically significant improvement.

### Temporal test analysis (baseline)

| Time step | Transactions | Illicit | Illicit % | Precision | Recall | F1 | AP |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 42 | 2,154 | 239 | 11.10 | 0.8977 | 0.8075 | 0.8502 | 0.8937 |
| 43 | 1,370 | 24 | 1.75 | 0.0000 | 0.0000 | 0.0000 | 0.0434 |
| 44 | 1,591 | 24 | 1.51 | 0.0357 | 0.0417 | 0.0385 | 0.0445 |
| 45 | 1,221 | 5 | 0.41 | 0.0000 | 0.0000 | 0.0000 | 0.0107 |
| 46 | 712 | 2 | 0.28 | 0.0556 | 0.5000 | 0.1000 | 0.2581 |
| 47 | 846 | 22 | 2.60 | 0.0000 | 0.0000 | 0.0000 | 0.0539 |
| 48 | 471 | 36 | 7.64 | 0.0000 | 0.0000 | 0.0000 | 0.2353 |
| 49 | 476 | 56 | 11.76 | 0.0909 | 0.0179 | 0.0299 | 0.2482 |

The validation-to-test drop is large: baseline AP falls from 0.9335 to 0.5582. Performance also varies sharply by test time step. For example, time step 42 is strong, while several later steps have zero recall at the fixed 0.5 threshold. Time step 46 contains only two illicit examples, so its metrics are unstable. Time step 48 has zero recall at threshold 0.5 but AP 0.2353, showing that fixed-threshold classification and ranking-based AP measure different aspects of performance.

In this project, `PR-AUC` is reported as scikit-learn's `average_precision_score` (Average Precision, AP). It is not claimed to be trapezoidal integration of a precision-recall curve.

## 7. Main conclusion

The three tested degree features did not improve overall held-out test performance in this experiment. The graph-enhanced model's slightly higher recall and F1 do not offset its lower precision and AP enough to support an overall improvement claim. The temporal analysis also shows that good validation scores did not translate into similarly strong results on later time steps.

This result applies only to the selected degree features, XGBoost configuration, dataset, chronological split, and graph-availability assumption. It does not show that graph information in general is unhelpful.

## 8. Limitations and future work

- Only three simple degree features were tested; richer structural or temporal graph features and GNNs were not evaluated.
- Within-time-step graph availability is assumed; strict prospective streaming inference was not tested.
- The illicit class is rare, especially in some individual time steps.
- A fixed 0.5 classification threshold was used; no validation-based threshold selection was performed.
- Only one chronological split was evaluated; no confidence intervals, bootstrap tests, or significance tests were run.
- Feature drift and prevalence change are possible explanations for temporal performance variation, but were not causally established.

Potential next steps include distribution-shift analysis, additional chronological evaluation windows, validation-only threshold selection, calibration, richer graph features, and prospective evaluation with only information available at scoring time.

## 9. Project structure

```text
graph_fraud_detection/
├── data/
│   ├── raw/
│   │   ├── elliptic_txs_features.csv
│   │   ├── elliptic_txs_edgelist.csv
│   │   └── elliptic_txs_classes.csv
│   └── processed/
│       ├── elliptic_merged.csv
│       ├── elliptic_labeled.csv
│       ├── train.csv
│       ├── validation.csv
│       ├── test.csv
│       ├── elliptic_graph_features.csv
│       ├── train_graph.csv
│       ├── validation_graph.csv
│       └── test_graph.csv
├── models/
│   ├── xgboost_baseline.json
│   └── xgboost_graph.json
├── results/                   # metrics, tables, figures
├── src/
│   ├── inspect_dataset.py
│   ├── preprocess.py
│   ├── prepare_labeled_data.py
│   ├── split_data.py
│   ├── eda.py
│   ├── feature_analysis.py
│   ├── class_feature_analysis.py
│   ├── train_baseline.py
│   ├── feature_importance.py
│   ├── inspect_graph.py
│   ├── create_graph_features.py
│   ├── validate_graph_features.py
│   ├── prepare_graph_data.py
│   ├── train_graph_model.py
│   ├── compare_models.py
│   ├── evaluate_test.py
│   ├── temporal_analysis.py
│   ├── final_visualizations.py
│   └── inspect_results.py
└── README.md
```

This tree documents the project layout used during the work. Some files may be generated during execution; see the cleanup guidance below before deleting them.

## 10. Environment and reproduction

The project was developed on Windows 11 with Python 3.11. From Git Bash:

```bash
cd /d/ML_Projects/graph_fraud_detection
python -m venv .venv
source .venv/Scripts/activate
python -m pip install --upgrade pip
python -m pip install pandas numpy scikit-learn xgboost matplotlib
```

Place the three raw CSV files in `data/raw/`. Run scripts from the project root using `python src/<script_name>.py` (or `cd src` and run `python <script_name>.py` if the script assumes the latter working directory). The captured run was launched from `src/`; retain the same working-directory convention used in each script.

Suggested end-to-end run order, subject to the path conventions in the scripts:

```bash
python src/inspect_dataset.py
python src/preprocess.py
python src/prepare_labeled_data.py
python src/split_data.py
python src/eda.py
python src/feature_analysis.py
python src/class_feature_analysis.py
python src/train_baseline.py
python src/feature_importance.py
python src/inspect_graph.py
python src/create_graph_features.py
python src/validate_graph_features.py
python src/prepare_graph_data.py
python src/train_graph_model.py
python src/compare_models.py
python src/evaluate_test.py
python src/temporal_analysis.py
python src/final_visualizations.py
python src/inspect_results.py
```

The exact run order should be checked against the current script path handling. The output log records scripts being run from the `src` directory in Git Bash.

### Key generated artifacts

- `models/xgboost_baseline.json` — trained baseline model.
- `models/xgboost_graph.json` — trained graph-enhanced model.
- `results/baseline_validation_metrics.json` — baseline validation metrics.
- `results/graph_validation_metrics.json` — graph model validation metrics.
- `results/validation_model_comparison.csv` — validation comparison.
- `results/test_evaluation_results.json` — held-out test evaluation.
- `results/temporal_analysis.csv` — per-time-step baseline analysis.
- `results/*.png` — EDA and final visualisations.

## 11. File cleanup: what is essential and what can be regenerated?

Do **not** delete files solely because they are intermediate. Use the following safe policy:

| File/category | Recommendation | Reason |
|---|---|---|
| Three original CSV files in `data/raw/` | **Keep** | Source data required to reproduce the project. |
| All scripts in `src/` | **Keep for now** | Some are pipeline scripts; others document inspections and validation. |
| `models/*.json` | **Keep** | Saved trained models used for evaluation/inference. |
| Metrics CSV/JSON and plots in `results/` | **Keep** | Evidence supporting the report. |
| `elliptic_merged.csv`, `elliptic_labeled.csv` | Keep until pipeline rerun succeeds; then optional | Derived files that can be regenerated from raw data and preprocessing scripts. |
| `train.csv`, `validation.csv`, `test.csv` | **Keep** | Exact chronological split snapshots; regenerable but useful for reproducibility. |
| `elliptic_graph_features.csv` | **Keep** | Cached graph features; regenerable from the edge list but useful for audit and avoiding recomputation. |
| `train_graph.csv`, `validation_graph.csv`, `test_graph.csv` | Keep until graph pipeline rerun succeeds; then optional | Derived graph-augmented split files. |
| `__pycache__/`, `*.pyc`, `.ipynb_checkpoints/`, temporary logs | **Safe to remove** | Recreated automatically or not required for the model. |
| `.venv/` | Keep locally; do not commit to Git | Recreate with pip if needed; it can be large. Save dependencies in `requirements.txt` before removing it. |
| Old duplicate report `.tex`/`.zip` files | Optional to remove after checking the new package | Avoid confusion between old and updated report versions. |

The inspection/EDA scripts (`inspect_dataset.py`, `eda.py`, `feature_analysis.py`, `class_feature_analysis.py`, `inspect_graph.py`, `validate_graph_features.py`, `feature_importance.py`, `inspect_results.py`) are diagnostic rather than strictly required for fitting the final models. They are not useless: they document checks and analysis. Archive them instead of deleting them if you want a smaller production-oriented pipeline.

**Important:** this README is based on the 46-page results PDF and the project structure/scripts previously reviewed. The PDF is not a complete directory listing. Before I can identify exact orphaned or duplicate files, compare dependencies, or recommend deleting a specific file with confidence, inspect the actual current folder or provide a ZIP of the project directory.

## 12. References

1. M. Weber et al. “Anti-money laundering in Bitcoin: Experimenting with graph convolutional networks for financial forensics.” KDD Workshop on Anomaly Detection in Finance, 2019. https://doi.org/10.1145/3338468.3358948
2. T. Chen and C. Guestrin. “XGBoost: A Scalable Tree Boosting System.” KDD, 2016. https://doi.org/10.1145/2939672.2939785
3. F. Pedregosa et al. “Scikit-learn: Machine Learning in Python.” *Journal of Machine Learning Research*, 12:2825–2830, 2011. https://jmlr.org/papers/v12/pedregosa11a.html

## 13. Report package

The accompanying `graph_based_financial_fraud_report.tex` follows the supplied ES 678 / NeurIPS 2026 template and uses the style file `neurips_2026.sty`. The selected report figures are included under `results/`. Replace the roll-number placeholder in the LaTeX file before submission. Verify the AI-use disclosure and all details against the course policy and your local project outputs.
