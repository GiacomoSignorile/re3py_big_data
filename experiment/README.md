# Data Scarcity Experiment for Re3py

## Overview

This experiment evaluates the **data scarcity robustness** of Re3py's Bagging ensemble on 9 datasets from the original paper.

### Key Characteristics

- **Model**: Bagging with AGG-All (all aggregates active)
- **Validation Strategy**: 10-fold Cross-Validation
- **Primary Metric**: Accuracy
- **Auxiliary Metrics**: F1, AUC
- **Data Reductions**: 10%, 20%, 50%, 70%, 90% (incremental)
- **Determinism**: Same instances removed at each step (cumulative removal)

## Datasets

| Dataset | Size | Type |
|---------|------|------|
| BASKET | 95 | Small |
| CARCINOGENESIS | 329 | Medium |
| IMDB | 8,816 | Large |
| MOVIE | 1,422 | Medium |
| MUTAGENESIS | 188 | Small |
| STACK | 5,855 | Large |
| UWCSE | 115 | Small |
| WEBKB | 500 | Small-Medium |
| YELP | 24,959 | Very Large |

## Directory Structure

```
scarcity_experiment/
├── data_scarcity_preprocessing.py    # Stage 1: Data reduction
├── fold_mapper.py                     # Stage 2: Fold mapping
├── run_experiment.py                  # Stage 3: Run Bagging experiments
├── run_pipeline.py                    # Orchestrator (runs all stages)
├── data_reduced/                      # Reduced datasets (created by Stage 1)
│   └── {dataset}/
│       ├── target_removed_10.txt      # Reduced target files
│       ├── target_removed_20.txt
│       ├── ...
│       ├── folds/
│       │   └── folds_removed_*.txt    # Mapped fold files
│       └── metadata/
│           └── *.json                 # Removal mappings & statistics
└── results/                           # Experiment results (created by Stage 3)
    └── {dataset}/
        ├── results_summary.json       # Per-dataset summary
        ├── results.csv                # Per-dataset metrics
        └── experiment.log
```

## Experiment Pipeline

### Stage 1: Data Preprocessing
**Script**: `data_scarcity_preprocessing.py`

Creates incremental, deterministic data reductions:
- Maintains label stratification
- Stores removed instance IDs for reproducibility
- Uses fixed random seed (default: 2864)

**Example**:
```bash
python data_scarcity_preprocessing.py --dataset basket --seed 2864
python data_scarcity_preprocessing.py --all  # All datasets
```

**Output**:
- `data_reduced/{dataset}/target_removed_*.txt` - Reduced target files
- `data_reduced/{dataset}/metadata/*.json` - Removal metadata

### Stage 2: Fold Mapping
**Script**: `fold_mapper.py`

Maps the existing 10-fold CV structure to reduced datasets:
- Filters fold instance IDs based on reduced datasets
- Maintains fold balance statistics
- Creates new fold files for each reduction level

**Example**:
```bash
python fold_mapper.py --dataset basket
python fold_mapper.py --all  # All datasets
```

**Output**:
- `data_reduced/{dataset}/folds/folds_removed_*.txt` - Mapped fold files

### Stage 3: Run Bagging Experiments
**Script**: `run_experiment.py`

Executes Bagging ensemble with 10-fold CV on each reduced dataset:
- Trains with AGG-All aggregates
- Calculates Accuracy, F1, AUC per fold
- Aggregates metrics (mean ± std)

**Example**:
```bash
python run_experiment.py --dataset basket
python run_experiment.py --all  # All datasets
```

**Output**:
- `results/{dataset}/results_summary.json` - Aggregated metrics
- `results/{dataset}/results.csv` - Tabular metrics
- `results/{dataset}/experiment.log` - Detailed log

### Stage 4: Full Pipeline Orchestrator
**Script**: `run_pipeline.py`

Runs all stages sequentially with integrated logging:

**Example**:
```bash
# Single dataset
python run_pipeline.py --dataset basket

# All datasets (full experiment)
python run_pipeline.py --all

# Custom seed
python run_pipeline.py --all --seed 12345
```

**Output**:
- `pipeline_YYYYMMDD_HHMMSS.log` - Complete pipeline log
- `pipeline_summary_YYYYMMDD_HHMMSS.json` - Pipeline execution summary
- `results/combined_results.csv` - All results in one file

## Incremental Data Reduction Logic

The key constraint is that reductions are **cumulative and deterministic**:

```
Original Dataset (100%)
    ├── Remove 10% (deterministic subset A)
    │   └── Remaining 90%
    │
    ├── Remove 20% total (subset A + new subset B)
    │   └── Remaining 80%
    │
    ├── Remove 50% total (A + B + C + D + E)
    │   └── Remaining 50%
    │
    └── ... and so on
```

This ensures:
- Reproducibility across runs
- Meaningful comparisons (100% is superset of all reductions)
- Label stratification at each level

## Results Format

### Summary JSON
```json
{
  "metadata": {
    "dataset": "basket",
    "model": "Bagging (AGG-All)",
    "validation": "10-fold CV",
    "metrics": ["accuracy", "f1", "auc"]
  },
  "results_by_reduction": {
    "10%": {
      "accuracy_mean": 0.8234,
      "accuracy_std": 0.0456,
      "f1_mean": 0.7912,
      "f1_std": 0.0523,
      "auc_mean": 0.8901,
      "auc_std": 0.0234,
      "n_folds": 10
    },
    ...
  }
}
```

### Results CSV
```csv
dataset,reduction_%,accuracy_mean,accuracy_std,f1_mean,f1_std,auc_mean,auc_std
basket,10,0.8234,0.0456,0.7912,0.0523,0.8901,0.0234
basket,20,0.8156,0.0489,0.7834,0.0561,0.8823,0.0267
...
```

## Configuration

Default parameters (editable in scripts):

| Parameter | Value | Notes |
|-----------|-------|-------|
| Random Seed | 2864 | For reproducibility |
| CV Folds | 10 | From original paper |
| Reductions | 10,20,50,70,90 | Percentages removed |
| Max Tree Depth | 5 | Bagging trees |
| N Estimators | 10 | Number of bag iterations |
| Label Stratification | Yes | Maintained at each reduction |

## Running the Full Experiment

### Quick Start (Recommended)
```bash
cd /Users/giacomo/Documents/re3py/scarcity_experiment

# Run all datasets with default settings
python run_pipeline.py --all
```

### Single Dataset for Testing
```bash
python run_pipeline.py --dataset basket
```

### Custom Configuration
```bash
python run_pipeline.py --all --seed 12345
```

## Monitoring Progress

Check the log file in real-time:
```bash
tail -f pipeline_YYYYMMDD_HHMMSS.log
```

## Expected Runtime

Approximate times (on modern machine):
- **Stage 1 (Preprocessing)**: ~2-5 minutes for all datasets
- **Stage 2 (Fold Mapping)**: ~1 minute
- **Stage 3 (Experiments)**: ~30-60 minutes (depends on dataset sizes)
- **Total**: ~1-2 hours

## Troubleshooting

### Issue: Fold file not found
```
Error: No fold file found in data/folds/{dataset}/
```
**Solution**: Ensure the dataset has `folds1.txt` in `data/folds/{dataset}/`

### Issue: Schema file (.s) not found
```
Error: No schema file found for {dataset}
```
**Solution**: Verify the `.s` file exists in `data/datasets/{dataset}/`

### Issue: Dataset creation fails
```
Error: Error creating dataset for XX% reduction
```
**Solution**: Check that preprocessing completed successfully before running experiments

### Issue: Out of memory on large datasets
**Solution**: Reduce `n_estimators` parameter in `run_experiment.py` or process datasets individually

## Reproducibility Checklist

- [x] Fixed random seed (2864)
- [x] Deterministic instance removal (cumulative)
- [x] Stratified sampling (maintains label distribution)
- [x] Saved removal mappings (in metadata/)
- [x] 10-fold CV structure preserved
- [x] Results saved with timestamps

## References

- **Original Paper**: Re3py framework paper (Petković & Škrlj)
- **Datasets**: From paper Table 1 (target facts)
- **Validation**: Paper section on evaluation (10-fold CV)
- **Model**: Bagging ensemble with AGG-All (Paper Table 2-5)
