# Experiment Folder Guide (`experiment/`)

This folder contains the pipeline for **data scarcity** experiments on Re3py, with progressive dataset reductions and 10-fold cross-validation evaluation.

## What's inside

- `data_scarcity_preprocessing.py`: creates reduced target files (10/20/50/70/90% removed) deterministically.
- `fold_mapper.py`: maps original folds to reduced datasets.
- `run_experiment_bagging.py`: runs Bagging experiments.
- `run_experiment_random_forest.py`: runs Random Forest experiments.
- `run_experiment_boosting.py`: runs Boosting experiments.
- `run_experiment_single_tree.py`: runs Single Tree experiments.
- `run_*_full_test.py`: extended variants for fuller testing.
- `run_pipeline.py`: multi-step orchestrator (preprocessing + fold mapping + experiment stage + aggregation).
- `cli.py`: interactive CLI.
- `analyze_results.py`: summary/analysis of aggregated results.

## Prerequisites

From the repository root:

```bash
pip install -r requirements.txt
python setup.py install
```

Then move into the experiment folder:

```bash
cd experiment
```

## Supported datasets

- `basket`
- `carcinogenesis`
- `imdb_big`
- `movie`
- `mutagenesis`
- `stack_big`
- `uwcse`
- `webkb`
- `yelp_big`

## Recommended workflow (manual and reliable)

### 1) Preprocessing (data reduction)

```bash
python data_scarcity_preprocessing.py --dataset basket --seed 2864
```

For all datasets:

```bash
python data_scarcity_preprocessing.py --dataset basket --all --seed 2864
```

Main output:
- `data/data_reduced/<dataset>/target_removed_*.txt`
- `data/data_reduced/<dataset>/metadata/removal_mapping.json`

### 2) Fold mapping

```bash
python fold_mapper.py --dataset basket
```

For all datasets:

```bash
python fold_mapper.py --dataset basket --all
```

Main output:
- `data/data_reduced/<dataset>/folds/folds_removed_*.txt`

### 3) Run experiments (choose model)

Bagging:

```bash
python run_experiment_bagging_full_test.py --dataset basket --config-name agg_all --n-jobs 2
```

Random Forest:

```bash
python run_experiment_random_forest_full_test.py --dataset basket --config-name agg_all --n-jobs 2
```

Boosting:

```bash
python run_experiment_boosting_full_test.py --dataset basket --config-name agg_all
```

Single Tree:

```bash
python run_experiment_single_tree_full_test.py --dataset basket --config-name agg_all --n-jobs 2
```

Main output:
- `experiment/results/<dataset>/results_summary.json`
- `experiment/results/<dataset>/results.csv`
- `experiment/results/<dataset>/results_detailed.json`
- logs in `logs/`

### 4) Analyze aggregated results

```bash
python analyze_results.py
```

Reads `experiment/results/combined_results.csv` and prints summary tables.

## Quick pipeline

Also available:

```bash
python run_pipeline.py --dataset basket --seed 2864
python run_pipeline.py --all --seed 2864
```

and interactive CLI:

```bash
python cli.py
```

## Important operational notes

- Reductions are **cumulative and deterministic** (10%, 20%, 50%, 70%, 90%).
- If `data/data_reduced/<dataset>` is missing, run preprocessing again.
- If reduced folds are missing, rerun `fold_mapper.py`.
- `--use-original-folds` on runners uses `data/folds/<dataset>/folds1.txt`.
- `--config-name` changes experiment suffix/configuration (default: `agg_all`).

## End-to-end example (single dataset)

```bash
cd experiment
python data_scarcity_preprocessing.py --dataset basket --seed 2864
python fold_mapper.py --dataset basket
python run_experiment_bagging.py --dataset basket --config-name agg_all --n-jobs 2
```

## Quick troubleshooting

- Error "Reduced data not found": run `data_scarcity_preprocessing.py` first.
- Error "No schema file found": check `.s` files in `data/datasets/<dataset>/`.
- Fold error: verify `data/folds/<dataset>/folds1.txt` and regenerate reduced folds.
- High runtime on large datasets: reduce `--n-jobs` or run one dataset at a time.
