# GradientBoosting Integration with Evaluation Metrics

## Summary of Changes

The `run_experiment.py` file has been updated to use **GradientBoosting** instead of RandomForest (Bagging), and now integrates the comprehensive evaluation metrics from the `re3py.eval.evaluation` module.

## Key Changes

### 1. **Model Switch: RandomForest → GradientBoosting**
   - Changed from `RandomForest` with `HeuristicGini` to `GradientBoosting` with `HeuristicVariance`
   - Updated default parameters:
     - `n_estimators`: 10 → 100 (more trees for boosting)
     - Added `shrinkage`: 0.1 (learning rate for gradient boosting)

### 2. **Evaluation Metrics Integration**
   - Imported evaluation classes from `re3py.eval.evaluation`:
     - **Classification**: `Accuracy`, `Precision`, `Recall`, `F1`
     - **Regression**: `MeanSquaredError`, `MeanAbsoluteError`, `RootMeanSquaredError`

### 3. **Automatic Task Detection**
   - The system now automatically detects whether the task is classification or regression
   - Based on target type: `str` → classification, `float/int` → regression

### 4. **Enhanced Metrics Calculation**

#### Classification Tasks:
   - **Binary Classification**:
     - Accuracy
     - Precision (requires positive class)
     - Recall (requires positive class)
     - F1 Score (harmonic mean of precision and recall)
   
   - **Multi-class Classification**:
     - Accuracy (primary metric)
     - Precision, Recall, F1 use accuracy as proxy

#### Regression Tasks:
   - Mean Squared Error (MSE)
   - Mean Absolute Error (MAE)
   - Root Mean Squared Error (RMSE)

### 5. **Updated Result Aggregation**
   - All metrics now computed with mean and standard deviation across folds
   - Updated CSV output to include all metrics with their statistics

### 6. **Method Renaming**
   - `run_bagging_cv()` → `run_boosting_cv()`
   - Updated all references and documentation

## Usage

Run experiments with the updated GradientBoosting model:

```bash
# Single dataset
python experiment/run_experiment.py --dataset basket

# With original folds
python experiment/run_experiment.py --dataset basket --use-original-folds

# All datasets
python experiment/run_experiment.py --all
```

## Output Files

The experiment generates three output files in `experiment/results/<dataset>/`:

1. **results_summary.json**: Complete results with metadata
2. **results.csv**: Tabular format with all metrics (mean ± std)
3. **experiment.log**: Full experiment log

### CSV Columns:
- `dataset`: Dataset name
- `reduction_%`: Data reduction percentage (10, 20, 50, 70, 90)
- For each metric: `{metric}_mean` and `{metric}_std`
  - Classification: accuracy, precision, recall, f1
  - Regression: mse, mae, rmse

## Implementation Details

### Evaluator Usage Pattern

```python
# Example: Accuracy calculation
acc_eval = Accuracy(all_classes)
acc_eval.add_many(ground_truth, predictions)
acc_eval.evaluate()
accuracy_value = acc_eval.get_measure_value()
```

### Key Features:
1. **Proper class handling**: Uses `get_all_target_values()` to extract all class values
2. **Positive class selection**: For binary classification, selects the first sorted class as positive
3. **Fallback metrics**: For multi-class, uses accuracy as proxy when precision/recall/F1 aren't directly applicable
4. **Error handling**: Gracefully handles prediction errors and missing data

## Benefits

1. **Standardized metrics**: All metrics computed using the official evaluation module
2. **Comprehensive evaluation**: Both classification and regression metrics available
3. **Flexible**: Automatically adapts to task type
4. **Consistent**: Uses the same evaluator classes across the entire codebase
5. **More powerful model**: GradientBoosting typically outperforms simple bagging

## Next Steps

To test the updated experiment:

```bash
# Test on basket dataset with 10-fold CV
cd /home/giaco/re3py_big_data
python experiment/run_experiment.py --dataset basket --use-original-folds
```

Check the results in `experiment/results/basket/results.csv`
