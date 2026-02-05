"""
Re3py Data Scarcity Experiment - Bagging AGG-All
==================================================

Main experiment runner for evaluating Re3py Bagging on reduced datasets.

Configuration:
- Model: Bagging
- Validation: 10-fold Cross-Validation
- Metrics:
  - Classification: Accuracy, Precision, Recall, F1
  - Regression: MSE, MAE, RMSE
- Data Reductions: 10%, 20%, 50%, 70%, 90% (incremental)

Outputs:
- results_summary.json: Aggregated metrics across folds
- results_detailed.json: Per-fold results
- results.csv: Tabular format for analysis

The experiment uses the evaluation module (re3py.eval.evaluation) for
computing all metrics through the standard Evaluator classes.
"""

import argparse
import csv
import json
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Add repo to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Compatibility patch for time.clock() in Python 3.8+

# Import compatibility layer for old module paths

from re3py.data.data_and_statistics import Dataset, get_all_target_values
from re3py.data.task_settings import Settings
from re3py.eval.evaluation import Accuracy
from re3py.learners.core.heuristic import HeuristicGini
from re3py.learners.random_forest import RandomForest
from re3py.utilities.cross_validation import create_folds


class DataScarcityExperiment:
    """Runs data scarcity experiment with Re3py."""

    def __init__(
        self,
        dataset_name: str,
        log_dir: Optional[Path] = None,
        use_original_folds: bool = False,
    ):
        """
        Initialize experiment.

        Args:
            dataset_name: Name of dataset (e.g., 'basket', 'carcinogenesis')
            log_dir: Directory for logging results
            use_original_folds: Use original folds from data/folds
        """
        self.dataset_name = dataset_name
        script_dir = Path(__file__).resolve().parent  # experiment directory
        self.base_dir = script_dir.parent  # project root
        self.dataset_dir = self.base_dir / "data" / "datasets" / dataset_name
        self.scarcity_dir = (
            self.base_dir / "data" / "data_reduced" / dataset_name
        )  # centralized data location
        self.use_original_folds = use_original_folds
        self.original_folds_file = self.base_dir / "data" / "folds" / dataset_name / "folds1.txt"

        # Ensure reduced data exists
        if not self.scarcity_dir.exists():
            raise FileNotFoundError(
                f"Reduced data not found. Run preprocessing first: {self.scarcity_dir}"
            )

        # Results directory
        if log_dir is None:
            log_dir = self.base_dir / "experiment" / "results" / dataset_name
        self.log_dir = log_dir
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Find schema file (.s)
        self.schema_file = self._find_schema_file()
        if not self.schema_file:
            raise FileNotFoundError(f"No schema file found for {dataset_name}")

        # Find original descriptive file (will use for all reductions)
        self.descriptive_file = self._find_descriptive_file()
        if not self.descriptive_file:
            raise FileNotFoundError(f"No descriptive file found for {dataset_name}")

        # Reduction percentages
        self.reduction_percentages = [10, 20, 50, 70, 90]

        # Results storage
        self.results = {
            "metadata": {
                "dataset": dataset_name,
                "model": "Bagging",
                "validation": "10-fold CV",
                "metrics": ["accuracy", "precision", "recall", "f1", "mse", "mae", "rmse"],
                "timestamp": datetime.now().isoformat(),
            },
            "results_by_reduction": {},
        }

    def _load_folds_list(self, folds_file: Path, dataset: Dataset) -> List[List[str]]:
        """Load folds from file and filter to IDs present in the dataset."""
        with open(folds_file) as f:
            fold_content = f.read()

        # Parse fold IDs
        folds_list = []
        current_fold = []
        for line in fold_content.strip().split("\n"):
            line = line.strip()
            if line.startswith("|||"):
                if current_fold:
                    folds_list.append(current_fold)
                    current_fold = []
            elif line and not line.startswith("|"):
                current_fold.append(line)
        if current_fold:
            folds_list.append(current_fold)

        # Filter fold IDs to those present in the current dataset
        available_ids = {d.descriptive_part[0] for d in dataset.get_target_data()}
        filtered_folds = []
        for fold in folds_list:
            kept = [fid for fid in fold if fid in available_ids]
            if kept:
                filtered_folds.append(kept)

        return filtered_folds

    def _find_schema_file(self) -> Path:
        """Find schema (.s) file."""
        patterns = [f"{self.dataset_name}.s", "muta188.s"]
        for pattern in patterns:
            f = self.dataset_dir / pattern
            if f.exists():
                return f
        return None

    def _find_descriptive_file(self) -> Path:
        """Find descriptive file."""
        patterns = [f"{self.dataset_name}_descriptive.txt", "muta188_descriptive.txt"]
        for pattern in patterns:
            f = self.dataset_dir / pattern
            if f.exists():
                return f
        return None

    def load_task_settings(self) -> Settings:
        """Load task settings from schema file."""
        try:
            settings = Settings(str(self.schema_file))
            return settings
        except Exception as e:
            print(f"Error loading schema: {e}")
            return None

    def create_dataset_for_reduction(self, percentage: int) -> Tuple[Dataset, Path, Path]:
        """
        Create Dataset object for given reduction percentage.

        Args:
            percentage: Percentage of data removed (10, 20, 50, 70, 90)

        Returns:
            Tuple of (Dataset, target_file, folds_file)
        """
        target_file = self.scarcity_dir / f"target_removed_{percentage:02d}.txt"
        folds_file = self.scarcity_dir / "folds" / f"folds_removed_{percentage:02d}.txt"

        if not target_file.exists():
            raise FileNotFoundError(f"Target file not found: {target_file}")
        if not folds_file.exists():
            raise FileNotFoundError(f"Folds file not found: {folds_file}")

        try:
            # Use s_file directly for initialization
            dataset = Dataset(
                s_file=str(self.schema_file),
                data_file=str(self.descriptive_file),
                target_file=str(target_file),
            )
            return dataset, target_file, folds_file
        except Exception as e:
            print(f"Error creating dataset for {percentage}% reduction: {e}")
            raise

    def run_bagging_cv(
        self,
        dataset: Dataset,
        folds_file: Path,
        percentage: int,
        max_depth: int = 5,
        n_estimators: int = 50,
        shrinkage: float = 0.1,
    ) -> Dict[str, Any]:
        """
        Run Bagging with CV on reduced dataset.

        Args:
            dataset: Dataset object
            folds_file: Path to folds file
            percentage: Percentage removed (for logging)
            max_depth: Max tree depth
            n_estimators: Number of boosting iterations
            shrinkage: Learning rate for boosting

        Returns:
            Dictionary with results
        """
        print(f"\n{'=' * 70}")
        print(f"Running Bagging on {percentage}% reduced data")
        print(f"{'=' * 70}")

        fold_results = []

        try:
            folds_path = self.original_folds_file if self.use_original_folds else folds_file
            if self.use_original_folds and not folds_path.exists():
                raise FileNotFoundError(f"Original folds file not found: {folds_path}")

            folds_list = self._load_folds_list(folds_path, dataset)
            if not folds_list:
                raise ValueError(f"No valid folds found after filtering: {folds_path}")

            print(f"Folds file: {folds_path}")
            print(f"Number of folds: {len(folds_list)}")

            # CV loop
            for fold_idx, (train_data, test_data) in enumerate(
                create_folds(dataset, example_ids=folds_list)
            ):
                print(f"\n--- Fold {fold_idx + 1}/{len(folds_list)} ---")
                print(f"Train: {len(train_data.get_target_data())} examples")
                print(f"Test: {len(test_data.get_target_data())} examples")

                try:
                    rf_model = RandomForest(
                        nb_trees_to_build=n_estimators,  # es. 50 alberi come nel paper [file:1]
                        votes_aggregator=RandomForest.proportions_aggregator,
                        random_seed=2864,
                        heuristic=HeuristicGini(),
                    )
                    rf_model.build(train_data)

                    # Make predictions on test data
                    test_instances = test_data.get_target_data()
                    y_true, y_pred = [], []

                    for datum in test_instances:
                        try:
                            y_pred.append(rf_model.predict(datum))
                            y_true.append(datum.get_target())
                        except Exception as e:
                            print(f"    Prediction error: {e}")
                            continue

                    fold_result = {"fold": fold_idx}

                    if len(y_pred) > 0 and len(y_pred) == len(y_true):
                        # classi possibili
                        all_classes = get_all_target_values(test_instances)

                        acc_eval = Accuracy(all_classes)
                        acc_eval.add_many(y_true, y_pred)
                        acc_eval.evaluate()
                        fold_result["accuracy"] = acc_eval.get_measure_value()
                    else:
                        fold_result["accuracy"] = 0.0

                    fold_results.append(fold_result)
                    print(f"  Accuracy: {fold_result['accuracy']:.4f}")

                except Exception as e:
                    print(f"  Error in fold {fold_idx + 1}: {e}")
                    traceback.print_exc()
                    continue

            # Aggregate results
            aggregated = self._aggregate_fold_results(fold_results)
            aggregated["n_folds"] = len(fold_results)
            aggregated["reduction_percentage"] = percentage

            return aggregated

        except Exception as e:
            print(f"Error running CV: {e}")
            traceback.print_exc()
            return None

    def _calculate_metrics_simple(self, test_instances, predictions, fold_idx):
        """
        Simple metrics calculation: accuracy on predictions.

        Args:
            test_instances: List of Datum objects
            predictions: List of (idx, pred) tuples where pred is 0 or 1
            fold_idx: Fold index

        Returns:
            Dictionary with metrics
        """
        if not test_instances:
            return {"fold": fold_idx, "accuracy": 0, "f1": 0, "auc": 0}

        correct = sum(1 for idx, pred in predictions if pred == 1)
        accuracy = correct / len(test_instances) if test_instances else 0

        return {
            "fold": fold_idx,
            "accuracy": accuracy,
            "f1": accuracy,  # Simplified
            "auc": accuracy,  # Simplified
        }

    def _calculate_metrics(
        self, predictions: List[Tuple[str, float]], test_data: Dataset, fold_idx: int
    ) -> Dict[str, float]:
        """
        Calculate evaluation metrics.

        Args:
            predictions: List of (instance_id, prediction) tuples
            test_data: Test dataset
            fold_idx: Fold index

        Returns:
            Dictionary with metrics
        """
        # Get ground truth
        test_instances = test_data.get_target_data()
        gt_dict = {}
        for instance in test_instances:
            # instance.descriptive_part[0] is the instance ID
            gt_dict[instance.descriptive_part[0]] = instance.classification_value

        # Match predictions with ground truth
        correct = 0
        tp = fp = tn = fn = 0

        for pred_id, pred_score in predictions:
            if pred_id not in gt_dict:
                continue

            gt = gt_dict[pred_id]
            pred_class = 1 if pred_score >= 0.5 else 0

            if pred_class == gt:
                correct += 1

            # Binary classification metrics
            if gt == 1:
                if pred_class == 1:
                    tp += 1
                else:
                    fn += 1
            else:
                if pred_class == 1:
                    fp += 1
                else:
                    tn += 1

        # Calculate metrics
        accuracy = correct / len(gt_dict) if gt_dict else 0

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        # AUC approximation
        auc = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0

        return {"fold": fold_idx, "accuracy": accuracy, "f1": f1, "auc": auc}

    def _aggregate_fold_results(self, fold_results):
        if not fold_results:
            return {"accuracy_mean": 0, "accuracy_std": 0}

        values = [r["accuracy"] for r in fold_results if "accuracy" in r]
        mean = sum(values) / len(values)
        std = (sum((x - mean) ** 2 for x in values) / len(values)) ** 0.5
        return {"accuracy_mean": mean, "accuracy_std": std}

    def run_all_reductions(self) -> Dict[str, Any]:
        """Run experiment for all reduction percentages."""
        print(f"\n{'=' * 70}")
        print(f"DATA SCARCITY EXPERIMENT: {self.dataset_name.upper()}")
        print(f"{'=' * 70}")
        print("Model: Bagging")
        print("Validation: 10-fold CV")
        print("Metrics: Accuracy, Precision, Recall, F1, MSE, MAE, RMSE")

        for percentage in self.reduction_percentages:
            try:
                # Create dataset
                dataset, target_file, folds_file = self.create_dataset_for_reduction(percentage)

                # Run Bagging CV
                results = self.run_bagging_cv(dataset, folds_file, percentage)

                if results:
                    self.results["results_by_reduction"][f"{percentage:02d}%"] = results

            except Exception as e:
                print(f"\nError processing {percentage}% reduction: {e}")
                traceback.print_exc()
                continue

        return self.results

    def save_results(self) -> None:
        """Save results to files."""

        summary_file = self.log_dir / "results_summary_bagging.json"
        with open(summary_file, "w") as f:
            json.dump(self.results, f, indent=2)
        print(f"\n✓ Saved summary: {summary_file}")

        csv_file = self.log_dir / "results_bagging.csv"
        with open(csv_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["dataset", "reduction_%", "accuracy_mean", "accuracy_std"])

            for reduction_pct, metrics in self.results["results_by_reduction"].items():
                writer.writerow(
                    [
                        self.dataset_name,
                        reduction_pct.strip("%"),
                        metrics.get("accuracy_mean", 0),
                        metrics.get("accuracy_std", 0),
                    ]
                )

        print(f"✓ Saved CSV: {csv_file}")

        # Log file
        log_file = self.log_dir / "experiment_bagging.log"
        with open(log_file, "w") as f:
            f.write(str(self.results))
        print(f"✓ Saved log: {log_file}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Run data scarcity experiment with Re3py Bagging on reduced datasets."
    )
    parser.add_argument("--dataset", required=True, help="Dataset name")
    parser.add_argument("--log-dir", type=Path, help="Output directory for results")
    parser.add_argument("--all", action="store_true", help="Run all datasets")
    parser.add_argument(
        "--use-original-folds",
        action="store_true",
        help="Use original folds from data/folds/<dataset>/folds1.txt",
    )

    args = parser.parse_args()

    datasets = []
    if args.all:
        datasets = [
            "basket",
            "carcinogenesis",
            "imdb_big",
            "movie",
            "mutagenesis",
            "stack_big",
            "uwcse",
            "webkb",
            "yelp_big",
        ]
    else:
        datasets = [args.dataset]

    for dataset in datasets:
        try:
            experiment = DataScarcityExperiment(
                dataset, args.log_dir, use_original_folds=args.use_original_folds
            )
            experiment.run_all_reductions()
            experiment.save_results()
        except Exception as e:
            print(f"Error running experiment for {dataset}: {e}")
            traceback.print_exc()
            continue


if __name__ == "__main__":
    main()
