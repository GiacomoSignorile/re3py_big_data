"""
Re3py Data Scarcity Experiment - Single Tree (Decision Tree)
==============================================================

Main experiment runner for evaluating Re3py Single Decision Tree on reduced datasets.

Configuration:
- Model: Single Decision Tree (DecisionTree)
- Validation: 10-fold Cross-Validation
- Metrics:
  - Classification: Accuracy, Precision, Recall, F1
- Data Reductions: 0%, 10%, 20%, 50%, 70%, 90%

Outputs:
- results_summary.json: Aggregated metrics across folds
- results.csv: Tabular format for analysis

This variant uses a single decision tree to understand baseline performance
without ensemble effects.
"""

import argparse
import csv
import io
import json
import logging
import random
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from joblib import Parallel, delayed

# Add repo to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from re3py.data.data_and_statistics import Dataset, get_all_target_values
from re3py.data.task_settings import Settings
from re3py.eval.evaluation import Accuracy, Precision, Recall, F1
from re3py.learners.core.heuristic import HeuristicGini
from re3py.learners.tree import DecisionTree
from re3py.utilities.cross_validation import create_folds


class DataScarcityExperiment:
    """Runs data scarcity experiment with Re3py Single Tree."""

    def __init__(
        self,
        dataset_name: str,
        log_dir: Optional[Path] = None,
        use_original_folds: bool = False,
        config_name: str = "agg_all",
        n_jobs: int = 2,
    ):
        """
        Initialize experiment.

        Args:
            dataset_name: Name of dataset (e.g., 'basket', 'carcinogenesis')
            log_dir: Directory for logging results
            use_original_folds: Use original folds from data/folds
            config_name: Configuration name for file suffix (e.g., 'agg_all', 'exist_only')
            n_jobs: Number of parallel workers for fold processing (default: 2 for large datasets)
        """
        self.dataset_name = dataset_name
        self.config_name = config_name
        self.n_jobs = n_jobs
        script_dir = Path(__file__).resolve().parent  # experiment directory
        self.base_dir = script_dir.parent  # project root
        self.dataset_dir = self.base_dir / "data" / "datasets" / dataset_name
        self.scarcity_dir = (
            self.base_dir / "data" / "data_reduced" / dataset_name
        )  # centralized data location
        self.use_original_folds = use_original_folds
        self.original_folds_file = self.base_dir / "data" / "folds" / dataset_name / "folds1.txt"
        self._original_dataset: Optional[Dataset] = None

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
        
        # Setup logging
        self._setup_logger()

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

        # Grid search parameters (Section 5.3, page 11)
        self.leaf_size_values = [1, 5, 10, 15, 20]
        self.impurity_values = [0.0, 0.01, 0.02, 0.05, 0.1, 0.2]
        self.inner_cv_folds = 3  # Inner 3-fold CV for hyperparameter tuning

        # Results storage
        self.results = {
            "metadata": {
                "dataset": dataset_name,
                "model": "SingleTree",
                "validation": "10-fold CV",
                "metrics": ["accuracy", "precision", "recall", "f1"],
                "timestamp": datetime.now().isoformat(),
            },
            "results_by_reduction": {},
        }
    
    def _setup_logger(self):
        """Setup logger to write to file and console."""
        self.logger = logging.getLogger(f"experiment_{self.dataset_name}_st")
        self.logger.setLevel(logging.DEBUG)
        
        # Create logs directory
        logs_dir = self.base_dir / "logs"
        logs_dir.mkdir(parents=True, exist_ok=True)
        
        # File handler
        log_file = logs_dir / f"single_tree_{self.dataset_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        
        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        # Add handlers
        if not self.logger.handlers:
            self.logger.addHandler(file_handler)
            self.logger.addHandler(console_handler)

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
            filtered_folds.append(kept)

        if not any(filtered_folds):
            return []

        return filtered_folds

    def _build_dataset(self, target_file: Path) -> Dataset:
        """Build a Dataset object from the given target file."""
        return Dataset(
            s_file=str(self.schema_file),
            data_file=str(self.descriptive_file),
            target_file=str(target_file),
        )

    def _get_original_dataset(self) -> Dataset:
        """Load and cache the original (full) dataset."""
        if self._original_dataset is None:
            target_file = self._find_original_target_file()
            self._original_dataset = self._build_dataset(target_file)
        return self._original_dataset

    def _get_full_classes(self) -> List[str]:
        """Return all target classes from the original dataset."""
        original_dataset = self._get_original_dataset()
        return sorted(list(get_all_target_values(original_dataset.get_target_data())))

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
    
    def _find_original_target_file(self) -> Path:
        """Find original target file."""
        patterns = [f"{self.dataset_name}_target.txt", "muta188_target.txt"]
        for pattern in patterns:
            f = self.dataset_dir / pattern
            if f.exists():
                return f
        raise FileNotFoundError(f"No original target file found for {self.dataset_name}")
    
    def _find_original_folds_file(self) -> Path:
        """Find original folds file."""
        f = self.base_dir / "data" / "folds" / self.dataset_name / "folds1.txt"
        if f.exists():
            return f
        raise FileNotFoundError(f"No original folds file found for {self.dataset_name}")

    def load_task_settings(self) -> Settings:
        """Load task settings from schema file."""
        try:
            settings = Settings(str(self.schema_file))
            return settings
        except Exception as e:
            print(f"Error loading schema: {e}")
            return None
    
    def _get_only_existential_flag(self) -> bool:
        """Determine only_existential flag based on config_name."""
        return "exist" in self.config_name.lower()

    def create_dataset_for_reduction(self, percentage: int) -> Tuple[Dataset, Path, Path]:
        """
        Create Dataset object for given reduction percentage.

        Args:
            percentage: Percentage of data removed (10, 20, 50, 70, 90)

        Returns:
            Tuple of (Dataset, target_file, folds_file)
        """
        if percentage == 0:
            target_file = self._find_original_target_file()
            folds_file = self._find_original_folds_file()
        else:
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

    def _tune_tree_parameters(
        self,
        train_data: Dataset,
        settings: Settings,
        fold_idx: int,
    ) -> Tuple[int, float, float]:
        """
        Tune tree parameters using inner 3-fold CV (Section 5.3, page 11).

        For each combination of (leaf_size, impurity), trains a tree and evaluates
        on validation fold. Returns the parameters with best average accuracy.

        Args:
            train_data: Training dataset for this fold
            settings: Task settings
            fold_idx: Outer fold index (for logging)

        Returns:
            Tuple of (best_leaf_size, best_impurity, best_inner_cv_score)
        """
        print(f"    Tuning parameters with inner 3-fold CV...")
        self.logger.debug(f"Fold {fold_idx + 1}: Starting inner 3-fold CV for parameter tuning")

        best_score = -1.0
        best_leaf_size = 1
        best_impurity = 0.0
        results_log = []

        try:
            inner_folds_list = self._load_or_create_inner_folds(train_data)

            # Grid search over all parameter combinations
            for leaf_size in self.leaf_size_values:
                for impurity in self.impurity_values:
                    fold_scores = []

                    # Inner CV
                    for inner_fold_idx, (inner_train, inner_val) in enumerate(
                        create_folds(train_data, example_ids=inner_folds_list)
                    ):
                        try:
                            tree_params = {
                                'allowed_atom_tests': settings.get_atom_tests_structured(),
                                'allowed_aggregators': settings.get_aggregates(),
                                'minimal_examples_in_leaf': leaf_size,
                                'minimal_impurity': impurity,
                                'java_port': None,
                                "per_class_bootstrap": True,
                                "only_existential": self._get_only_existential_flag(),
                                "longest_atom_test_chain": 2,
                                "heuristic": HeuristicGini(),
                            }

                            inner_tree = DecisionTree(
                                random_seed=2864 + inner_fold_idx,
                                **tree_params
                            )

                            # Build tree
                            captured_output = io.StringIO()
                            original_stdout = sys.stdout
                            try:
                                sys.stdout = captured_output
                                inner_tree.build(inner_train)
                            finally:
                                sys.stdout = original_stdout
                                captured_output.close()

                            # Evaluate on validation fold
                            val_classes = sorted(list(get_all_target_values(inner_val.get_target_data())))
                            y_true, y_pred = [], []

                            for datum in inner_val.get_target_data():
                                try:
                                    y_pred.append(inner_tree.predict(datum))
                                    y_true.append(datum.get_target())
                                except:
                                    pass

                            if len(y_pred) > 0 and len(y_pred) == len(y_true):
                                acc_eval = Accuracy(val_classes)
                                acc_eval.add_many(y_true, y_pred)
                                acc_eval.evaluate()
                                fold_scores.append(acc_eval.get_measure_value())

                        except Exception as e:
                            self.logger.debug(
                                f"Inner fold {inner_fold_idx + 1} failed for leaf_size={leaf_size}, impurity={impurity}: {e}"
                            )
                            fold_scores.append(0.0)

                    # Average score for this parameter combination
                    if fold_scores:
                        avg_score = sum(fold_scores) / len(fold_scores)
                        results_log.append(
                            {
                                "leaf_size": leaf_size,
                                "impurity": impurity,
                                "avg_score": avg_score,
                            }
                        )

                        if avg_score > best_score:
                            best_score = avg_score
                            best_leaf_size = leaf_size
                            best_impurity = impurity

        except Exception as e:
            self.logger.error(
                f"Error during parameter tuning for fold {fold_idx + 1}: {e}",
                exc_info=True,
            )
            print(f"    Warning: Parameter tuning failed, using defaults (leaf_size=1, impurity=0)")

        self.logger.debug(
            f"Fold {fold_idx + 1}: Inner CV results - {results_log}. Best: leaf_size={best_leaf_size}, impurity={best_impurity}, score={best_score:.4f}"
        )

        return best_leaf_size, best_impurity, best_score

    def _load_or_create_inner_folds(self, data: Dataset) -> List[List[str]]:
        """
        Create 3-fold CV splits for inner parameter tuning.

        Args:
            data: Dataset to split

        Returns:
            List of 3 folds, each containing example IDs
        """
        target_data = data.get_target_data()
        example_ids = [d.descriptive_part[0] for d in target_data]

        # Shuffle and split into 3 folds
        random.shuffle(example_ids)
        fold_size = len(example_ids) // self.inner_cv_folds
        inner_folds = []

        for i in range(self.inner_cv_folds):
            start_idx = i * fold_size
            end_idx = (i + 1) * fold_size if i < self.inner_cv_folds - 1 else len(example_ids)
            inner_folds.append(example_ids[start_idx:end_idx])

        return inner_folds

    def _run_single_fold(
        self,
        fold_idx: int,
        train_data: Dataset,
        test_data: Dataset,
        full_classes: List[str],
        n_folds: int,
    ) -> Tuple[Dict[str, Any], int]:
        """
        Run a single fold of single tree training.

        Args:
            fold_idx: Fold index
            train_data: Training dataset
            test_data: Test dataset
            full_classes: List of all possible classes
            n_folds: Total number of folds

        Returns:
            Tuple of (fold_result dict, test_size)
        """
        test_instances = test_data.get_target_data()

        try:
            settings = self.load_task_settings()
            
            # Tune hyperparameters using inner 3-fold CV (Section 5.3, page 11)
            best_leaf_size, best_impurity, inner_cv_score = self._tune_tree_parameters(
                train_data, settings, fold_idx
            )
            
            # Build final tree with best parameters
            tree_params = {
                'allowed_atom_tests': settings.get_atom_tests_structured(),
                'allowed_aggregators': settings.get_aggregates(),
                'minimal_examples_in_leaf': best_leaf_size,
                'minimal_impurity': best_impurity,
                'java_port': None,
                "per_class_bootstrap": True,
                "only_existential": self._get_only_existential_flag(),
                "longest_atom_test_chain": 2,  # Look-ahead depth = 2 (Section 5.3, page 11)
                "heuristic": HeuristicGini()  # GINI index as split criterion (Section 4.2, page 7)
            }

            # Single Decision Tree with tuned parameters
            single_tree = DecisionTree(
                random_seed=2864,
                **tree_params
            )
            
            # Capture tree building output and log it
            self.logger.debug(f"Starting SingleTree build for fold {fold_idx + 1}/{n_folds}")
            captured_output = io.StringIO()
            original_stdout = sys.stdout
            try:
                sys.stdout = captured_output
                single_tree.build(train_data)
            finally:
                sys.stdout = original_stdout
                build_output = captured_output.getvalue()
                if build_output:
                    self.logger.debug(f"Tree building output for fold {fold_idx + 1}:\n{build_output}")
                captured_output.close()

            # Make predictions on test data
            y_true, y_pred = [], []

            for datum in test_instances:
                try:
                    y_pred.append(single_tree.predict(datum))
                    y_true.append(datum.get_target())
                except Exception as e:
                    self.logger.debug(f"Prediction error in fold {fold_idx + 1}: {e}")
                    continue

            fold_result = {"fold": fold_idx, "test_instances": len(test_instances)}

            if len(y_pred) > 0 and len(y_pred) == len(y_true):
                acc_eval = Accuracy(full_classes)
                acc_eval.add_many(y_true, y_pred)
                acc_eval.evaluate()
                fold_result["accuracy"] = acc_eval.get_measure_value()
                
                # Calculate precision, recall, f1
                try:
                    if len(full_classes) == 2:
                        positive_class = full_classes[0]
                        prec_eval = Precision(full_classes, positive_class=positive_class)
                        prec_eval.add_many(y_true, y_pred)
                        prec_eval.evaluate()
                        fold_result["precision"] = prec_eval.get_measure_value()
                        
                        rec_eval = Recall(full_classes, positive_class=positive_class)
                        rec_eval.add_many(y_true, y_pred)
                        rec_eval.evaluate()
                        fold_result["recall"] = rec_eval.get_measure_value()
                        
                        f1_eval = F1(full_classes, positive_class=positive_class)
                        f1_eval.add_many(y_true, y_pred)
                        f1_eval.evaluate()
                        fold_result["f1"] = f1_eval.get_measure_value()
                    else:
                        fold_result["precision"] = fold_result["accuracy"]
                        fold_result["recall"] = fold_result["accuracy"]
                        fold_result["f1"] = fold_result["accuracy"]
                except (KeyError, ValueError, ZeroDivisionError):
                    fold_result["precision"] = fold_result["accuracy"]
                    fold_result["recall"] = fold_result["accuracy"]
                    fold_result["f1"] = fold_result["accuracy"]
            else:
                fold_result["accuracy"] = 0.0
                fold_result["precision"] = 0.0
                fold_result["recall"] = 0.0
                fold_result["f1"] = 0.0

            self.logger.info(f"Fold {fold_idx + 1}/{n_folds} completed - Accuracy: {fold_result['accuracy']:.4f}, Precision: {fold_result['precision']:.4f}, Recall: {fold_result['recall']:.4f}, F1: {fold_result['f1']:.4f}")
            return fold_result, len(test_instances)

        except Exception as e:
            self.logger.error(f"Error in fold {fold_idx + 1}: {e}", exc_info=True)
            fold_result = {
                "fold": fold_idx,
                "test_instances": len(test_instances),
                "accuracy": 0.0,
                "precision": 0.0,
                "recall": 0.0,
                "f1": 0.0,
            }
            return fold_result, len(test_instances)

    def run_single_tree_cv(
        self,
        dataset: Dataset,
        folds_file: Path,
        percentage: int,
    ) -> Dict[str, Any]:
        """
        Run Single Decision Tree with CV on reduced dataset.

        Args:
            dataset: Dataset object
            folds_file: Path to folds file
            percentage: Percentage removed (for logging)

        Returns:
            Dictionary with results
        """
        print(f"\nRunning Single Tree on {percentage}% reduced data ({len(dataset.get_target_data())} examples)")

        fold_results = []
        test_sizes = []
        self.logger.info(f"Starting Single Tree on {percentage}% reduced data")

        try:
            folds_path = self.original_folds_file if self.use_original_folds else folds_file
            if self.use_original_folds and not folds_path.exists():
                raise FileNotFoundError(f"Original folds file not found: {folds_path}")

            folds_list = self._load_folds_list(folds_path, dataset)
            if not folds_list:
                raise ValueError(f"No valid folds found after filtering: {folds_path}")

            full_classes = self._get_full_classes()

            print(f"Folds file: {folds_path}")
            print(f"Number of folds: {len(folds_list)}")

            # Collect all folds first for parallel processing
            all_folds = list(
                enumerate(create_folds(dataset, example_ids=folds_list))
            )

            # Run folds in parallel using joblib
            self.logger.info(f"Running {len(all_folds)} folds in parallel with {self.n_jobs} workers")
            fold_results_with_sizes = Parallel(n_jobs=self.n_jobs, verbose=10)(
                delayed(self._run_single_fold)(
                    fold_idx=fold_idx,
                    train_data=train_data,
                    test_data=test_data,
                    full_classes=full_classes,
                    n_folds=len(all_folds),
                )
                for fold_idx, (train_data, test_data) in all_folds
            )

            # Unpack results
            fold_results = []
            test_sizes = []
            for fold_result, test_size in fold_results_with_sizes:
                fold_results.append(fold_result)
                test_sizes.append(test_size)
                print(
                    f"  Fold {fold_result['fold'] + 1} - Accuracy: {fold_result['accuracy']:.4f} | "
                    f"Precision: {fold_result['precision']:.4f} | Recall: {fold_result['recall']:.4f} | "
                    f"F1: {fold_result['f1']:.4f}"
                )

            # Aggregate results
            aggregated = self._aggregate_fold_results(fold_results)
            aggregated["n_folds"] = len(fold_results)
            aggregated["reduction_percentage"] = percentage
            aggregated["avg_test_instances"] = sum(test_sizes) / len(test_sizes) if test_sizes else 0
            aggregated["total_test_instances"] = sum(test_sizes)

            return aggregated

        except Exception as e:
            print(f"Error running CV: {e}")
            self.logger.error(f"Error running CV: {e}", exc_info=True)
            traceback.print_exc()
            return None

    def _aggregate_fold_results(self, fold_results):
        if not fold_results:
            return {
                "accuracy_mean": 0, "accuracy_std": 0,
                "precision_mean": 0, "precision_std": 0,
                "recall_mean": 0, "recall_std": 0,
                "f1_mean": 0, "f1_std": 0,
            }

        metrics = ["accuracy", "precision", "recall", "f1"]
        result = {}
        
        for metric in metrics:
            values = [r[metric] for r in fold_results if metric in r]
            if values:
                mean = sum(values) / len(values)
                std = (sum((x - mean) ** 2 for x in values) / len(values)) ** 0.5
                result[f"{metric}_mean"] = mean
                result[f"{metric}_std"] = std
            else:
                result[f"{metric}_mean"] = 0
                result[f"{metric}_std"] = 0
        
        return result

    def _load_checkpoint(self) -> bool:
        """Load existing results from checkpoint files if available."""
        summary_file = self.log_dir / f"{self.dataset_name}_summary_single_tree_{self.config_name}.json"
        
        if summary_file.exists():
            try:
                with open(summary_file, "r") as f:
                    saved_results = json.load(f)
                self.results = saved_results
                print(f"\n✓ Loaded checkpoint from: {summary_file}")
                self.logger.info(f"Loaded checkpoint from: {summary_file}")
                print(f"  Resuming from {len(self.results.get('results_by_reduction', {}))} completed reductions")
                return True
            except Exception as e:
                print(f"Warning: Could not load checkpoint: {e}")
                self.logger.warning(f"Could not load checkpoint: {e}")
                return False
        return False

    def _reduction_already_completed(self, percentage: int) -> bool:
        """Check if a reduction percentage has already been completed."""
        reduction_key = f"{percentage:02d}%"
        return reduction_key in self.results.get("results_by_reduction", {})

    def _save_checkpoint(self) -> None:
        """Save intermediate results after each reduction."""
        summary_file = self.log_dir / f"{self.dataset_name}_summary_single_tree_{self.config_name}.json"
        with open(summary_file, "w") as f:
            json.dump(self.results, f, indent=2)
        
        csv_file = self.log_dir / f"{self.dataset_name}_results_single_tree_{self.config_name}.csv"
        with open(csv_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "dataset", "reduction_%", "avg_test_instances",
                "accuracy_mean", "accuracy_std",
                "precision_mean", "precision_std",
                "recall_mean", "recall_std",
                "f1_mean", "f1_std"
            ])

            for reduction_pct, metrics in self.results["results_by_reduction"].items():
                writer.writerow([
                    self.dataset_name,
                    reduction_pct.strip("%"),
                    int(metrics.get("avg_test_instances", 0)),
                    metrics.get("accuracy_mean", 0),
                    metrics.get("accuracy_std", 0),
                    metrics.get("precision_mean", 0),
                    metrics.get("precision_std", 0),
                    metrics.get("recall_mean", 0),
                    metrics.get("recall_std", 0),
                    metrics.get("f1_mean", 0),
                    metrics.get("f1_std", 0),
                ])

    def run_all_reductions(self) -> Dict[str, Any]:
        """Run experiment for all reduction percentages."""
        print(f"\n{'=' * 70}")
        print(f"DATA SCARCITY EXPERIMENT: {self.dataset_name.upper()}")
        print(f"{'=' * 70}")
        print("Model: Single Decision Tree")
        print("Validation: 10-fold CV")
        print("Metrics: Accuracy, Precision, Recall, F1")

        # Try to load checkpoint
        checkpoint_loaded = self._load_checkpoint()
        if checkpoint_loaded:
            print(f"Resuming experiment from checkpoint...")
        else:
            print(f"Starting new experiment...")

        for percentage in self.reduction_percentages:
            # Skip if already completed
            if self._reduction_already_completed(percentage):
                print(f"\n⊘ Skipping {percentage}% reduction (already completed)")
                self.logger.info(f"Skipping {percentage}% reduction (already completed)")
                continue

            try:
                # Create dataset
                dataset, target_file, folds_file = self.create_dataset_for_reduction(percentage)

                # Run Single Tree CV
                results = self.run_single_tree_cv(dataset, folds_file, percentage)

                if results:
                    self.results["results_by_reduction"][f"{percentage:02d}%"] = results
                    # Save checkpoint immediately after this reduction completes
                    self._save_checkpoint()
                    print(f"✓ Checkpoint saved after {percentage}% reduction")
                    self.logger.info(f"Checkpoint saved after {percentage}% reduction")

            except Exception as e:
                print(f"\nError processing {percentage}% reduction: {e}")
                self.logger.error(f"Error processing {percentage}% reduction: {e}", exc_info=True)
                traceback.print_exc()
                continue

        return self.results

    def save_results(self) -> None:
        """Save final results and detailed breakdown."""

        # Summary JSON file
        summary_file = self.log_dir / f"{self.dataset_name}_summary_single_tree_{self.config_name}.json"
        with open(summary_file, "w") as f:
            json.dump(self.results, f, indent=2)
        print(f"\n✓ Saved summary: {summary_file}")
        self.logger.info(f"Saved summary: {summary_file}")

        # Detailed results JSON
        detailed_file = self.log_dir / f"{self.dataset_name}_detailed_single_tree_{self.config_name}.json"
        detailed_results = {
            "metadata": self.results["metadata"],
            "detailed_results_by_reduction": self.results["results_by_reduction"]
        }
        with open(detailed_file, "w") as f:
            json.dump(detailed_results, f, indent=2)
        print(f"✓ Saved detailed results: {detailed_file}")
        self.logger.info(f"Saved detailed results: {detailed_file}")

        # CSV results file
        csv_file = self.log_dir / f"{self.dataset_name}_results_single_tree_{self.config_name}.csv"
        with open(csv_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "dataset",
                    "reduction_%",
                    "avg_test_instances",
                    "accuracy_mean",
                    "accuracy_std",
                    "precision_mean",
                    "precision_std",
                    "recall_mean",
                    "recall_std",
                    "f1_mean",
                    "f1_std",
                ]
            )

            for reduction_pct, metrics in self.results["results_by_reduction"].items():
                writer.writerow(
                    [
                        self.dataset_name,
                        reduction_pct.strip("%"),
                        int(metrics.get("avg_test_instances", 0)),
                        metrics.get("accuracy_mean", 0),
                        metrics.get("accuracy_std", 0),
                        metrics.get("precision_mean", 0),
                        metrics.get("precision_std", 0),
                        metrics.get("recall_mean", 0),
                        metrics.get("recall_std", 0),
                        metrics.get("f1_mean", 0),
                        metrics.get("f1_std", 0),
                    ]
                )
        print(f"✓ Saved CSV results: {csv_file}")
        self.logger.info(f"Saved CSV results: {csv_file}")

        print(f"\n{'=' * 70}")
        print(f"Experiment results saved in: {self.log_dir}")
        print("Files created:")
        print(f"  - {summary_file.name}")
        print(f"  - {csv_file.name}")
        print(f"  - {detailed_file.name}")
        print(f"{'=' * 70}")
        self.logger.info(f"Experiment completed successfully")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Run data scarcity experiment with Re3py Single Tree on reduced datasets."
    )
    parser.add_argument("--dataset", required=True, help="Dataset name")
    parser.add_argument("--log-dir", type=Path, help="Output directory for results")
    parser.add_argument("--all", action="store_true", help="Run all datasets")
    parser.add_argument(
        "--use-original-folds",
        action="store_true",
        help="Use original folds from data/folds/<dataset>/folds1.txt",
    )
    parser.add_argument(
        "--config-name",
        type=str,
        default="agg_all",
        help="Configuration name for file suffix (e.g., 'agg_all', 'exist_only')",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=2,
        help="Number of parallel workers for fold processing (default: 2)",
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
                dataset, args.log_dir, use_original_folds=args.use_original_folds, config_name=args.config_name, n_jobs=args.n_jobs
            )
            experiment.logger.info(f"Starting experiment for dataset: {dataset}")
            experiment.run_all_reductions()
            experiment.save_results()
            experiment.logger.info(f"Successfully completed experiment for dataset: {dataset}")
        except Exception as e:
            print(f"Error running experiment for {dataset}: {e}")
            logging.getLogger(f"experiment_{dataset}_st").error(f"Error running experiment: {e}", exc_info=True)
            traceback.print_exc()
            continue


if __name__ == "__main__":
    main()
