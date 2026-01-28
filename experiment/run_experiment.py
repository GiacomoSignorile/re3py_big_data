"""
Re3py Data Scarcity Experiment - Bagging AGG-All
=================================================

Main experiment runner for evaluating Re3py Bagging ensemble on reduced datasets.

Configuration:
- Model: Bagging with AGG-All (all aggregates active)
- Validation: 10-fold Cross-Validation
- Metric: Accuracy (primary), F1 and AUC (auxiliary)
- Data Reductions: 10%, 20%, 50%, 70%, 90% (incremental)

Outputs:
- results_summary.json: Aggregated metrics across folds
- results_detailed.json: Per-fold results
- results.csv: Tabular format for analysis
"""

import json
import csv
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Any
from datetime import datetime
import sys
import traceback

# Add repo to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Compatibility patch for time.clock() in Python 3.8+
import time_compat

# Import compatibility layer for old module paths
import import_compat

from re3py.utilities.cross_validation import create_folds
from re3py.data.data_and_statistics import Dataset
from re3py.data.task_settings import Settings
from re3py.learners.boosting import GradientBoosting


class DataScarcityExperiment:
    """Runs data scarcity experiment with Re3py."""
    
    def __init__(self, dataset_name: str, log_dir: Path = None):
        """
        Initialize experiment.
        
        Args:
            dataset_name: Name of dataset (e.g., 'basket', 'carcinogenesis')
            log_dir: Directory for logging results
        """
        self.dataset_name = dataset_name
        script_dir = Path(__file__).resolve().parent  # experiment directory
        self.base_dir = script_dir.parent  # project root
        self.dataset_dir = self.base_dir / "data" / "datasets" / dataset_name
        self.scarcity_dir = self.base_dir / "data" / "data_reduced" / dataset_name  # centralized data location
        
        # Ensure reduced data exists
        if not self.scarcity_dir.exists():
            raise FileNotFoundError(f"Reduced data not found. Run preprocessing first: {self.scarcity_dir}")
        
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
            'metadata': {
                'dataset': dataset_name,
                'model': 'Bagging (AGG-All)',
                'validation': '10-fold CV',
                'metrics': ['accuracy', 'f1', 'auc'],
                'timestamp': datetime.now().isoformat()
            },
            'results_by_reduction': {}
        }
    
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
        patterns = [
            f"{self.dataset_name}_descriptive.txt",
            "muta188_descriptive.txt"
        ]
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
                target_file=str(target_file)
            )
            return dataset, target_file, folds_file
        except Exception as e:
            print(f"Error creating dataset for {percentage}% reduction: {e}")
            raise
    
    def run_bagging_cv(self, dataset: Dataset, folds_file: Path,
                      percentage: int, max_depth: int = 5,
                      n_estimators: int = 10) -> Dict[str, Any]:
        """
        Run Bagging with CV on reduced dataset.
        
        Args:
            dataset: Dataset object
            folds_file: Path to folds file
            percentage: Percentage removed (for logging)
            max_depth: Max tree depth
            n_estimators: Number of bagging iterations
            
        Returns:
            Dictionary with results
        """
        print(f"\n{'='*70}")
        print(f"Running Bagging on {percentage}% reduced data")
        print(f"{'='*70}")
        
        fold_results = []
        fold_num = 0
        
        try:
            # Read folds
            with open(folds_file, 'r') as f:
                fold_content = f.read()
            
            # Parse fold IDs
            folds_list = []
            current_fold = []
            for line in fold_content.strip().split('\n'):
                line = line.strip()
                if line.startswith('|||'):
                    if current_fold:
                        folds_list.append(current_fold)
                        current_fold = []
                elif line and not line.startswith('|'):
                    current_fold.append(line)
            
            if current_fold:
                folds_list.append(current_fold)
            
            print(f"Number of folds: {len(folds_list)}")
            
            # CV loop
            for fold_idx, (train_data, test_data) in enumerate(
                create_folds(dataset, example_ids=folds_list)
            ):
                print(f"\n--- Fold {fold_idx + 1}/{len(folds_list)} ---")
                print(f"Train: {len(train_data.get_target_data())} examples")
                print(f"Test: {len(test_data.get_target_data())} examples")
                
                try:
                    # Train Bagging ensemble with AGG-All
                    bagging = BaggingEnsemble(
                        dataset=train_data,
                        max_depth=max_depth,
                        n_estimators=n_estimators,
                        use_all_aggregates=True  # AGG-All
                    )
                    
                    # Make predictions
                    predictions = bagging.predict(test_data.get_target_data())
                    
                    # Calculate metrics
                    fold_result = self._calculate_metrics(
                        predictions, test_data, fold_idx
                    )
                    fold_results.append(fold_result)
                    
                    print(f"  Accuracy: {fold_result['accuracy']:.4f}")
                    print(f"  F1: {fold_result['f1']:.4f}")
                    print(f"  AUC: {fold_result['auc']:.4f}")
                    
                except Exception as e:
                    print(f"  Error in fold {fold_idx + 1}: {e}")
                    traceback.print_exc()
                    continue
            
            # Aggregate results
            aggregated = self._aggregate_fold_results(fold_results)
            aggregated['n_folds'] = len(fold_results)
            aggregated['reduction_percentage'] = percentage
            
            return aggregated
            
        except Exception as e:
            print(f"Error running CV: {e}")
            traceback.print_exc()
            return None
    
    def _calculate_metrics(self, predictions: List[Tuple[str, float]],
                          test_data: Dataset, fold_idx: int) -> Dict[str, float]:
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
        
        return {
            'fold': fold_idx,
            'accuracy': accuracy,
            'f1': f1,
            'auc': auc
        }
    
    def _aggregate_fold_results(self, fold_results: List[Dict]) -> Dict[str, float]:
        """Aggregate metrics across folds."""
        if not fold_results:
            return {'accuracy': 0, 'f1': 0, 'auc': 0}
        
        metrics = ['accuracy', 'f1', 'auc']
        aggregated = {}
        
        for metric in metrics:
            values = [r[metric] for r in fold_results if metric in r]
            if values:
                aggregated[f"{metric}_mean"] = sum(values) / len(values)
                aggregated[f"{metric}_std"] = (
                    (sum((x - aggregated[f"{metric}_mean"]) ** 2 for x in values) 
                     / len(values)) ** 0.5
                )
            else:
                aggregated[f"{metric}_mean"] = 0
                aggregated[f"{metric}_std"] = 0
        
        return aggregated
    
    def run_all_reductions(self) -> Dict[str, Any]:
        """Run experiment for all reduction percentages."""
        print(f"\n{'='*70}")
        print(f"DATA SCARCITY EXPERIMENT: {self.dataset_name.upper()}")
        print(f"{'='*70}")
        print(f"Model: Bagging (AGG-All)")
        print(f"Validation: 10-fold CV")
        print(f"Metrics: Accuracy, F1, AUC")
        
        for percentage in self.reduction_percentages:
            try:
                # Create dataset
                dataset, target_file, folds_file = self.create_dataset_for_reduction(percentage)
                
                # Run Bagging CV
                results = self.run_bagging_cv(dataset, folds_file, percentage)
                
                if results:
                    self.results['results_by_reduction'][f"{percentage:02d}%"] = results
                
            except Exception as e:
                print(f"\nError processing {percentage}% reduction: {e}")
                traceback.print_exc()
                continue
        
        return self.results
    
    def save_results(self) -> None:
        """Save results to files."""
        # Summary JSON
        summary_file = self.log_dir / "results_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        print(f"\n✓ Saved summary: {summary_file}")
        
        # CSV for easy import
        csv_file = self.log_dir / "results.csv"
        with open(csv_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['dataset', 'reduction_%', 'accuracy_mean', 'accuracy_std',
                            'f1_mean', 'f1_std', 'auc_mean', 'auc_std'])
            
            for reduction_pct, metrics in self.results['results_by_reduction'].items():
                writer.writerow([
                    self.dataset_name,
                    reduction_pct.strip('%'),
                    metrics.get('accuracy_mean', 0),
                    metrics.get('accuracy_std', 0),
                    metrics.get('f1_mean', 0),
                    metrics.get('f1_std', 0),
                    metrics.get('auc_mean', 0),
                    metrics.get('auc_std', 0)
                ])
        
        print(f"✓ Saved CSV: {csv_file}")
        
        # Log file
        log_file = self.log_dir / "experiment.log"
        with open(log_file, 'w') as f:
            f.write(str(self.results))
        print(f"✓ Saved log: {log_file}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Run data scarcity experiment with Re3py Bagging"
    )
    parser.add_argument('--dataset', required=True,
                       help='Dataset name')
    parser.add_argument('--log-dir', type=Path,
                       help='Output directory for results')
    parser.add_argument('--all', action='store_true',
                       help='Run all datasets')
    
    args = parser.parse_args()
    
    datasets = []
    if args.all:
        datasets = ['basket', 'carcinogenesis', 'imdb_big', 'movie',
                   'mutagenesis', 'stack_big', 'uwcse', 'webkb', 'yelp_big']
    else:
        datasets = [args.dataset]
    
    for dataset in datasets:
        try:
            experiment = DataScarcityExperiment(dataset, args.log_dir)
            experiment.run_all_reductions()
            experiment.save_results()
        except Exception as e:
            print(f"Error running experiment for {dataset}: {e}")
            traceback.print_exc()
            continue


if __name__ == '__main__':
    main()
