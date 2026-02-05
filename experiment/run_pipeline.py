#!/usr/bin/env python3
"""
Data Scarcity Experiment - Full Pipeline Orchestrator
======================================================

Orchestrates the complete experimental pipeline:
1. Data preprocessing (incremental reduction)
2. Fold mapping to reduced datasets
3. Running Bagging AGG-All experiments
4. Results aggregation and visualization

Usage:
    python run_pipeline.py --all
    python run_pipeline.py --dataset basket
"""

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


class PipelineOrchestrator:
    """Orchestrates the full experimental pipeline."""

    def __init__(self, dataset_names: list, seed: int = 2864):
        """Initialize orchestrator."""
        self.dataset_names = dataset_names
        self.seed = seed
        self.base_dir = Path(__file__).parent
        self.logs_dir = self.base_dir.parent / "logs"
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self.log_file = self.logs_dir / f"pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        self.results_summary = {
            "timestamp": datetime.now().isoformat(),
            "datasets": dataset_names,
            "seed": seed,
            "stages": {},
        }

    def log(self, message: str, level: str = "INFO") -> None:
        """Log message to console and file."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_msg = f"[{timestamp}] [{level}] {message}"
        print(log_msg)

        with open(self.log_file, "a") as f:
            f.write(log_msg + "\n")

    def run_stage(self, stage_name: str, command: list) -> bool:
        """Run a pipeline stage."""
        self.log(f"\n{'=' * 70}")
        self.log(f"STAGE: {stage_name}")
        self.log(f"{'=' * 70}")

        try:
            result = subprocess.run(
                command, cwd=str(self.base_dir), capture_output=False, timeout=3600
            )

            if result.returncode == 0:
                self.log(f"✓ {stage_name} completed successfully")
                return True
            else:
                self.log(f"✗ {stage_name} failed with code {result.returncode}", "ERROR")
                return False

        except subprocess.TimeoutExpired:
            self.log(f"✗ {stage_name} timed out", "ERROR")
            return False
        except Exception as e:
            self.log(f"✗ {stage_name} error: {e}", "ERROR")
            return False

    def stage_1_preprocessing(self) -> bool:
        """Stage 1: Data preprocessing."""
        self.log("Starting Stage 1: Data Preprocessing")
        self.log(f"Processing {len(self.dataset_names)} datasets...")

        for dataset in self.dataset_names:
            self.log(f"\nPreprocessing: {dataset}")
            command = [
                sys.executable,
                str(self.base_dir / "data_scarcity_preprocessing.py"),
                "--dataset",
                dataset,
                "--seed",
                str(self.seed),
            ]

            if not self.run_stage(f"Preprocess {dataset}", command):
                self.log(f"Warning: Preprocessing failed for {dataset}", "WARN")
                return False

        self.results_summary["stages"]["preprocessing"] = "completed"
        return True

    def stage_2_fold_mapping(self) -> bool:
        """Stage 2: Fold mapping."""
        self.log("Starting Stage 2: Fold Mapping")

        for dataset in self.dataset_names:
            self.log(f"\nMapping folds: {dataset}")
            command = [sys.executable, str(self.base_dir / "fold_mapper.py"), "--dataset", dataset]

            if not self.run_stage(f"Map folds {dataset}", command):
                self.log(f"Warning: Fold mapping failed for {dataset}", "WARN")
                return False

        self.results_summary["stages"]["fold_mapping"] = "completed"
        return True

    def stage_3_experiments(self) -> bool:
        """Stage 3: Run experiments."""
        self.log("Starting Stage 3: Run Bagging AGG-All Experiments")

        for dataset in self.dataset_names:
            self.log(f"\nRunning experiment: {dataset}")
            command = [
                sys.executable,
                str(self.base_dir / "run_experiment.py"),
                "--dataset",
                dataset,
            ]

            if not self.run_stage(f"Experiment {dataset}", command):
                self.log(f"Warning: Experiment failed for {dataset}", "WARN")
                # Continue with other datasets

        self.results_summary["stages"]["experiments"] = "completed"
        return True

    def stage_4_aggregation(self) -> bool:
        """Stage 4: Aggregate results."""
        self.log("Starting Stage 4: Results Aggregation")

        try:
            results_dir = self.base_dir / "results"

            # Aggregate all CSV files
            all_results = []

            for dataset_dir in results_dir.glob("*/"):
                csv_file = dataset_dir / "results.csv"
                if csv_file.exists():
                    with open(csv_file) as f:
                        lines = f.readlines()
                        if len(lines) > 1:
                            all_results.extend(lines[1:])  # Skip header

            # Write combined CSV
            combined_csv = results_dir / "combined_results.csv"
            with open(combined_csv, "w") as f:
                # Write header
                f.write(
                    "dataset,reduction_%,accuracy_mean,accuracy_std,f1_mean,f1_std,auc_mean,auc_std\n"
                )
                f.writelines(all_results)

            self.log(f"✓ Combined results saved to {combined_csv}")
            self.results_summary["stages"]["aggregation"] = "completed"
            return True

        except Exception as e:
            self.log(f"Error during aggregation: {e}", "ERROR")
            self.results_summary["stages"]["aggregation"] = "failed"
            return False

    def run_full_pipeline(self) -> bool:
        """Run complete pipeline."""
        self.log("=" * 70)
        self.log("DATA SCARCITY EXPERIMENT - FULL PIPELINE")
        self.log("=" * 70)
        self.log(f"Datasets: {', '.join(self.dataset_names)}")
        self.log(f"Seed: {self.seed}")
        self.log(f"Log file: {self.log_file}")

        start_time = time.time()

        stages = [
            ("Preprocessing", self.stage_1_preprocessing),
            ("Fold Mapping", self.stage_2_fold_mapping),
            ("Experiments", self.stage_3_experiments),
            ("Aggregation", self.stage_4_aggregation),
        ]

        completed = 0
        for stage_name, stage_func in stages:
            if stage_func():
                completed += 1
            else:
                self.log(f"Pipeline halted at stage: {stage_name}", "ERROR")
                break

        elapsed = time.time() - start_time

        self.log("\n" + "=" * 70)
        self.log("PIPELINE SUMMARY")
        self.log("=" * 70)
        self.log(f"Completed: {completed}/{len(stages)} stages")
        self.log(f"Elapsed time: {elapsed / 60:.1f} minutes")
        self.log(f"Log file: {self.log_file}")

        # Save summary
        summary_file = (
            self.logs_dir / f"pipeline_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )
        with open(summary_file, "w") as f:
            json.dump(self.results_summary, f, indent=2)

        self.log(f"Summary saved to: {summary_file}")

        return completed == len(stages)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Run full data scarcity experiment pipeline")
    parser.add_argument("--dataset", help="Single dataset to process")
    parser.add_argument("--all", action="store_true", help="Process all datasets")
    parser.add_argument("--datasets", nargs="+", help="List of datasets")
    parser.add_argument("--seed", type=int, default=2864, help="Random seed")

    args = parser.parse_args()

    # Determine datasets
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
    elif args.datasets:
        datasets = args.datasets
    elif args.dataset:
        datasets = [args.dataset]
    else:
        parser.print_help()
        return

    # Run pipeline
    orchestrator = PipelineOrchestrator(datasets, seed=args.seed)
    success = orchestrator.run_full_pipeline()

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
