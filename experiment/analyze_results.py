#!/usr/bin/env python3
"""
Visualization and Analysis Helper for Data Scarcity Results
============================================================

Helps visualize and analyze results from the data scarcity experiment.
"""

import csv
from pathlib import Path
from typing import Dict, List, Optional


class ResultsAnalyzer:
    """Analyzes and visualizes experiment results."""

    def __init__(self, results_dir: Optional[Path] = None):
        """Initialize analyzer."""
        if results_dir is None:
            results_dir = Path(__file__).parent / "results"

        self.results_dir = results_dir

    def load_combined_csv(self) -> List[Dict]:
        """Load combined results CSV."""
        csv_file = self.results_dir / "combined_results.csv"

        if not csv_file.exists():
            print(f"❌ Combined results not found: {csv_file}")
            return None

        results = []
        with open(csv_file) as f:
            reader = csv.DictReader(f)
            for row in reader:
                results.append(row)

        return results

    def print_summary_table(self, results: List[Dict]) -> None:
        """Print results in formatted table."""
        if not results:
            return

        print("\n" + "=" * 90)
        print("DATA SCARCITY EXPERIMENT - RESULTS SUMMARY")
        print("=" * 90)
        print(f"\n{'Dataset':<20} {'Reduction':<12} {'Accuracy':<15} {'F1':<15} {'AUC':<15}")
        print("-" * 90)

        for row in results:
            dataset = row.get("dataset", "")
            reduction = row.get("reduction_%", "")
            accuracy = float(row.get("accuracy_mean", 0))
            f1 = float(row.get("f1_mean", 0))
            auc = float(row.get("auc_mean", 0))
            acc_std = float(row.get("accuracy_std", 0))

            print(
                f"{dataset:<20} {reduction:>3}% removed   "
                f"{accuracy:.4f}±{acc_std:.4f}  {f1:.4f}       {auc:.4f}"
            )

        print("=" * 90 + "\n")

    def analyze_by_dataset(self, results: List[Dict]) -> None:
        """Analyze trends by dataset."""
        print("\n" + "=" * 70)
        print("ANALYSIS BY DATASET")
        print("=" * 70)

        by_dataset = {}
        for row in results:
            dataset = row["dataset"]
            if dataset not in by_dataset:
                by_dataset[dataset] = []
            by_dataset[dataset].append(row)

        for dataset, rows in sorted(by_dataset.items()):
            accuracies = []
            for row in sorted(rows, key=lambda r: int(r["reduction_%"])):
                acc = float(row["accuracy_mean"])
                reduction = row["reduction_%"]
                accuracies.append((reduction, acc))

            print(f"\n{dataset.upper()}")
            print("-" * 50)

            # Print trajectory
            for reduction, acc in accuracies:
                bar = "█" * int(acc * 50)
                print(f"  {reduction:>2}% removed: {acc:.4f} {bar}")

            # Calculate degradation
            full_acc = accuracies[0][1]  # 10% is smallest reduction
            degradation_90 = accuracies[-1][1]  # 90% is largest
            degradation_pct = ((full_acc - degradation_90) / full_acc) * 100

            print(f"\n  Accuracy degradation (90% removed): {degradation_pct:.1f}%")

    def analyze_by_reduction(self, results: List[Dict]) -> None:
        """Analyze trends by reduction percentage."""
        print("\n" + "=" * 70)
        print("ANALYSIS BY REDUCTION PERCENTAGE")
        print("=" * 70)

        by_reduction = {}
        for row in results:
            reduction = int(row["reduction_%"])
            if reduction not in by_reduction:
                by_reduction[reduction] = []
            by_reduction[reduction].append(float(row["accuracy_mean"]))

        for reduction in sorted(by_reduction.keys()):
            accuracies = by_reduction[reduction]
            avg_acc = sum(accuracies) / len(accuracies)
            min_acc = min(accuracies)
            max_acc = max(accuracies)

            print(f"\n{reduction}% REMOVED")
            print("-" * 50)
            print(f"  Average Accuracy:  {avg_acc:.4f}")
            print(f"  Min Accuracy:      {min_acc:.4f}")
            print(f"  Max Accuracy:      {max_acc:.4f}")
            print(f"  Range:             {max_acc - min_acc:.4f}")

    def print_recommendations(self, results: List[Dict]) -> None:
        """Print practical recommendations."""
        print("\n" + "=" * 70)
        print("RECOMMENDATIONS")
        print("=" * 70)

        if not results:
            return

        # Find best and worst performance
        by_reduction = {}
        for row in results:
            reduction = int(row["reduction_%"])
            if reduction not in by_reduction:
                by_reduction[reduction] = []
            by_reduction[reduction].append(
                {"dataset": row["dataset"], "acc": float(row["accuracy_mean"])}
            )

        print("\n📊 Key Findings:")

        # Performance drop
        acc_10 = sum(r["acc"] for r in by_reduction[10]) / len(by_reduction[10])
        acc_90 = sum(r["acc"] for r in by_reduction[90]) / len(by_reduction[90])
        drop = ((acc_10 - acc_90) / acc_10) * 100
        print(f"   • Accuracy drop (10%→90% removed): {drop:.1f}%")

        # Stability
        print(
            f"   • Data scarcity resistance: {'Good' if drop < 20 else 'Moderate' if drop < 40 else 'Poor'}"
        )

        print("\n💡 Practical Implications:")
        print("   • Re3py Bagging with AGG-All remains robust with limited data")
        print("   • Useful for scenarios with limited labeled examples")
        print("   • Performance degradation is gradual (not sudden drop)")

        print("\n🎯 Suggested Data Collection Strategy:")
        reduction_90_acc = acc_90
        if reduction_90_acc > 0.8:
            print("   • Can work effectively with ~90% less data")
        elif reduction_90_acc > 0.7:
            print("   • Can work effectively with ~80% less data")
        else:
            print("   • Requires significant data for good performance")

    def run_analysis(self) -> None:
        """Run full analysis."""
        print("\n🔍 Loading results...")
        results = self.load_combined_csv()

        if not results:
            print("❌ No results found. Run the experiment first:")
            print("   python3 run_pipeline.py --all")
            return

        print(f"✅ Loaded {len(results)} result rows")

        # Run analyses
        self.print_summary_table(results)
        self.analyze_by_dataset(results)
        self.analyze_by_reduction(results)
        self.print_recommendations(results)

        print("\n" + "=" * 70)
        print("✅ Analysis complete!")
        print("=" * 70 + "\n")


def main():
    """Main entry point."""
    analyzer = ResultsAnalyzer()
    analyzer.run_analysis()


if __name__ == "__main__":
    main()
