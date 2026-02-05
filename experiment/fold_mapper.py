"""
Fold Mapping for Data Scarcity Experiment
==========================================

Maps the pre-existing 10-fold cross-validation to the reduced datasets.
Each fold file contains example IDs; we filter to keep only those in the reduced dataset.

This maintains the 10-fold CV structure while respecting the incremental data reduction.
"""

import argparse
import json
from pathlib import Path
from typing import List, Set


class FoldMapper:
    """Maps folds to reduced datasets."""

    def __init__(self, dataset_name: str):
        """Initialize fold mapper."""
        self.dataset_name = dataset_name

        # Paths - use absolute path resolution
        script_dir = Path(__file__).resolve().parent  # experiment directory
        self.base_dir = script_dir.parent  # project root
        self.folds_dir = self.base_dir / "data" / "folds" / dataset_name
        self.scarcity_dir = (
            self.base_dir / "data" / "data_reduced" / dataset_name
        )  # centralized data location
        self.metadata_dir = self.scarcity_dir / "metadata"

        # Find fold file
        self.fold_file = None
        for f in self.folds_dir.glob("folds*.txt"):
            self.fold_file = f
            break

        if not self.fold_file:
            raise FileNotFoundError(f"No fold file found in {self.folds_dir}")

        # Create folds directory
        self.folds_scarcity_dir = self.scarcity_dir / "folds"
        self.folds_scarcity_dir.mkdir(parents=True, exist_ok=True)

    def load_folds(self) -> List[Set[str]]:
        """
        Load folds from file.

        Format:
        |||
        id1
        id2
        ...
        |||
        ...

        Returns:
            List of sets, each set contains instance IDs for that fold
        """
        folds = []
        current_fold = set()

        with open(self.fold_file) as f:
            for line in f:
                line = line.strip()
                if line.startswith("|||"):
                    if current_fold:
                        folds.append(current_fold)
                        current_fold = set()
                elif line and not line.startswith("|"):
                    current_fold.add(line)

        if current_fold:
            folds.append(current_fold)

        print(f"Loaded {len(folds)} folds from {self.fold_file.name}")
        for i, fold in enumerate(folds):
            print(f"  Fold {i}: {len(fold)} instances")

        return folds

    def load_removed_instances(self, percentage: int) -> List[str]:
        """Load instances removed at given percentage."""
        metadata_file = self.metadata_dir / f"metadata_removed_{percentage:02d}.json"

        if not metadata_file.exists():
            raise FileNotFoundError(f"Metadata file not found: {metadata_file}")

        with open(metadata_file) as f:
            metadata = json.load(f)

        return metadata["removed_instance_ids"]

    def create_mapped_folds(self, removed_instances: List[str], percentage: int) -> None:
        """
        Create mapped folds for reduced dataset.

        Args:
            removed_instances: List of instance IDs to remove (cumulative)
            percentage: Percentage removed
        """
        # Load original folds
        folds = self.load_folds()

        # Convert to set for efficient lookup
        removed_set = set(removed_instances)

        # Filter folds
        filtered_folds = []
        for fold in folds:
            filtered_fold = fold - removed_set
            filtered_folds.append(filtered_fold)

        # Write mapped folds
        output_file = self.folds_scarcity_dir / f"folds_removed_{percentage:02d}.txt"

        with open(output_file, "w") as f:
            for fold in filtered_folds:
                f.write("|||\n")
                for instance_id in sorted(fold):
                    f.write(f"{instance_id}\n")
            f.write("|||\n")  # Final separator

        # Log statistics
        print(f"\n✓ Created mapped folds for {percentage}% reduction")
        for i, (original, filtered) in enumerate(zip(folds, filtered_folds)):
            print(f"  Fold {i}: {len(original)} → {len(filtered)} instances")

    def map_all_reductions(self) -> None:
        """Create fold mappings for all reduction percentages."""
        percentages = [10, 20, 50, 70, 90]

        print(f"\n{'=' * 70}")
        print(f"Mapping Folds for Reduced Datasets: {self.dataset_name.upper()}")
        print(f"{'=' * 70}\n")

        for percentage in percentages:
            try:
                removed = self.load_removed_instances(percentage)
                self.create_mapped_folds(removed, percentage)
            except Exception as e:
                print(f"Error processing {percentage}%: {e}")
                continue

        print(f"\n{'=' * 70}")
        print("Fold mapping complete!")
        print(f"Output directory: {self.folds_scarcity_dir}")
        print(f"{'=' * 70}\n")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Map folds to reduced datasets")
    parser.add_argument("--dataset", required=True, help="Dataset name")
    parser.add_argument("--all", action="store_true", help="Process all datasets")

    args = parser.parse_args()

    datasets_to_process = []

    if args.all:
        datasets_to_process = [
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
        datasets_to_process = [args.dataset]

    for dataset in datasets_to_process:
        try:
            mapper = FoldMapper(dataset)
            mapper.map_all_reductions()
        except Exception as e:
            print(f"Error processing {dataset}: {e}")
            continue


if __name__ == "__main__":
    main()
