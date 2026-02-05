"""
Data Scarcity Preprocessing Script
===================================

This script implements incremental and deterministic data reduction for the Re3py
data scarcity experiment.

Key Features:
- Incremental removal: 10%, 20%, 50%, 70%, 90%
- Deterministic: same instances removed at each step (cumulative)
- Maintains stratification of target values
- Saves removed instances for reproducibility
- Creates reduced target files and fold mappings

Usage:
    python data_scarcity_preprocessing.py --dataset basket --seed 2864
"""

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Tuple


class DataScarcityPreprocessor:
    """Handles incremental and deterministic data reduction."""

    def __init__(self, dataset_name: str, seed: int = 2864):
        """
        Initialize preprocessor.

        Args:
            dataset_name: Name of dataset (e.g., 'basket', 'carcinogenesis')
            seed: Random seed for reproducibility
        """
        self.dataset_name = dataset_name
        self.seed = seed
        random.seed(seed)

        # Base paths - use absolute path resolution
        script_dir = Path(__file__).resolve().parent  # experiment directory
        self.base_dir = script_dir.parent  # project root
        self.dataset_dir = self.base_dir / "data" / "datasets" / dataset_name
        self.scarcity_dir = self.base_dir / "data" / "data_reduced"  # centralized data location
        self.scarcity_dir.mkdir(parents=True, exist_ok=True)

        # Output directories
        self.dataset_scarcity_dir = self.scarcity_dir / dataset_name
        self.dataset_scarcity_dir.mkdir(parents=True, exist_ok=True)

        self.metadata_dir = self.dataset_scarcity_dir / "metadata"
        self.metadata_dir.mkdir(parents=True, exist_ok=True)

        # Find target file
        self.target_file = self._find_target_file()
        if not self.target_file:
            raise FileNotFoundError(f"No target file found for dataset {dataset_name}")

        # Reduction percentages (cumulative)
        self.reduction_percentages = [10, 20, 50, 70, 90]

    def _find_target_file(self) -> Path:
        """Find the target file for this dataset."""
        target_patterns = [
            f"{self.dataset_name}_target.txt",
            "muta188_target.txt",  # Special case for mutagenesis
        ]

        for pattern in target_patterns:
            target_file = self.dataset_dir / pattern
            if target_file.exists():
                return target_file
        return None

    def load_target_data(self) -> List[Tuple[str, str]]:
        """
        Load target data.

        Returns:
            List of (instance_id, label) tuples
        """
        data = []
        with open(self.target_file) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                # Parse line: relation(instance_id, label)
                # Extract instance_id and label
                start = line.find("(")
                end = line.rfind(")")
                if start == -1 or end == -1:
                    continue

                content = line[start + 1 : end]
                parts = content.rsplit(",", 1)
                if len(parts) == 2:
                    instance_id = parts[0].strip()
                    label = parts[1].strip()
                    data.append((instance_id, label))

        return data

    def stratified_sample(
        self, data: List[Tuple[str, str]], percentage: float
    ) -> List[Tuple[str, str]]:
        """
        Sample instances to remove while maintaining label stratification.

        Args:
            data: All instances
            percentage: Percentage to remove

        Returns:
            List of instances to remove (stratified)
        """

        by_label = defaultdict(list)
        for instance_id, label in data:
            by_label[label].append(instance_id)

        total_remove = int(len(data) * percentage / 100)
        to_remove = []

        for _label, instance_ids in by_label.items():
            label_remove = max(1, int(len(instance_ids) * percentage / 100))
            label_remove = min(label_remove, len(instance_ids))

            sampled = random.sample(instance_ids, label_remove)
            to_remove.extend(sampled)

        # Ensure we remove approximately the right percentage
        if len(to_remove) < total_remove:
            remaining = [iid for iid, _ in data if iid not in to_remove]
            additional = random.sample(remaining, total_remove - len(to_remove))
            to_remove.extend(additional)
        elif len(to_remove) > total_remove:
            to_remove = random.sample(to_remove, total_remove)

        return to_remove

    def create_reduced_dataset(self, removed_instances: List[str], percentage: int) -> None:
        """
        Create reduced target file and save metadata.

        Args:
            removed_instances: IDs of instances to remove
            percentage: Percentage removed (for naming)
        """
        # Read original target file
        with open(self.target_file) as f:
            lines = f.readlines()

        # Relation name from first line
        for line in lines:
            if "(" in line:
                _ = line[: line.find("(")]
                break

        # Write reduced target file
        output_file = self.dataset_scarcity_dir / f"target_removed_{percentage:02d}.txt"
        removed_set = set(removed_instances)
        kept_count = 0

        with open(output_file, "w") as f:
            for line in lines:
                line = line.strip()
                if not line or "(" not in line:
                    continue

                # Extract instance ID
                start = line.find("(")
                end = line.rfind(")")
                content = line[start + 1 : end]
                parts = content.rsplit(",", 1)

                if len(parts) == 2:
                    instance_id = parts[0].strip()
                    if instance_id not in removed_set:
                        f.write(line + "\n")
                        kept_count += 1

        # Save metadata
        metadata = {
            "dataset": self.dataset_name,
            "percentage_removed": percentage,
            "total_instances_original": len(removed_instances) + kept_count,
            "total_instances_kept": kept_count,
            "total_instances_removed": len(removed_instances),
            "removed_instance_ids": removed_instances,
            "seed": self.seed,
            "output_file": str(output_file),
        }

        metadata_file = self.metadata_dir / f"metadata_removed_{percentage:02d}.json"
        with open(metadata_file, "w") as f:
            json.dump(metadata, f, indent=2)

        print(f"✓ Created reduced dataset: {percentage}% removed")
        print(f"  Original: {metadata['total_instances_original']} instances")
        print(f"  Kept: {kept_count} instances")
        print(f"  Removed: {len(removed_instances)} instances")

    def preprocess(self) -> Dict[int, List[str]]:
        """
        Execute full preprocessing pipeline.

        Returns:
            Dictionary mapping percentage to list of removed instance IDs (cumulative)
        """
        print(f"\n{'=' * 70}")
        print(f"Data Scarcity Preprocessing: {self.dataset_name.upper()}")
        print(f"{'=' * 70}")

        # Load original data
        data = self.load_target_data()
        print(f"\nOriginal dataset size: {len(data)} instances")

        # Label distribution
        labels = [label for _, label in data]
        label_counts = Counter(labels)
        print(f"Label distribution: {dict(label_counts)}")

        # Track cumulative removals
        all_removed_cumulative = []
        removed_mapping = {}

        # Process each reduction percentage
        for percentage in self.reduction_percentages:
            print(f"\n--- Processing {percentage}% reduction ---")

            # Sample instances to remove AT THIS STEP (new instances only)
            current_step_remove_count = int(len(data) * percentage / 100)
            additional_remove_count = current_step_remove_count - len(all_removed_cumulative)

            # Get remaining instances (not yet removed)
            remaining = [iid for iid, _ in data if iid not in all_removed_cumulative]

            # Sample from remaining
            if additional_remove_count > 0 and remaining:
                # Stratified sampling from remaining
                remaining_data = [(iid, lbl) for iid, lbl in data if iid in remaining]

                # Calculate what percentage this represents of remaining
                pct_of_remaining = (additional_remove_count / len(remaining_data)) * 100

                new_removed = self.stratified_sample(remaining_data, pct_of_remaining)
                all_removed_cumulative.extend(new_removed)

            id_to_label = dict(data)
            kept_ids = [iid for iid, _ in data if iid not in all_removed_cumulative]
            kept_labels = [id_to_label[iid] for iid in kept_ids]
            kept_counts = Counter(kept_labels)

            print(f"  After {percentage}% removal, kept label distribution: {dict(kept_counts)}")

            if any(c == 0 for c in kept_counts.values()):
                print(f"  WARNING: some class has 0 instances after {percentage}% removal")

            # Create reduced dataset
            removed_mapping[percentage] = all_removed_cumulative.copy()
            self.create_reduced_dataset(all_removed_cumulative, percentage)

        # Save cumulative mapping
        mapping_file = self.metadata_dir / "removal_mapping.json"
        cumulative_mapping = {f"removed_{p:02d}": ids for p, ids in removed_mapping.items()}
        with open(mapping_file, "w") as f:
            json.dump(cumulative_mapping, f, indent=2)

        print(f"\n{'=' * 70}")
        print("Preprocessing complete!")
        print(f"Output directory: {self.dataset_scarcity_dir}")
        print(f"{'=' * 70}\n")

        return removed_mapping


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Data scarcity preprocessing for Re3py experiment")
    parser.add_argument(
        "--dataset", required=True, help="Dataset name (e.g., basket, carcinogenesis)"
    )
    parser.add_argument("--seed", type=int, default=2864, help="Random seed for reproducibility")
    parser.add_argument("--all", action="store_true", help="Process all datasets")

    args = parser.parse_args()

    datasets_to_process = []

    if args.all:
        # Paper datasets
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
            preprocessor = DataScarcityPreprocessor(dataset, seed=args.seed)
            preprocessor.preprocess()
        except Exception as e:
            print(f"Error processing {dataset}: {e}")
            continue


if __name__ == "__main__":
    main()
