#!/usr/bin/env python3
"""
Data Scarcity Experiment - Interactive CLI
===========================================

Provides interactive guidance for running the data scarcity experiment.
"""

import sys
import subprocess
from pathlib import Path
from typing import List


class ExperimentCLI:
    """Interactive CLI for experiment execution."""
    
    def __init__(self):
        self.base_dir = Path(__file__).parent
        self.datasets = [
            'basket', 'carcinogenesis', 'imdb_big', 'movie',
            'mutagenesis', 'stack_big', 'uwcse', 'webkb', 'yelp_big'
        ]
    
    def print_banner(self):
        """Print welcome banner."""
        print("""
╔════════════════════════════════════════════════════════════════╗
║                                                                ║
║     DATA SCARCITY EXPERIMENT FOR RE3PY                         ║
║     ═══════════════════════════════════════════               ║
║                                                                ║
║     Model:       Bagging (AGG-All)                             ║
║     Validation:  10-fold Cross-Validation                      ║
║     Metric:      Accuracy (primary)                            ║
║     Datasets:    9 benchmark datasets                          ║
║     Reductions:  10%, 20%, 50%, 70%, 90%                       ║
║                                                                ║
╚════════════════════════════════════════════════════════════════╝
        """)
    
    def print_menu(self):
        """Print main menu."""
        print("\n┌─ SELECT EXPERIMENT MODE ─────────────────────────┐")
        print("│                                                  │")
        print("│  [1] Quick Test (basket dataset only)          │")
        print("│  [2] Full Experiment (all 9 datasets)          │")
        print("│  [3] Single Dataset (custom)                   │")
        print("│  [4] View Documentation                        │")
        print("│  [5] Exit                                      │")
        print("│                                                  │")
        print("└──────────────────────────────────────────────────┘")
    
    def run_quick_test(self):
        """Run quick test on basket dataset."""
        print("\n🚀 Starting quick test on 'basket' dataset...")
        print("   ℹ️  Estimated time: 5-10 minutes\n")
        
        self._run_pipeline(['--dataset', 'basket'])
    
    def run_full_experiment(self):
        """Run full experiment on all datasets."""
        print("\n🚀 Starting full experiment on all 9 datasets...")
        print("   ℹ️  Estimated time: 1-2 hours")
        print("   💡 You can safely interrupt with Ctrl+C\n")
        
        response = input("⚠️  Continue? (yes/no): ").strip().lower()
        if response in ['yes', 'y']:
            self._run_pipeline(['--all'])
        else:
            print("❌ Experiment cancelled.")
    
    def run_single_dataset(self):
        """Run experiment on single dataset."""
        print("\n📊 Available datasets:")
        for i, ds in enumerate(self.datasets, 1):
            print(f"   [{i}] {ds}")
        
        choice = input("\nSelect dataset (number or name): ").strip()
        
        # Try to parse as number
        try:
            idx = int(choice) - 1
            if 0 <= idx < len(self.datasets):
                dataset = self.datasets[idx]
            else:
                print("❌ Invalid selection.")
                return
        except ValueError:
            # Try as name
            if choice in self.datasets:
                dataset = choice
            else:
                print(f"❌ Dataset '{choice}' not found.")
                return
        
        print(f"\n🚀 Starting experiment on '{dataset}' dataset...")
        self._run_pipeline(['--dataset', dataset])
    
    def view_documentation(self):
        """Show documentation."""
        print("\n📚 DOCUMENTATION")
        print("=" * 50)
        print("""
📄 Main README:
   /scarcity_experiment/README.md
   
📋 Implementation Summary:
   /scarcity_experiment/IMPLEMENTATION_SUMMARY.md
   
⚙️  Configuration:
   /scarcity_experiment/config.yaml
   
Python Files:
   • data_scarcity_preprocessing.py  - Stage 1: Data reduction
   • fold_mapper.py                  - Stage 2: Fold mapping
   • run_experiment.py               - Stage 3: Run experiments
   • run_pipeline.py                 - Stage 4: Full pipeline
   
Results:
   • results/                        - Output directory
   • results/{dataset}/results_summary.json
   • results/combined_results.csv

Quick Commands:
   
   # Test su dataset singolo
   python3 run_pipeline.py --dataset basket
   
   # Esperimento completo
   python3 run_pipeline.py --all
   
   # Con seed personalizzato
   python3 run_pipeline.py --all --seed 12345
        """)
    
    def _run_pipeline(self, args: List[str]):
        """Run pipeline with given arguments."""
        try:
            cmd = [sys.executable, str(self.base_dir / 'run_pipeline.py')] + args
            
            print("─" * 50)
            print(f"Command: {' '.join(cmd)}")
            print("─" * 50 + "\n")
            
            result = subprocess.run(cmd, cwd=str(self.base_dir))
            
            if result.returncode == 0:
                print("\n✅ Experiment completed successfully!")
                print(f"\n📊 Results saved to: {self.base_dir / 'results'}/")
                
                # Show results summary
                import json
                results_dir = self.base_dir / 'results'
                if results_dir.exists():
                    csv_file = results_dir / 'combined_results.csv'
                    if csv_file.exists():
                        print("\n📈 Results preview:")
                        with open(csv_file, 'r') as f:
                            lines = f.readlines()
                            for line in lines[:min(6, len(lines))]:
                                print(f"   {line.rstrip()}")
                        if len(lines) > 6:
                            print(f"   ... ({len(lines)-6} more rows)")
            else:
                print(f"\n❌ Experiment failed with code {result.returncode}")
                
        except KeyboardInterrupt:
            print("\n\n⚠️  Experiment interrupted by user.")
            print("💾 Intermediate results have been saved.")
        except Exception as e:
            print(f"\n❌ Error: {e}")
    
    def run(self):
        """Run interactive CLI."""
        self.print_banner()
        
        while True:
            self.print_menu()
            choice = input("Enter choice (1-5): ").strip()
            
            if choice == '1':
                self.run_quick_test()
            elif choice == '2':
                self.run_full_experiment()
            elif choice == '3':
                self.run_single_dataset()
            elif choice == '4':
                self.view_documentation()
            elif choice == '5':
                print("\n👋 Goodbye!\n")
                break
            else:
                print("\n❌ Invalid choice. Please try again.\n")


def main():
    """Main entry point."""
    if len(sys.argv) > 1:
        # Non-interactive mode
        import subprocess
        cmd = [sys.executable, 'run_pipeline.py'] + sys.argv[1:]
        subprocess.run(cmd)
    else:
        # Interactive mode
        cli = ExperimentCLI()
        cli.run()


if __name__ == '__main__':
    main()
