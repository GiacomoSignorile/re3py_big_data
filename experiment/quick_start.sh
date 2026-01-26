#!/bin/bash
# Quick Start Guide for Data Scarcity Experiment
# ===============================================

set -e  # Exit on error

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

echo "=================================================="
echo "DATA SCARCITY EXPERIMENT - QUICK START"
echo "=================================================="
echo ""
echo "This script will run the complete experiment pipeline."
echo ""

# Parse arguments
EXPERIMENT_TYPE="${1:-single}"  # single, all, or dataset name
SEED="${2:-2864}"

case "$EXPERIMENT_TYPE" in
  "all")
    echo "Mode: All datasets"
    python3 run_pipeline.py --all --seed "$SEED"
    ;;
  "single")
    echo "Mode: Single dataset (basket)"
    python3 run_pipeline.py --dataset basket --seed "$SEED"
    ;;
  *)
    echo "Mode: Single dataset ($EXPERIMENT_TYPE)"
    python3 run_pipeline.py --dataset "$EXPERIMENT_TYPE" --seed "$SEED"
    ;;
esac

echo ""
echo "=================================================="
echo "EXPERIMENT COMPLETE"
echo "=================================================="
echo ""
echo "Results location:"
ls -lh results/ 2>/dev/null || echo "  No results found yet"
echo ""
echo "Check combined results:"
cat results/combined_results.csv 2>/dev/null || echo "  Results will be available after first run"
