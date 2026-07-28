#!/bin/bash

# =============================================================================
# RSA Script
# =============================================================================
# This script is executed inside the Apptainer container to run the actual
# process.
# =============================================================================

# =============================================================================
# Directory Setup
# =============================================================================
# Navigate to the cloned repository directory
cd /mnt/
# Set the minimum interval for tqdm progress bars
export TQDM_MININTERVAL=5
export PYTHONUNBUFFERED=1

echo "running..."
# =============================================================================
# Execution
# =============================================================================
export PYTHONPATH=.
echo "before python..."
python -m experiments.analysis rdm-regression "$@"
echo "after python..."
