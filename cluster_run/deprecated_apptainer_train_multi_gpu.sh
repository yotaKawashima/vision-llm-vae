#!/bin/bash

# =============================================================================
# SLURM Batch Script for Multi-GPU Model Training with Apptainer
# =============================================================================
# Submits a job to SLURM to train a model (see `config.py`) on multiple GPUs
# inside an Apptainer container, by running `train.sh`.
# =============================================================================

# SLURM Job Configuration
# These directives configure the SLURM job scheduler
#SBATCH --mail-type=ALL                           # Send email on all job events
#SBATCH --mail-user=XXX                           # Email address for notifications
#SBATCH --job-name=train                          # Job name as shown in queue
#SBATCH --output=vae-train-%j.out                 # Output file with job ID
#SBATCH --nodes=1                                 # Number of compute nodes
#SBATCH --ntasks=1                                # Total number of tasks
#SBATCH --ntasks-per-node=1                       # Tasks per node
#SBATCH --cpus-per-task=32                        # CPU cores per task
#SBATCH --gres=gpu:a5000:4                        # Request 4 GPU
#SBATCH --mem=90G                                 # Memory allocation
#SBATCH --time=00:05:00
#SBATCH --partition=XXX
#SBATCH --qos=standard

REPO_DIR="/scratch/yota/vision-llm-vae"
SIF="/scratch/yota/apptainer-vision-llm-vae/image_mini.sif"

cd ${REPO_DIR}
git pull

# Execution
# Run using Apptainer container
# Apptainer options explained:
# --nv: Enable NVIDIA GPU support in container
# --bind: Mount host directories into the container
echo "Starting Apptainer..."

apptainer run --nv \
              --bind ${REPO_DIR}:/mnt \
              ${SIF} bash /mnt/cluster_run/train.sh

echo "Job complted successfully!"
