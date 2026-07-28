#!/bin/bash

# =============================================================================
# SLURM Batch Script for RSA with Apptainer
# =============================================================================
# Submits a job to SLURM to run representational similarity analysis (RSA)
# between model activations and fMRI RDMs, inside an Apptainer container,
# by running `rsa.sh`.
#
# Usage: sbatch apptainer_rsa.sh
# =============================================================================

# SLURM Job Configuration
# These directives configure the SLURM job scheduler
#SBATCH --mail-type=ALL     # Send email on all job events
#SBATCH --mail-user=XXX     # Email address for notifications
#SBATCH --job-name=rsa      # Job name as shown in queue
#SBATCH --output=rsa-%j.out # Output file with job ID
#SBATCH --nodes=1           # Number of compute nodes
#SBATCH --ntasks=1          # Total number of tasks
#SBATCH --ntasks-per-node=1 # Tasks per node
#SBATCH --cpus-per-task=4   # CPU cores per task
##SBATCH --gres=gpu:a5000:1 # Request 1 GPU
#SBATCH --mem=4G            # 50G for resnet18 Memory allocation
#SBATCH --time=00:10:00     # 03:30 not enough for ae 
##SBATCH --partition=XXX
#SBATCH --qos=standard

REPO_DIR="/home/yota/vision-llm-vae"
SIF="/home/yota/apptainer-vision-llm-vae/image_mini.sif"

cd ${REPO_DIR}
#git pull

# Execution
# Run using Apptainer container
# Apptainer options explained:
# --nv: Enable NVIDIA GPU support in container
# --bind: Mount host directories into the container
echo "Starting Apptainer..."

apptainer run --nv \
              --bind ${REPO_DIR}:/mnt \
              --bind /scratch/yota/vision-llm-vae-data:/mnt/data \
              ${SIF} bash /mnt/cluster_run/rsa.sh

echo "Job complted successfully!"
