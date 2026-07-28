# cluster_run

Scripts for running jobs on the SLURM cluster via Apptainer.

- `apptainer_XX.sh`
    SLURM batch script, submitted with `sbatch apptainer_XX.sh`. Before running,
  fill in your email address, partition, and GPU in the `#SBATCH` settings, and the bind-mount directories for the container. Thhis script runs `XX.sh` inside the Apptainer container. The script assumes that data is stored in `/scratch/yota/vision-llm-vae-data/`

- `XX.sh` 
    runs the actual Python code inside the container.
