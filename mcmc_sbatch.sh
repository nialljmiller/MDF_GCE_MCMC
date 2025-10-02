#!/bin/bash
#SBATCH --job-name=bulge_MCMC
#SBATCH --output=logs/bulge_MCMC_%j.out
#SBATCH --error=logs/bulge_MCMC_%j.err
#SBATCH --account=galacticbulge
#SBATCH --partition=mb                 # Medicine Bow, preemptible
#SBATCH --qos=fast                     # 12h max; higher priority
#SBATCH --time=11:59:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=96             # grab all cores on the node
#SBATCH --exclusive
#SBATCH --requeue
#SBATCH --signal=B:USR1@120            # warning before preempt/time limit

set -euo pipefail
mkdir -p logs

PYENV="${PYENV:-$HOME/python_projects/venv}"
source "$PYENV/bin/activate"

cd "$SLURM_SUBMIT_DIR"

# one process bound to all 96 cores
FILE_PATH="${1:?usage: $0 <some_file_path>}"
srun --cpu-bind=cores -n 1 python MDF_MCMC_Launcher.py "$FILE_PATH"
