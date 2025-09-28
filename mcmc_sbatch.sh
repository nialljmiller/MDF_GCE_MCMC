#!/usr/bin/env bash
#SBATCH -J bulge_MCMC
#SBATCH -A <YOUR_ACCOUNT>       # fill this in
#SBATCH -p mb                   # or teton/bearto
#SBATCH -N 1
#SBATCH --exclusive             # grab the whole node
#SBATCH -t 1-00:00:00           # 1 day
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err

set -euo pipefail
mkdir -p logs

# activate your env via alias
shopt -s expand_aliases
source ~/.bashrc
zenv

cd "$SLURM_SUBMIT_DIR"
srun python MDF_MCMC_Launcher.py
