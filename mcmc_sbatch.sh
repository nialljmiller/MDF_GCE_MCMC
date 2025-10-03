#!/bin/bash
#SBATCH --job-name=BulgeMCMC
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --account=galacticbulge
#SBATCH --partition=mb
#SBATCH --qos=fast
#SBATCH --time=23:59:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=${SLURM_CPUS_PER_TASK:-96}
#SBATCH --exclusive
#SBATCH --requeue
#SBATCH --signal=B:USR1@120

set -euo pipefail
mkdir -p logs

# Use venv if provided
PYENV="${PYENV:-$HOME/python_projects/venv}"
if [[ -f "$PYENV/bin/activate" ]]; then
  source "$PYENV/bin/activate"
fi

# In run dir; use the per-run pcard we just wrote
PCARD="./bulge_pcard.txt"
if [[ ! -f "$PCARD" ]]; then
  echo "Missing per-run pcard: $PCARD" >&2
  exit 2
fi

# Launch the MCMC (assumes your launcher accepts pcard path)
if command -v python3 >/dev/null 2>&1; then PY=python3; else PY=python; fi

echo "Starting ${SLURM_JOB_NAME} in ${PWD}"
exec srun -n 1 -c "${SLURM_CPUS_PER_TASK}" "$PY" -u MDF_MCMC_Launcher.py "$PCARD"
