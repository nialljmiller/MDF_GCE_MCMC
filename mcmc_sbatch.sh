#!/usr/bin/env bash
# mcmc_sbatch.sh — CPU job, 1 node

#SBATCH -J bulge_MCMC
#SBATCH -A <YOUR_ACCOUNT>              # <-- put your ARCC account
#SBATCH -p mb                          # or teton / bearto
#SBATCH -N 1
#SBATCH --ntasks-per-node=96
#SBATCH --cpus-per-task=1
#SBATCH --mem=0
#SBATCH -t 06:00:00
#SBATCH --hint=nomultithread
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err

set -euo pipefail

mkdir -p logs
module --force purge

# ---- Activate env via your alias `zenv` (preferred), else VENV_PATH fallback ----
# Enable aliases in this non-interactive shell and source common rc files.
shopt -s expand_aliases || true
for rc in ~/.bashrc ~/.bash_profile ~/.profile; do
  [[ -f "$rc" ]] && source "$rc"
done

if alias zenv &>/dev/null; then
  zenv
elif [[ -n "${VENV_PATH:-}" && -d "$VENV_PATH/bin" ]]; then
  # shellcheck disable=SC1090
  source "$VENV_PATH/bin/activate"
else
  echo "No env: alias 'zenv' not found and VENV_PATH not set. Proceeding with system Python." >&2
fi

# Keep threaded libs from oversubscribing the node
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

cd "$SLURM_SUBMIT_DIR"

# Sanity check so failures are obvious in logs
python -V
python -c "import sys,site; print(sys.executable); print(site.getsitepackages())" || {
  echo "Python not usable. Bailing." >&2; exit 3; }

# Run
srun --cpu-bind=cores python MDF_MCMC_Launcher.py
