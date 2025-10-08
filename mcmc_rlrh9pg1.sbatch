#!/bin/bash
#SBATCH --job-name=Bulge_MCMC-config_comp_iniab_output_feh_p000-txt_imf_salpeter_sn1a_sn1a
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

FILE_PATH='/home/njm/MDF_GCE_MCMC/config_comp_iniab_output_feh_p000.txt_imf_salpeter_sn1a_sn1a_gronow.txt_stellar_yield_agb_and_massive_stars_c15_n13_0_0_hne.txt_sn1a_power_law'

# Check for config file in the provided directory
PCARD="$FILE_PATH/bulge_pcard.txt"
if [[ ! -f "$PCARD" ]]; then
  echo "Missing per-run pcard: $PCARD" >&2
  exit 2
fi

unset SLURM_MEM_PER_CPU SLURM_MEM_PER_GPU SLURM_MEM_PER_NODE

srun --cpu-bind=cores -n 1 python MDF_MCMC_Launcher.py "$FILE_PATH"