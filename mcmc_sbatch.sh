#!/bin/bash
#SBATCH --job-name=BulgeMCMC
#SBATCH --output=logs/BulgeMCMC_%j.out
#SBATCH --error=logs/BulgeMCMC_%j.err
#SBATCH --account=galacticbulge
#SBATCH --partition=mb            # adjust to your cluster
#SBATCH --qos=fast                # adjust if needed
#SBATCH --time=23:59:00
#SBATCH --nodes=1
#SBATCH --ntasks=1                # one Python proc
#SBATCH --cpus-per-task=96        # grab the full node (edit if your node has different core count)
#SBATCH --exclusive
#SBATCH --requeue
#SBATCH --signal=B:USR1@120

set -euo pipefail

# ------------------ Set up environment ------------------
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

# Pin math libs to one thread per process (emcee will parallelize walkers via multiprocessing)
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export MPLBACKEND=Agg

# ------------------ Run directory ------------------
STAMP=$(date +"%Y%m%d_%H%M%S")
RUN_DIR="mcmc_run_${STAMP}"
mkdir -p "${RUN_DIR}"

# Copy launcher + inlist into run dir (we patch copies, not your sources)
cp MDF_MCMC_Launcher.py "${RUN_DIR}/"
cp bulge_pcard.txt     "${RUN_DIR}/bulge_pcard.mcmc.txt"

# ------------------ Choose walkers & steps ------------------
# As many walkers as practical: 4x cores is a good high-throughput setting
# (You can bump this higher if your model is cheap; emcee needs >= 2*ndim)
CPUS=${SLURM_CPUS_PER_TASK:-${SLURM_CPUS_ON_NODE:-64}}
WALKERS=$(( 4 * CPUS ))
if [ "${WALKERS}" -lt 64 ]; then WALKERS=64; fi

# At least 2048 steps; set higher for better posteriors
STEPS=4096

# ------------------ Patch the pcard copy with overrides ------------------
# We override only what we need; the launcher’s parser will take the last seen key.
{
  echo ""
  echo "# ---- overrides for this MCMC run ----"
  echo "output_path: '${RUN_DIR}/'"
  echo "walkers: ${WALKERS}"
  echo "steps: ${STEPS}"
  # (optional) output every N steps; set to 128 for regular corner/fit dumps
  echo "output_interval: 128"
} >> "${RUN_DIR}/bulge_pcard.mcmc.txt"

# ------------------ Force PP=True in the copied launcher ------------------
# Two occurrences: constructor & run() call.
sed -i 's/PP=False/PP=True/g' "${RUN_DIR}/MDF_MCMC_Launcher.py"

# ------------------ Run ------------------
echo "Launching MCMC in ${RUN_DIR} with WALKERS=${WALKERS}, STEPS=${STEPS}, CPUS=${CPUS}"
srun -n 1 -c "${CPUS}" \
  python -u "${RUN_DIR}/MDF_MCMC_Launcher.py" "${RUN_DIR}/bulge_pcard.mcmc.txt"

echo "Done."

