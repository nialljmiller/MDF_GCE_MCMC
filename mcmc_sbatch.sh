# mcmc_sbatch.sh (CPU, 1 full node)
#SBATCH -J bulge_mcmc_one
#SBATCH -p mb
#SBATCH -N 1
#SBATCH --ntasks-per-node=96     # one linux task per core
#SBATCH --cpus-per-task=1
#SBATCH --mem=0                  # all memory on the node
#SBATCH -t 06:00:00
#SBATCH --hint=nomultithread

module purge
source /path/to/venv/bin/activate
srun --cpu-bind=cores python MDF_MCMC_Launcher.py
