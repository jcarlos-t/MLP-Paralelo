#!/bin/bash -l
#SBATCH --job-name=impacto_n
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=slurm_impacto_n_%j.out
#SBATCH --error=slurm_impacto_n_%j.err
#SBATCH --mail-type=END,FAIL

set -euo pipefail
module purge
module load gnu12/12.4.0 python3/3.13.2
source "$HOME/cpd-env/bin/activate"

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

cd "$SLURM_SUBMIT_DIR"
echo "nodo: $(hostname)  cpus: $SLURM_CPUS_PER_TASK"
# Caso 2 (impacto de n): q=32 fijo, rejilla n x p; n>=5000 se reutiliza del cache si es el mismo nodo
python3 bench.py --ns 500,1000,2000,5000,10000,20000,40000 --ps 1,2,4,8,16,32 --q 32 \
    --solo impacto_n --out resultados/impacto_n.csv
