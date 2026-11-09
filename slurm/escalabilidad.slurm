#!/bin/bash -l
#SBATCH --job-name=escalabilidad
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=slurm_escalabilidad_%j.out
#SBATCH --error=slurm_escalabilidad_%j.err
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
# Caso 1 (escalabilidad): barrido q x p a n=10000 fija; solo combinaciones q multiplo de p
python3 bench.py --ns 10000 --ps 1,2,4,8,16,32 --qs 16,32,64,128,256 \
    --solo escalabilidad --out resultados/escalabilidad.csv
