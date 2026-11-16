#!/bin/bash -l
#SBATCH --job-name=escalabilidad_n100
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=slurm_escalabilidad_n100_%j.out
#SBATCH --error=slurm_escalabilidad_n100_%j.err
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
# Régimen extremo: n=100 fijo, hiperparámetros por defecto (E=200, h=32, B=1024),
# barrido q x p; solo combinaciones q multiplo de p. 6 reps (default).
python3 bench.py --ns 100 --ps 1,2,4,8,16,32 --qs 8,16,32,64,256,1024 \
    --solo escalabilidad --out resultados/escalabilidad_n100.csv
