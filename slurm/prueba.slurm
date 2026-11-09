#!/bin/bash -l
#SBATCH --job-name=cpd_mlp_prueba
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:20:00
#SBATCH --output=slurm_prueba_%j.out
#SBATCH --mail-type=END,FAIL

set -euo pipefail
module purge
module load gnu12/12.4.0 python3/3.13.2
source "$HOME/cpd-env/bin/activate"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

cd "$SLURM_SUBMIT_DIR"
python3 dev/beta1.py --n 5000 --q 8 --E 50 --B 1024
python3 mlp_paralelo.py --n 5000 --q 8 --p 4 --E 50
python3 bench.py --ns 2000,5000 --ps 1,2,4 --q 8 --qs 8,16 --reps 1 --E 50 \
    --solo impacto_n,escalabilidad --out resultados/prueba.csv
python3 analisis.py --csv resultados/prueba.csv --out figs_prueba
