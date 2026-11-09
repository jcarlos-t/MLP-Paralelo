#!/bin/bash -l
#SBATCH --job-name=analisis
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:15:00
#SBATCH --output=slurm_analisis_%j.out
#SBATCH --error=slurm_analisis_%j.err

set -euo pipefail
module purge
module load gnu12/12.4.0 python3/3.13.2
source "$HOME/cpd-env/bin/activate"

cd "$SLURM_SUBMIT_DIR"
echo "nodo: $(hostname)"
# Post-procesamiento: consolida todos los CSV de resultados y genera figuras
python3 analisis.py --csv resultados/*.csv --out figs
