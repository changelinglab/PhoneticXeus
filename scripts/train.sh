#!/bin/bash
# Customize SBATCH directives for your cluster
#SBATCH -A YOUR_ACCOUNT
#SBATCH -p YOUR_PARTITION
#SBATCH -J train
#SBATCH -N 1
#SBATCH --gpus-per-node=1
#SBATCH -c 72
#SBATCH --mem=120G
#SBATCH -t 48:00:00
#SBATCH -o exp/slurm_logs/%x/%j.out
#SBATCH -e exp/slurm_logs/%x/%j.out

# run with
# sbatch -J train scripts/train.sh
# [args for main.py]

# === Directory Setup ===
cd "${SLURM_SUBMIT_DIR:-.}"
mkdir -p "exp/slurm_logs/${SLURM_JOB_NAME}"
# === Environment setup ===
source .venv/bin/activate
# espeak-ng (needed for wav2vec2-phoneme models)
export PHONEMIZER_ESPEAK_LIBRARY="${PHONEMIZER_ESPEAK_LIBRARY:?Set PHONEMIZER_ESPEAK_LIBRARY}"
export ESPEAK_DATA_PATH="${ESPEAK_DATA_PATH:?Set ESPEAK_DATA_PATH}"
###########################

python src/main.py experiment=train/ipapack_xeuspr "$@"
