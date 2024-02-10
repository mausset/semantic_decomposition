#!/usr/bin/env bash
#SBATCH --mem  20GB
#SBATCH --gres gpu:4
#SBATCH --cpus-per-task 16
#SBATCH --time 14:00:00
#SBATCH --constrain "arwen"
#SBATCH --mail-type FAIL
#SBATCH --mail-user maus@kth.se
#SBATCH --output /Midgard/home/%u/
#SBATCH --error  /Midgard/home/%u/

echo "Starting job ${SLURM_JOB_ID} on ${SLURMD_NODENAME}"
nvidia-smi
. ~/miniconda3/etc/profile.d/conda.sh
conda activate dev
cd ~/projects/semantic_decomposition/
python train.py fit -c configs/test.yaml


