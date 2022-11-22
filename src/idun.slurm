#!/bin/sh
#SBATCH --partition=GPUQ
#SBATCH --gres=gpu:1
#SBATCH --account=ie-idi
#SBATCH --time 04:00:00
#SBATCH --nodes=1
#SBATCH -c 28
#SBATCH --mem=12000
#SBATCH --job-name="Stock prediction training"
#SBATCH --output=prediction-srun.out
#SBATCH --mail-user=hannagn@stud.ntnu.no
#SBATCH --mail-type=ALL