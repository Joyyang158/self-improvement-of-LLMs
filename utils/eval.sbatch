#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:a100:4
#SBATCH --mem=80GB
#SBATCH --time=01:00:00
#SBATCH --job-name eval-zephyr-iter0
#SBATCH --output logs/eval-zephyr-iter0.log


module load gcc/8.1.0
module load git

source /gpfs/home/hy2847/.bashrc

conda activate evalenv

cd /gpfs/home/hy2847/self-improvement-of-LLMs

bash utils/eval.sh