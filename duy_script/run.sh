#!/usr/bin/env bash

#SBATCH -J "ResNet-d"
#SBATCH -o logs/logs_%A.txt
#SBATCH --mem-per-cpu=70000
#SBATCH --nodes 2
#SBATCH -p gpu
#SBATCH --gres=gpu:teslav100:2
#SBATCH --mail-user=duy.vu@tuni.fi
#SBATCH -t 2-23:59:00

export PYTHONPATH=$PYTHONPATH:.
source torch-dcase/bin/activate/

module load CUDA
python training.py 