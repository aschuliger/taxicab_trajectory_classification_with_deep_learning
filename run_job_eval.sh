#!/bin/bash
#SBATCH -N 1
#SBATCH -n 4
#SBATCH --gres=gpu:1
#SBATCH --mem=115G

module load cuda92/toolkit
module load cudnn
python preprocess_data.py validation_data
python validation.py