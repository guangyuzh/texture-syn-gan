#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --time=48:00:00
#SBATCH --gres=gpu:1
#SBATCH --partition=p40_4
#SBATCH --mem=100GB
#SBATCH --job-name=gan
#SBATCH --mail-type=END
#SBATCH --mail-user=gz612@nyu.edu
#SBATCH --output=logs/gan.out

module purge
module load pytorch/python3.6/0.3.0_4


CONFIG='default'

cd ../src/dcgan
python3 -u main.py --dataset folder --dataroot ../../data/dtd/ --config $CONFIG > logs/gan.log

