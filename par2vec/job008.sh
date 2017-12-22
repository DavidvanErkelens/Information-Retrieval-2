#!/bin/bash
#SBATCH -n 1
#SBATCH -t 30
#SBATCH -p gpu
#SBATCH --output job008d.out

echo 'EXPERIMENT 8d'
python -u main.py --save_name ex8d --epochs 1000000 --friendly_print --print_freq 100 --backup_freq 15000 --learning_rate 1e-3 --dropout 0.0 --pos_sample_size 2048 --embedding_size_w 128 --embedding_size_d 32 --n_neg_samples 64 --window_size 8 --window_batch_size 128 --h_layers 128 64 8 --train

