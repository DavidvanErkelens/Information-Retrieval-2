#!/bin/bash
#SBATCH -n 1
#SBATCH -t 300
#SBATCH -p gpu
echo 'EXPERIMENT 4'
python -u main.py --save_name ex4 --epochs 100000 --friendly_print --print_freq 100 --backup_freq 5000 --learning_rate 1e-3 --dropout 0.0 --pos_sample_size 128 --embedding_size_w 128 --embedding_size_d 16 --n_neg_samples 64 --window_size 8 --window_batch_size 128 --h_layers 64 20
