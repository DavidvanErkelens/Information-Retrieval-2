#!/bin/bash
#SBATCH -n 1
#SBATCH -t 2500
#SBATCH -p gpu
#SBATCH --output job005c.out
echo 'EXPERIMENT 5c'
python -u main.py --save_name ex5c --epochs 1000000 --friendly_print --print_freq 100 --backup_freq 15000 --learning_rate 1e-3 --dropout 0.0 --pos_sample_size 1024 --embedding_size_w 128 --embedding_size_d 128 --n_neg_samples 64 --window_size 8 --window_batch_size 128 --h_layers 64 20 --train
