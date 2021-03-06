#!/bin/bash
#SBATCH -n 1
#SBATCH -t 1500
#SBATCH -p gpu
#SBATCH --output job010b.out
echo 'EXPERIMENT 10b'
python -u main.py --save_name ex10b --epochs 1000000 --friendly_print --print_freq 100 --backup_freq 15000 --learning_rate 1e-3 --dropout 0.0 --pos_sample_size 2048 --embedding_size_w 128 --embedding_size_d 128 --n_neg_samples 64 --window_size 8 --window_batch_size 128 --h_layers 32 8 --train --load_model models/ex1b_15000.ckpt
