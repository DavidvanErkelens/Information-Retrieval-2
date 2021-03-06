#!/bin/bash
#SBATCH -n 1
#SBATCH -t 2500
#SBATCH -p gpu
#SBATCH --output job007b.out
echo 'EXPERIMENT 7b'
python -u main.py --save_name ex7b --epochs 1000000 --friendly_print --print_freq 100 --backup_freq 15000 --learning_rate 1e-3 --dropout 0.0 --pos_sample_size 1024 --embedding_size_w 128 --embedding_size_d 64 --n_neg_samples 64 --window_size 8 --window_batch_size 128 --h_layers 32 16 16 --train --load_model models/ex7b_15000.ckpt

