#!/bin/bash
export CUDA_VISIBLE_DEVICES=0

# Experiment 4
uv run ./cs336_basics/script/train.py \
  --vocab_size 10000 \
  --context_length 256 \
  --d_model 512 \
  --num_layers 4 \
  --num_heads 16 \
  --d_ff 1344 \
  --rope_theta 10000 \
  --lr 3e-5 \
  --num_train_epochs 40000 \
  --batch_size 32 \
  --device cuda \
  --train_dataset ./data/tokenizer/train_token_ids_10000.bin \
  --val_dataset ./data/tokenizer/valid_token_ids_10000.bin \
  --out checkpoints/ablation/no_rmsnorm4 \
  --src checkpoints/ablation/no_rmsnorm4 \
  --max_learning_rate 3e-5 \
  --min_learning_rate 5e-6 \
  --warmup_iters 4000 \
  --cosine_cycle_iters 40000 \
  --betas "(0.9, 0.95)" \
  --eps 1e-8 \
  --weight_decay 0.1 \
  --max_l2_norm 1.0 \
  --keep_last_n_ckpt 5 \
  --save_every 500 \
  --no_rmsnorm \
  --act_fn swiglu

# Experiment 5
uv run ./cs336_basics/script/train.py \
  --vocab_size 10000 \
  --context_length 256 \
  --d_model 512 \
  --num_layers 4 \
  --num_heads 16 \
  --d_ff 1344 \
  --rope_theta 10000 \
  --lr 5e-4 \
  --num_train_epochs 40000 \
  --batch_size 32 \
  --device cuda \
  --train_dataset ./data/tokenizer/train_token_ids_10000.bin \
  --val_dataset ./data/tokenizer/valid_token_ids_10000.bin \
  --out checkpoints/ablation/yes_postnorm \
  --src checkpoints/ablation/yes_postnorm \
  --max_learning_rate 5e-4 \
  --min_learning_rate 1e-5 \
  --warmup_iters 4000 \
  --cosine_cycle_iters 40000 \
  --betas "(0.9, 0.95)" \
  --eps 1e-8 \
  --weight_decay 0.1 \
  --max_l2_norm 1.0 \
  --keep_last_n_ckpt 5 \
  --save_every 500 \
  --post_norm \
  --act_fn swiglu
