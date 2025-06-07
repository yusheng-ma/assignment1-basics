#!/bin/bash
export CUDA_VISIBLE_DEVICES=0

# Run experiment 0
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
  --out checkpoints/ablation/no_rmsnorm0 \
  --src checkpoints/ablation/no_rmsnorm0 \
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
  --no_rmsnorm \
  --act_fn swiglu

# Run experiment 1
uv run ./cs336_basics/script/train.py \
  --vocab_size 10000 \
  --context_length 256 \
  --d_model 512 \
  --num_layers 4 \
  --num_heads 16 \
  --d_ff 1344 \
  --rope_theta 10000 \
  --lr 1e-4 \
  --num_train_epochs 40000 \
  --batch_size 32 \
  --device cuda \
  --train_dataset ./data/tokenizer/train_token_ids_10000.bin \
  --val_dataset ./data/tokenizer/valid_token_ids_10000.bin \
  --out checkpoints/ablation/no_rmsnorm1 \
  --src checkpoints/ablation/no_rmsnorm1 \
  --max_learning_rate 1e-4 \
  --min_learning_rate 1e-5 \
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
