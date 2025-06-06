#!/bin/bash
export CUDA_VISIBLE_DEVICES=3

# Experiment 6
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
  --train_dataset /mnt/disk3/yusheng/assignment1-basics/data/tokenizer/train_token_ids_10000.bin \
  --val_dataset /mnt/disk3/yusheng/assignment1-basics/data/tokenizer/valid_token_ids_10000.bin \
  --out checkpoints/ablation/no_rope \
  --src checkpoints/ablation/no_rope \
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
  --no_rope \
  --act_fn swiglu

# Experiment 7
uv run ./cs336_basics/script/train.py \
  --vocab_size 10000 \
  --context_length 256 \
  --d_model 512 \
  --num_layers 4 \
  --num_heads 16 \
  --d_ff 2048 \
  --rope_theta 10000 \
  --lr 5e-4 \
  --num_train_epochs 40000 \
  --batch_size 32 \
  --device cuda \
  --train_dataset /mnt/disk3/yusheng/assignment1-basics/data/tokenizer/train_token_ids_10000.bin \
  --val_dataset /mnt/disk3/yusheng/assignment1-basics/data/tokenizer/valid_token_ids_10000.bin \
  --out checkpoints/ablation/silu \
  --src checkpoints/ablation/silu \
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
  --act_fn silu
