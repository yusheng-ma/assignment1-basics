#!/bin/bash

BASE_LR=5e-4
BASE_BS=32
BATCH_LIST=(64 128 256)

for BS in "${BATCH_LIST[@]}"; do
  # Scale LR linearly with batch size
  SCALE=$(echo "$BS / $BASE_BS" | bc -l)
  LR=$(python3 -c "print($BASE_LR * $SCALE)")

  OUTDIR=checkpoints/bs_sweep/bs_$BS
  SRCDIR=checkpoints/bs_sweep/bs_$BS

  echo "Running with batch size $BS and learning rate $LR"

  mkdir -p $OUTDIR

  uv run ./cs336_basics/script/train.py \
    --vocab_size 10000 \
    --context_length 256 \
    --d_model 512 \
    --num_layers 4 \
    --num_heads 16 \
    --d_ff 1344 \
    --rope_theta 10000 \
    --lr $LR \
    --num_train_epochs 40000 \
    --batch_size $BS \
    --device cuda:3 \
    --train_dataset /mnt/disk3/yusheng/assignment1-basics/data/tokenizer/train_token_ids_10000.bin \
    --val_dataset /mnt/disk3/yusheng/assignment1-basics/data/tokenizer/valid_token_ids_10000.bin \
    --out $OUTDIR \
    --src $SRCDIR \
    --max_learning_rate $LR \
    --min_learning_rate 1e-5 \
    --warmup_iters 4000 \
    --cosine_cycle_iters 40000 \
    --betas "(0.9, 0.95)" \
    --eps 1e-8 \
    --weight_decay 0.1 \
    --max_l2_norm 1.0 \
    --keep_last_n_ckpt 5 \
    --save_every 500
done
