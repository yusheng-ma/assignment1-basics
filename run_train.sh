#!/bin/bash

VOCAB_SIZE=50257
CONTEXT_LEN=256
DMODEL=768
NLAYERS=12
NHEADS=12
DFF=3072
LR=5e-4
EPOCHS=2
BATCH_SIZE=32
DEVICE=cuda
OUTDIR=checkpoints/run1
SRCDIR=checkpoints/run1
DATASET=/mnt/disk3/yusheng/assignment1-basics/data/tokenizer/valid_token_ids_10000.npy
MAX_L2_NORM=1.0

mkdir -p $OUTDIR

uv run ./cs336_basics/script/train.py \
  --vocab_size $VOCAB_SIZE \
  --context_length $CONTEXT_LEN \
  --d_model $DMODEL \
  --num_layers $NLAYERS \
  --num_heads $NHEADS \
  --d_ff $DFF \
  --rope_theta 10000 \
  --lr $LR \
  --num_train_epochs $EPOCHS \
  --batch_size $BATCH_SIZE \
  --device $DEVICE \
  --dataset $DATASET \
  --max_l2_norm $MAX_L2_NORM \
  --out $OUTDIR \
  --src $SRCDIR
