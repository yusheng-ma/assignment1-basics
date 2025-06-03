#!/bin/bash

VOCAB_SIZE=50257
MAX_LEN=256
DMODEL=768
NLAYERS=12
NHEADS=12
DFF=3072
LR=5e-4
EPOCHS=10
OUTDIR=checkpoints/run1
SRCDIR=checkpoints/run1

mkdir -p $OUTDIR

uv run ./cs336_basics/script/train.py \
  --vocab_size $VOCAB_SIZE \
  --max_context_length $MAX_LEN \
  --d_model $DMODEL \
  --num_layers $NLAYERS \
  --num_heads $NHEADS \
  --d_ff $DFF \
  --rope_theta 10000 \
  --lr $LR \
  --num_train_epochs $EPOCHS \
  --out $OUTDIR \
  --src $SRCDIR
