#!/bin/bash

VOCAB_SIZE=32000
CONTEXT_LEN=256
DMODEL=512
NLAYERS=4
NHEADS=16
DFF=1344 # 8/3 * d_model
LR=0.001
EPOCHS=40000
BATCH_SIZE=64
DEVICE=cuda
OUTDIR=checkpoints/owt_muon
SRCDIR=checkpoints/owt_muon
TRAIN_DATASET=./data/tokenizer/train_token_ids_32000.bin
VAL_DATASET=./data/tokenizer/valid_token_ids_32000.bin
KEEP_LAST_N_CKPT=5
SAVE_EVERY=500

# LR scheduler parameters
MAX_LR=0.002
MIN_LR=1e-5
WARMUP=4000
COSINE_ITERS=40000

# AdamW optimizer parameters
BETA1=0.9
BETA2=0.95
EPS=1e-8
WEIGHT_DECAY=0.1
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
  --train_dataset $TRAIN_DATASET \
  --val_dataset $VAL_DATASET \
  --out $OUTDIR \
  --src $SRCDIR \
  --max_learning_rate $MAX_LR \
  --min_learning_rate $MIN_LR \
  --warmup_iters $WARMUP \
  --cosine_cycle_iters $COSINE_ITERS \
  --betas "($BETA1, $BETA2)" \
  --eps $EPS \
  --weight_decay $WEIGHT_DECAY \
  --max_l2_norm $MAX_L2_NORM \
  --keep_last_n_ckpt $KEEP_LAST_N_CKPT \
  --save_every $SAVE_EVERY