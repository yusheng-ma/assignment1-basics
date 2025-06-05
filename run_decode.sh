#!/bin/bash

VOCAB_SIZE=10000
CONTEXT_LEN=256
DMODEL=512
NLAYERS=4
NHEADS=16
DFF=1344
DEVICE=cuda

VOCAB_PKL="/mnt/disk3/yusheng/assignment1-basics/data/tokenizer/vocab_10000.pkl"
MERGES_PKL="/mnt/disk3/yusheng/assignment1-basics/data/tokenizer/merges_10000.pkl"

# run_decode.sh
if [ -n "$1" ]; then
    CKPT=$1
else
    CKPT="checkpoints/run1/ckpt_39999.pt"
fi

PROMPT="Once upon a time"
MAX_NEW_TOKENS=256
TEMPERATURE=1.0
TOP_P=0.9

uv run ./cs336_basics/script/decode.py \
  --ckpt $CKPT \
  --prompt "$PROMPT" \
  --device $DEVICE \
  --vocab_pkl $VOCAB_PKL \
  --merges_pkl $MERGES_PKL \
  --vocab_size $VOCAB_SIZE \
  --context_length $CONTEXT_LEN \
  --d_model $DMODEL \
  --num_layers $NLAYERS \
  --num_heads $NHEADS \
  --d_ff $DFF \
  --rope_theta 10000 \
  --max_new_tokens $MAX_NEW_TOKENS \
  --temperature $TEMPERATURE \
  --top_p $TOP_P
