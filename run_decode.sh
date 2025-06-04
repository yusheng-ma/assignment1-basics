#!/bin/bash

VOCAB_SIZE=10000
CONTEXT_LEN=256
DMODEL=768
NLAYERS=12
NHEADS=12
DFF=3072
DEVICE=cpu

VOCAB_PKL="/mnt/disk3/yusheng/assignment1-basics/data/tokenizer/vocab_10000.pkl"
MERGES_PKL="/mnt/disk3/yusheng/assignment1-basics/data/tokenizer/merges_10000.pkl"

CKPT=checkpoints/run1/ckpt_0006.pt

PROMPT="The future of AI is"
MAX_NEW_TOKENS=100
TEMPERATURE=1.0

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
  --temperature $TEMPERATURE
